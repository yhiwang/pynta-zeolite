"""Build the bare zeolite: an IZA framework from maze, repeated to a
supercell, one or more T-sites substituted by Al, centred on their
midpoint, and the adsorption sites around every Al (its first-shell oxygens) worked
out from connectivity.

``build_framework("MOR", ("T4",))`` gives a single Al. ``unique_pairs``
lists every symmetry-distinct pair of second-order T neighbours (Lowenstein
forbids Al-O-Al, so first-order is excluded) by name, e.g. "T1-T4_5.14",
and ``build_framework(code, t_labels, indices=...)`` builds the one you
pick. Without ``indices`` a second label means its nearest second-order
neighbour.

Sites follow a *site rule*, chosen per framework and stored with it:

* ``"all"``   every first-shell oxygen of every Al is a monodentate site and
              every two of them within ``max_span`` form a pair -- a two-Al
              framework then also carries everything a single-Al run does;
* ``"cross"`` with two Al only the *cross* pairs count -- one oxygen on each
              Al, i.e. the O-Si-O bridges between them -- and only their
              oxygens are monodentate sites, for when the single-Al chemistry
              is already covered by that Al's own run. A single-Al framework
              has no cross pairs, so there the rule behaves like ``"all"``.

``to_dict`` / ``from_dict`` round-trip everything except the atoms
themselves (the caller keeps those as an xyz file next to the json) so
every later step reads the same atoms and the same site list. Sites are
plain dicts so they can be dumped to json unchanged.

maze reads ``./data/<code>.cif`` if it exists and downloads it from the IZA
database otherwise, so keep ``data/`` next to where you run.
"""

from itertools import combinations

import numpy as np
import ase
from ase.neighborlist import NeighborList, natural_cutoffs
from maze.zeolite import Zeolite

from .geometry import unit

T_SYMBOLS = ("Si", "Al")

SUPERCELL_MIN_LENGTH = 12.0     # A, repeat the unit cell until every edge is at least this
SITE_PAIR_MAX_SPAN = 3.5        # A, longest O-O distance that still counts as a site pair
SITE_RULES = ("all", "cross")   # see the module docstring
DEFAULT_SITE_RULE = "all"


# --------------------------------------------------------------------------
# helpers on the maze unit cell
# --------------------------------------------------------------------------

def split_site_labels(zeolite):
    """({T label: [indices]}, {O label: [indices]}) from maze's site table."""
    t_sites, o_sites = {}, {}
    for label in zeolite.get_site_types():
        if label.startswith("T"):
            t_sites[label] = zeolite.site_to_atom_indices[label]
        elif label.startswith("O"):
            o_sites[label] = zeolite.site_to_atom_indices[label]
    return t_sites, o_sites


def make_supercell(zeolite, min_length=SUPERCELL_MIN_LENGTH):
    """(supercell Atoms, repeats) with every cell edge >= ``min_length``."""
    repeats = [int(np.ceil(min_length / length))
               for length in zeolite.cell.cellpar()[:3]]
    return ase.Atoms(zeolite).repeat(repeats), repeats


def expand_site_indices(sites, n_unit, n_super):
    """Map {label: [unit-cell indices]} onto the supercell, where copy k of
    unit atom i is atom ``i + k * n_unit``."""
    n_copies = n_super // n_unit
    return {label: sorted(i + k * n_unit for i in indices
                          for k in range(n_copies))
            for label, indices in sites.items()}


def substitute_al_and_center(atoms, indices):
    """Turn every atom in ``indices`` into Al and move their midpoint
    (minimum image) to the middle of the cell."""
    for i in indices:
        atoms[i].symbol = "Al"
    vectors = atoms.get_distances(indices[0], indices, mic=True, vector=True)
    midpoint = atoms[indices[0]].position + vectors.mean(axis=0)
    atoms.translate(atoms.cell.sum(axis=0) / 2 - midpoint)
    atoms.wrap()


def _site_to_json(site):
    out = {}
    for key, value in site.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, tuple):
            out[key] = [int(v) for v in value]
        elif isinstance(value, np.integer):
            out[key] = int(value)
        elif isinstance(value, np.floating):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def _site_from_json(site):
    out = dict(site)
    for key in ("position", "normal", "tangent_ref"):
        out[key] = np.array(site[key], dtype=float)
    out["indices"] = tuple(site["indices"])
    return out


# --------------------------------------------------------------------------
# the framework object
# --------------------------------------------------------------------------

class ZeoliteFramework:
    """Al-substituted zeolite supercell plus the sites around every Al.

    Attributes
    ----------
    code : str          IZA code ("MOR")
    t_labels : list     substituted T-site labels, in order (["T4", "T1"])
    al_indices : list   the Al atoms, same order; their midpoint is centred
    atoms : ase.Atoms   the supercell with the Al in place
    repeats : list      supercell repeats
    t_sites, o_sites    {label: [supercell indices]}
    max_span : float    longest O-O distance kept in ``bi_sites``
    site_rule : str     "all" or "cross", see the module docstring
    mono_sites : list   [site dict] the monodentate sites (first-shell oxygens)
    bi_sites : list     [(site, site)] pairs of mono sites within max_span
    """

    def __init__(self, code, atoms, t_labels, al_indices, repeats,
                 t_sites, o_sites, site_rule=DEFAULT_SITE_RULE):
        self.code = code
        self.atoms = atoms
        self.t_labels = list(t_labels)
        self.al_indices = [int(i) for i in al_indices]
        self.repeats = repeats
        self.t_sites = t_sites
        self.o_sites = o_sites
        self.max_span = SITE_PAIR_MAX_SPAN
        self.site_rule = self._check_rule(site_rule)
        self.mono_sites = []
        self.bi_sites = []

    @staticmethod
    def _check_rule(site_rule):
        if site_rule not in SITE_RULES:
            raise ValueError("site_rule must be one of %s, got %r"
                             % (SITE_RULES, site_rule))
        return site_rule

    # -- connectivity --------------------------------------------------------

    def _neighbor_list(self):
        nl = NeighborList(natural_cutoffs(self.atoms),
                          bothways=True, self_interaction=False)
        nl.update(self.atoms)
        return nl

    def label_of(self, index):
        """Maze site label of atom ``index`` ("T4", "O7") or None."""
        for table in (self.t_sites, self.o_sites):
            for label, indices in table.items():
                if index in indices:
                    return label
        return None

    def bonded(self, index, symbols, nl):
        """Sorted neighbours of ``index`` whose symbol is in ``symbols``."""
        neighbors, _ = nl.get_neighbors(index)
        return sorted(int(j) for j in neighbors
                      if self.atoms[j].symbol in symbols)

    def first_shell_oxygens(self, al_index, nl=None):
        """{o_index: O label} for the oxygens bonded to ``al_index``."""
        nl = nl or self._neighbor_list()
        return {i: self.label_of(i) for i in self.bonded(al_index, ("O",), nl)}

    def first_shell_silicons(self, al_index, nl=None):
        """{o_index: t_index} -- the T atom on the far side of each
        first-shell oxygen of ``al_index``."""
        nl = nl or self._neighbor_list()
        silicon_of = {}
        for o_index in self.first_shell_oxygens(al_index, nl):
            for j in self.bonded(o_index, T_SYMBOLS, nl):
                if j != al_index:
                    silicon_of[o_index] = j
                    break
        return silicon_of

    def second_shell_oxygens(self, al_index, nl=None):
        """{t_index: [o_index, ...]} for the framework oxygens on each
        first-shell T atom, excluding the first-shell oxygen it came from."""
        nl = nl or self._neighbor_list()
        first_shell = set(self.first_shell_oxygens(al_index, nl))
        oxygens_of = {}
        for t_index in set(self.first_shell_silicons(al_index, nl).values()):
            oxygens_of[t_index] = [o for o in self.bonded(t_index, ("O",), nl)
                                   if o not in first_shell]
        return oxygens_of

    def shell_indices(self, cutoff=5.0):
        """Framework atoms within ``cutoff`` A of any Al (minimum image).
        The complement is what stays frozen in a shell relaxation."""
        n = len(self.atoms)
        keep = set()
        for al_index in self.al_indices:
            distances = self.atoms.get_distances(al_index, range(n), mic=True)
            keep.update(i for i in range(n) if distances[i] <= cutoff)
        return sorted(keep)

    def neighborhood(self, center, other=None, nl=None):
        """Everything around T atom ``center``, all as {index: label}:
        ``first_o``, ``first_t`` (share an O with center), ``second_o``
        (the other O on each first-order T), ``second_t`` (reached through
        a second-order O; first-order T and center removed).

        With ``other`` given also ``other_first_o``, ``bridge_t`` (T atoms
        first-order to both), ``bridge_o`` ({bridge t: {o: label}} the O
        linking that T to center or other) and ``distance``."""
        nl = nl or self._neighbor_list()

        def labels(indices):
            return {i: self.label_of(i) for i in indices}

        first_o = self.bonded(center, ("O",), nl)
        first_t = sorted({j for o in first_o
                          for j in self.bonded(o, T_SYMBOLS, nl) if j != center})
        second_o = sorted({o for t in first_t
                           for o in self.bonded(t, ("O",), nl) if o not in first_o})
        second_t = sorted({j for o in second_o
                           for j in self.bonded(o, T_SYMBOLS, nl)
                           if j != center and j not in first_t})

        hood = {"center": center, "label": self.label_of(center),
                "first_o": labels(first_o), "first_t": labels(first_t),
                "second_o": labels(second_o), "second_t": labels(second_t)}

        if other is not None:
            other_first_o = self.bonded(other, ("O",), nl)
            bridge_t = [t for t in first_t
                        if set(self.bonded(t, ("O",), nl)) & set(other_first_o)]
            hood["other"] = other
            hood["other_label"] = self.label_of(other)
            hood["other_first_o"] = labels(other_first_o)
            hood["bridge_t"] = labels(bridge_t)
            hood["bridge_o"] = {
                t: labels(o for o in self.bonded(t, ("O",), nl)
                          if o in first_o or o in other_first_o)
                for t in bridge_t}
            hood["distance"] = float(self.atoms.get_distance(center, other, mic=True))
        return hood

    def second_order_sites(self, center, label=None, nl=None):
        """[(distance, index)] of the second-order T neighbours of
        ``center``, one per set of symmetry replicas, nearest first.
        ``label`` restricts to one T label. Replicas are recognised by
        equal (label, Al-Al distance)."""
        nl = nl or self._neighbor_list()
        hood = self.neighborhood(center, nl=nl)
        seen = {}
        for j, j_label in hood["second_t"].items():
            if label is not None and j_label != label:
                continue
            if j in self.al_indices:
                continue
            d = round(float(self.atoms.get_distance(center, j, mic=True)), 2)
            key = (j_label, d)
            if key not in seen:
                seen[key] = (d, j)
        return sorted(seen.values())

    # -- sites --------------------------------------------------------------

    def monodentate_sites(self):
        """One site dict per first-shell oxygen of every Al, in Al order.

        ``site`` is "<T label>_<O label>" ("T4_O7"). ``normal`` points out
        of the framework at the oxygen (bisector of the Al->O and Si->O
        directions); ``tangent_ref`` is the O->Al direction projected into
        the plane perpendicular to the normal and defines zero degrees for
        the monodentate spin sweep."""
        nl = self._neighbor_list()
        sites = []
        for al_index, t_label in zip(self.al_indices, self.t_labels):
            oxygens = self.first_shell_oxygens(al_index, nl)
            silicon_of = self.first_shell_silicons(al_index, nl)
            al_position = self.atoms[al_index].position
            for o_index, o_label in oxygens.items():
                o_position = self.atoms[o_index].position
                al_to_o = unit(o_position - al_position)
                si_to_o = unit(o_position - self.atoms[silicon_of[o_index]].position)
                normal = unit(al_to_o + si_to_o)

                o_to_al = al_position - o_position
                tangent_ref = unit(o_to_al - np.dot(o_to_al, normal) * normal)

                sites.append({
                    "site": "%s_%s" % (t_label, o_label),
                    "label": o_label,
                    "position": o_position,
                    "normal": normal,
                    "tangent_ref": tangent_ref,
                    "indices": (o_index,),
                    "si_index": silicon_of[o_index],
                    "al_index": al_index,
                    "t_label": t_label,
                    "framework": self.code,
                })
        return sites

    @staticmethod
    def span(site_a, site_b):
        return float(np.linalg.norm(site_b["position"] - site_a["position"]))

    def site_pairs(self, sites, max_span):
        """[(site, site)] over every pair of ``sites`` closer than ``max_span``."""
        return [(a, b) for a, b in combinations(sites, 2)
                if self.span(a, b) <= max_span]

    def pair_records(self):
        index_of = {id(site): i for i, site in enumerate(self.mono_sites)}
        return [{"sites": [index_of[id(a)], index_of[id(b)]],
                 "names": [a["site"], b["site"]],
                 "span": round(self.span(a, b), 3)}
                for a, b in self.bi_sites]

    def bridge_sites(self):
        """Midpoint geometry for each pair in ``bi_sites``: the span and the
        outward direction at the midpoint (away from the Al, or from the
        Al-Al midpoint for a cross pair). For inspection, not placement."""
        sites = []
        for site_a, site_b in self.bi_sites:
            position = 0.5 * (site_a["position"] + site_b["position"])
            al_position = 0.5 * (self.atoms[site_a["al_index"]].position
                                 + self.atoms[site_b["al_index"]].position)
            sites.append({
                "site": "%s-%s" % (site_a["site"], site_b["site"]),
                "position": position,
                "normal": unit(position - al_position),
                "indices": (site_a["indices"][0], site_b["indices"][0]),
                "span": self.span(site_a, site_b),
            })
        return sites

    def is_cross_pair(self, site_a, site_b):
        """Whether two sites sit on different Al atoms."""
        return site_a["al_index"] != site_b["al_index"]

    def cross_only(self, site_rule=None):
        """Whether the cross rule is in force: rule "cross" *and* more than
        one Al (a single Al has no cross pairs to restrict to)."""
        rule = self.site_rule if site_rule is None else self._check_rule(site_rule)
        return rule == "cross" and len(self.al_indices) > 1

    def find_pairs(self, max_span=SITE_PAIR_MAX_SPAN, sites=None, site_rule=None):
        """[(site, site)] within ``max_span`` by the site rule (the
        framework's own unless ``site_rule`` is given).

        "all": any two of ``sites`` (default: every first-shell oxygen).
        "cross" with several Al: only pairs with one oxygen on each of two
        different Al; within the default span that leaves the O-Si-O
        bridges between the Al.
        """
        sites = self.monodentate_sites() if sites is None else sites
        pairs = self.site_pairs(sites, max_span)
        if self.cross_only(site_rule):
            pairs = [(a, b) for a, b in pairs if self.is_cross_pair(a, b)]
        return pairs

    def find_sites(self, max_span=SITE_PAIR_MAX_SPAN, site_rule=None):
        """Fill ``mono_sites`` and ``bi_sites`` by the site rule (setting
        the framework's rule when ``site_rule`` is given); returns self.

        "all": every first-shell oxygen is a site. "cross" with several
        Al: only the oxygens taking part in a cross pair (:meth:`find_pairs`)
        are sites, so monodentate species land where the two-Al chemistry
        happens and nowhere a single-Al run already put them.
        """
        if site_rule is not None:
            self.site_rule = self._check_rule(site_rule)
        self.max_span = max_span
        oxygens = self.monodentate_sites()
        self.bi_sites = self.find_pairs(max_span, oxygens)
        if self.cross_only():
            if not self.bi_sites:
                raise ValueError("no cross pair within %.2f A between Al %s -- "
                                 "widen max_span or use site_rule='all'"
                                 % (max_span, self.al_indices))
            used = {id(site) for pair in self.bi_sites for site in pair}
            oxygens = [site for site in oxygens if id(site) in used]
        self.mono_sites = oxygens
        return self

    def bridging_si(self, site_a, site_b, nl=None):
        """Si atoms bonded to both oxygens of a pair -- the Si of an O-Si-O
        bridge between two Al. [] for a same-Al pair (they share only
        their Al) or when the oxygens share nothing."""
        nl = nl or self._neighbor_list()
        shared = (set(self.bonded(site_a["indices"][0], T_SYMBOLS, nl))
                  & set(self.bonded(site_b["indices"][0], T_SYMBOLS, nl)))
        return sorted(shared - set(self.al_indices))

    def report(self):
        """Human-readable summary: the Al atoms and every site and pair."""
        nl = self._neighbor_list()
        lines = []
        for al_index, label in zip(self.al_indices, self.t_labels):
            lines.append("Al %s at index %d" % (label, al_index))
        for a, b in combinations(self.al_indices, 2):
            lines.append("Al-Al %d-%d  %.2f A"
                         % (a, b, self.atoms.get_distance(a, b, mic=True)))
        if len(self.al_indices) > 1:
            lines.append("site rule %r: %s" % (self.site_rule, (
                "only cross pairs (one O on each Al) and their oxygens"
                if self.cross_only() else
                "every first-shell O of every Al, every pair within the span")))
        for i, site in enumerate(self.mono_sites):
            lines.append("site %02d: %-7s O%-4d %s"
                         % (i, site["site"], site["indices"][0],
                            np.round(site["position"], 3)))
        for k, (site_a, site_b) in enumerate(self.bi_sites):
            bridge = self.bridging_si(site_a, site_b, nl)
            lines.append("pair %02d: %-7s %-7s %.2f A%s"
                         % (k, site_a["site"], site_b["site"], self.span(site_a, site_b),
                            "  via " + ", ".join("%s(%d)" % (self.label_of(t), t)
                                                 for t in bridge) if bridge else ""))
        return "\n".join(lines)

    # -- serialisation ------------------------------------------------------

    def to_dict(self):
        """Everything but the atoms, as plain json-able values. Store the
        atoms separately (an xyz file) and give both to :meth:`from_dict`."""
        return {"code": self.code,
                "t_labels": self.t_labels,
                "al_indices": self.al_indices,
                "repeats": [int(r) for r in self.repeats],
                "max_span": float(self.max_span),
                "site_rule": self.site_rule,
                "t_sites": self.t_sites,
                "o_sites": self.o_sites,
                "mono_sites": [_site_to_json(site) for site in self.mono_sites],
                "bi_sites": self.pair_records()}

    @classmethod
    def from_dict(cls, record, atoms):
        """Rebuild from :meth:`to_dict` output plus the atoms; sites are read
        back, not recomputed, so indices stay comparable across runs."""
        framework = cls(record["code"], atoms, record["t_labels"],
                        record["al_indices"], record["repeats"],
                        record["t_sites"], record["o_sites"],
                        site_rule=record.get("site_rule", DEFAULT_SITE_RULE))
        framework.max_span = record["max_span"]
        framework.mono_sites = [_site_from_json(site) for site in record["mono_sites"]]
        framework.bi_sites = [(framework.mono_sites[i], framework.mono_sites[j])
                              for i, j in (pair["sites"] for pair in record["bi_sites"])]
        return framework


def bare_supercell(code, min_length=SUPERCELL_MIN_LENGTH):
    """(supercell, repeats, t_sites, o_sites) for IZA ``code``: the pure
    silica framework repeated so every cell edge >= ``min_length``, with the
    maze site tables mapped onto supercell indices. Every framework of one
    code is built from this, so indices are comparable across runs."""
    zeolite = Zeolite.make(code)
    n_unit = len(zeolite)
    t_sites, o_sites = split_site_labels(zeolite)
    supercell, repeats = make_supercell(zeolite, min_length)
    t_sites = expand_site_indices(t_sites, n_unit, len(supercell))
    o_sites = expand_site_indices(o_sites, n_unit, len(supercell))
    return supercell, repeats, t_sites, o_sites


def unique_pairs(code, min_length=SUPERCELL_MIN_LENGTH):
    """{name: record} of every symmetry-distinct second-order T pair in
    ``code``, e.g. ``"T1-T4_5.14": {"t_labels": ["T1", "T4"],
    "indices": [first, second], "distance": 5.14, "repeats": [...]}``.

    One copy of each T label is taken as the first site and its unique
    second-order neighbours (by label and Al-Al distance) as the second.
    A-B and B-A at the same distance are the same pair and kept once, with
    the labels in sorted order. Feed a record to ``build_framework`` as
    ``build_framework(code, record["t_labels"], indices=record["indices"])``.
    """
    supercell, repeats, t_sites, o_sites = bare_supercell(code, min_length)
    framework = ZeoliteFramework(code, supercell, [], [], repeats, t_sites, o_sites)
    nl = framework._neighbor_list()

    pairs = {}
    for label in sorted(t_sites):
        first = t_sites[label][0]
        for d, j in framework.second_order_sites(first, nl=nl):
            other = framework.label_of(j)
            if (other, label) < (label, other):
                continue
            name = "%s-%s_%.2f" % (label, other, d)
            if name not in pairs:
                pairs[name] = {"t_labels": [label, other],
                               "indices": [int(first), int(j)],
                               "distance": d,
                               "repeats": [int(r) for r in repeats]}
    return pairs


def build_framework(code, t_labels, indices=None, min_length=SUPERCELL_MIN_LENGTH,
                    max_span=SITE_PAIR_MAX_SPAN, site_rule=DEFAULT_SITE_RULE):
    """Bare zeolite ``code`` with the T-sites in ``t_labels`` replaced by
    Al, their midpoint centred, sites found (pairs up to ``max_span``, by
    ``site_rule`` -- "all" or "cross", see the module docstring).

    ``t_labels`` is a tuple like ("T4",) or ("T4", "T1"); a bare string is
    taken as one label. The first Al is the first copy of its label; each
    later Al is the nearest second-order T neighbour of the first Al with
    the requested label. ``indices`` (supercell atom indices, same length
    as ``t_labels``) overrides that choice.
    """
    if isinstance(t_labels, str):
        t_labels = (t_labels,)
    t_labels = list(t_labels)

    supercell, repeats, t_sites, o_sites = bare_supercell(code, min_length)

    for label in t_labels:
        if label not in t_sites:
            raise ValueError("%s is not a T-site in %s; available: %s"
                             % (label, code, list(t_sites)))

    if indices is not None:
        al_indices = [int(i) for i in indices]
        if len(al_indices) != len(t_labels):
            raise ValueError("indices and t_labels differ in length")
        substitute_al_and_center(supercell, al_indices)
        return ZeoliteFramework(code, supercell, t_labels, al_indices, repeats,
                                t_sites, o_sites).find_sites(max_span, site_rule)

    first = t_sites[t_labels[0]][0]
    substitute_al_and_center(supercell, [first])
    framework = ZeoliteFramework(code, supercell, t_labels[:1], [first], repeats,
                                 t_sites, o_sites)

    nl = framework._neighbor_list()
    for label in t_labels[1:]:
        options = framework.second_order_sites(first, label, nl)
        if not options:
            raise ValueError("no second-order %s neighbour of %s(%d) in %s"
                             % (label, t_labels[0], first, code))
        _, j = options[0]
        framework.al_indices.append(j)
        framework.t_labels.append(label)

    substitute_al_and_center(framework.atoms, framework.al_indices)
    return framework.find_sites(max_span, site_rule)