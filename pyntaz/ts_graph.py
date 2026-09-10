"""Step 5 of the workflow: transition-state guesses built straight from the
reaction graph, without any relaxed adsorbate.

How it works
------------
:class:`TSGraph` merges the reactant and product adjacency lists of one
reaction into a single graph that carries every bond of either side and
remembers which ones form, break or change order on the way across. That
graph is an adjacency list like any other, so :class:`~pyntaz.adjlist.
AdjacencyStructure` can build 3D coordinates for it; the forming and
breaking bonds can be lengthened through ``bond_scales``.

The graph's ``X`` sites are seated on real framework oxygens by
:class:`PairSweep`. In stage one every torsion x axial combination is built
at a series of site-bond scales and the scale whose X-X span best matches
the O-O distance is kept, provided it lands within a tolerance. In stage
two each survivor is placed with its X atoms on the two oxygens and rolled
about the O-O axis; every roll is scored by the clearance of its tail from
the framework.

Only two-site reactions (two X in the graph) are handled; a single-site
seating needs the site normal instead of an O-O axis and is not written yet.
"""

from collections import namedtuple
from itertools import combinations, product

import numpy as np
from ase import Atoms

from .adjlist import AdjacencyStructure
from .geometry import min_clearance, rotate_vector, unit

ORDER_STR = {1.0: "S", 1.5: "B", 2.0: "D", 3.0: "T"}

DEFAULT_STRETCH = {"form": 1.0, "break": 1.0}   # bond length factors by change
TORSION_STEPS = 24          # steps over 360 deg for every free torsion (15 deg)
ROLL_STEPS = 24             # steps over 360 deg about the O-O axis (15 deg)
SPAN_TOLERANCE = 0.2        # A, |fitted X-X span - O-O distance| allowed
SITE_SCALE_RANGE = (1.0, 2.0, 0.05)     # X-atom bond scale: start, stop, step


# --------------------------------------------------------------------------
# the merged reaction graph
# --------------------------------------------------------------------------

def bond_map(mol):
    """{(i, j): order} with i < j over every bond of an RMG molecule."""
    index = {id(atom): i for i, atom in enumerate(mol.atoms)}
    return {tuple(sorted((index[id(atom)], index[id(other)]))): bond.get_order_num()
            for atom in mol.atoms for other, bond in atom.bonds.items()}


class TSGraph:
    """Reactant and product of one reaction merged into one graph.

    Both sides must number the same atoms the same way (the convention of
    ``reaction.yaml``). Every bond present on either side is kept, at the
    higher of its two orders, and ``bonds[(i, j)]`` records ``reactant`` /
    ``product`` orders, the ``change`` ("form", "break", "order" or "none")
    and the length ``stretch`` factor that change earns from ``stretch``.
    """

    def __init__(self, reactant, product, stretch=None):
        stretch = DEFAULT_STRETCH if stretch is None else stretch
        if len(reactant.atoms) != len(product.atoms):
            raise ValueError("sides have %d and %d atoms"
                             % (len(reactant.atoms), len(product.atoms)))
        self.elements = [atom.element.symbol for atom in reactant.atoms]
        self.stars = {i: atom.label for i, atom in enumerate(reactant.atoms)
                      if atom.label}
        self.flags = [(atom.radical_electrons, atom.lone_pairs, atom.charge)
                      for atom in reactant.atoms]
        self.multiplicity = reactant.multiplicity
        self.adj = {i: {} for i in range(len(self.elements))}
        self.bonds = {}
        before_map, after_map = bond_map(reactant), bond_map(product)
        for i, j in sorted(set(before_map) | set(after_map)):
            before, after = before_map.get((i, j)), after_map.get((i, j))
            change = ("form" if before is None else
                      "break" if after is None else
                      "order" if before != after else "none")
            order = max(o for o in (before, after) if o is not None)
            self.bonds[(i, j)] = {"order": order, "reactant": before,
                                  "product": after, "change": change,
                                  "stretch": stretch.get(change, 1.0)}
            self.adj[i][j] = self.adj[j][i] = order

    def tag(self, i):
        """"C1*1" style name of atom ``i``: element, 1-based label, star."""
        return "%s%d%s" % (self.elements[i], i + 1, self.stars.get(i, ""))

    def sites(self):
        """Graph indices of the X atoms, in order."""
        return sorted(i for i, element in enumerate(self.elements) if element == "X")

    def changes(self):
        """[(change, i, j, reactant order, product order, stretch)] for every
        bond that is not the same on both sides."""
        return [(info["change"], i, j, info["reactant"], info["product"],
                 info["stretch"])
                for (i, j), info in sorted(self.bonds.items())
                if info["change"] != "none"]

    def bond_scales(self):
        """{(label, label): factor} in adjacency-list (1-based) labels for
        every bond whose stretch is not 1, ready for AdjacencyStructure."""
        return {(i + 1, j + 1): info["stretch"]
                for (i, j), info in self.bonds.items()
                if info["stretch"] != 1.0}

    def adjlist(self):
        """The merged graph as an RMG adjacency list."""
        lines = ["multiplicity %d" % self.multiplicity]
        for i, symbol in enumerate(self.elements):
            u, p, c = self.flags[i]
            bonds = " ".join("{%d,%s}" % (j + 1, ORDER_STR[order])
                             for j, order in sorted(self.adj[i].items()))
            lines.append(" ".join(part for part in (
                str(i + 1), self.stars.get(i, ""), symbol,
                "u%d p%d c%d" % (u, p, c), bonds) if part))
        return "\n".join(lines)

    def report(self):
        """Human-readable summary of the bond changes."""
        lines = []
        for change, i, j, before, after, stretch in self.changes():
            lines.append("  %-6s %-14s %s -> %s   x%.2f"
                         % (change, "%s-%s" % (self.tag(i), self.tag(j)),
                            ORDER_STR.get(before, "-"), ORDER_STR.get(after, "-"),
                            stretch))
        return "\n".join(lines)


# --------------------------------------------------------------------------
# which torsions and axial choices are worth sweeping
# --------------------------------------------------------------------------

def axial_options(struct):
    """{label: [(a, b), ...]} -- every 2-combination of a five-coordinate
    atom's neighbours, i.e. every candidate axial pair."""
    return {label: list(combinations(neighbors, 2))
            for label, neighbors in struct.hypervalent_atoms().items()}


def effective_torsions(struct):
    """Rotatable bonds whose rotation actually moves the geometry.

    A bond only counts when each end carries at least one non-X neighbour
    besides the other end -- otherwise there is nothing real hanging off
    that side to rotate (a bond to a lone migrating H whose only other
    neighbour is a site X is not a usable torsion). Also skips single-atom
    moving sides and symmetric tops (methyl and the like).
    """
    keep = []
    for key, info in struct.rotatable_bonds().items():
        a, b = key
        a_real = [n for n in struct.neighbors[a]
                  if n != b and struct.element(n) != "X"]
        b_real = [n for n in struct.neighbors[b]
                  if n != a and struct.element(n) != "X"]
        if not a_real or not b_real:
            continue
        moves = info["moves"]
        if len(moves) == 1:
            continue
        hub = key[1]
        others = [m for m in moves if m != hub]
        if others and all(struct.element(m) == "H"
                          and struct.neighbors[m] == [hub] for m in others):
            continue
        keep.append(key)
    return keep


# --------------------------------------------------------------------------
# building and seating one geometry
# --------------------------------------------------------------------------

def built_positions(struct, n_atoms):
    """(n_atoms, 3) positions of a built structure in graph order, X sites
    at their anchor positions."""
    out = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        label = i + 1
        if label in struct.site_anchors:
            out[i] = struct.site_anchors[label]["position"]
        else:
            out[i] = struct.positions[label]
    return out


def build_at(adjlist, site_elements, scale, n_atoms, torsions=None, axial=None,
             bond_scales=None):
    """Positions of the graph built at one site-bond ``scale``."""
    struct = AdjacencyStructure.from_adjlist(adjlist, site_elements=site_elements,
                                             site_bond_scale=scale,
                                             bond_scales=bond_scales)
    struct.build(torsions=torsions, axial=axial)
    return built_positions(struct, n_atoms)


def scale_values(scale_range=SITE_SCALE_RANGE):
    """The site-bond scales tried by :func:`fit_site_scale`, from a
    (start, stop, step) triple, stop included."""
    start, stop, step = scale_range
    return np.arange(start, stop + 0.5 * step, step)


def fit_site_scale(adjlist, site_elements, n_atoms, index_a, index_b,
                   position_a, position_b, torsions=None, axial=None,
                   bond_scales=None, scale_range=SITE_SCALE_RANGE):
    """(scale, span, error): the site-bond scale at which the distance
    between graph atoms ``index_a`` and ``index_b`` comes closest to the
    distance between ``position_a`` and ``position_b``."""
    target = float(np.linalg.norm(np.asarray(position_a) - np.asarray(position_b)))
    best = None
    for scale in scale_values(scale_range):
        positions = build_at(adjlist, site_elements, float(scale), n_atoms,
                             torsions, axial, bond_scales)
        span = float(np.linalg.norm(positions[index_a] - positions[index_b]))
        error = abs(span - target)
        if best is None or error < best[2]:
            best = (float(scale), span, error)
    return best


def seat(positions, index_a, index_b, target_a, target_b, roll_deg):
    """Rigidly move ``positions`` so atoms ``index_a`` / ``index_b`` straddle
    ``target_a`` / ``target_b`` (midpoints coincide, axes aligned), then
    roll everything by ``roll_deg`` about the target axis."""
    axis_from = positions[index_b] - positions[index_a]
    axis_to = np.asarray(target_b) - np.asarray(target_a)
    moved = positions - 0.5 * (positions[index_a] + positions[index_b])

    cross = np.cross(unit(axis_from), unit(axis_to))
    if np.linalg.norm(cross) > 1e-8:
        angle = np.degrees(np.arcsin(np.clip(np.linalg.norm(cross), -1.0, 1.0)))
        if np.dot(unit(axis_from), unit(axis_to)) < 0:
            angle = 180.0 - angle
        moved = np.array([rotate_vector(p, unit(cross), angle) for p in moved])

    moved = np.array([rotate_vector(p, unit(axis_to), roll_deg) for p in moved])
    return moved + 0.5 * (np.asarray(target_a) + np.asarray(target_b))


# --------------------------------------------------------------------------
# the sweep for one reaction on one pair of oxygens
# --------------------------------------------------------------------------

# angles: torsion angle per key of PairSweep.torsion_keys (deg, rounded)
# axial:  {label: (a, b)} axial pair per five-coordinate atom
# scale, span, error: fitted site-bond scale, the X-X span it gives, |span - d(O-O)|
Survivor = namedtuple("Survivor", "angles axial scale span error")


def guess_stem(flip, angles, axial_index, roll, has_axial=False):
    """Directory / file stem of one guess: ``flip0_tor045-120_ax1_roll090``.
    The torsion part is left out when there are no torsions, the axial part
    unless the graph has five-coordinate atoms. Zero-padded to sort."""
    parts = ["flip%d" % flip]
    if angles:
        parts.append("tor" + "-".join("%03d" % round(a) for a in angles))
    if has_axial:
        parts.append("ax%d" % axial_index)
    parts.append("roll%03d" % round(roll))
    return "_".join(parts)


class PairSweep:
    """One :class:`TSGraph` seated on one ordered pair of framework oxygens.

    ``oxygens`` = (index for the first X, index for the second X) into
    ``framework``; swap them for the other way round.

    Attributes worth reading: ``torsion_keys`` (the free torsions swept),
    ``axial_labels`` / ``axial_grid`` (the axial choices), ``target_span``
    (the O-O distance) and ``n_builds`` (stage-one cost).
    """

    def __init__(self, ts, framework, oxygens, torsion_steps=TORSION_STEPS,
                 roll_steps=ROLL_STEPS, span_tolerance=SPAN_TOLERANCE,
                 scale_range=SITE_SCALE_RANGE):
        sites = ts.sites()
        if len(sites) != 2:
            raise NotImplementedError("PairSweep needs a two-site reaction, "
                                      "this one has %d X" % len(sites))
        self.ts = ts
        self.framework = framework
        self.site_a, self.site_b = sites
        self.oxygen_a, self.oxygen_b = oxygens
        self.torsion_steps = torsion_steps
        self.roll_steps = roll_steps
        self.span_tolerance = span_tolerance
        self.scale_range = scale_range

        self.target_a = framework.positions[self.oxygen_a]
        self.target_b = framework.positions[self.oxygen_b]
        self.target_span = float(np.linalg.norm(self.target_a - self.target_b))
        self.site_elements = {self.site_a + 1: framework[self.oxygen_a].symbol,
                              self.site_b + 1: framework[self.oxygen_b].symbol}

        self.adjlist = ts.adjlist()
        self.bond_scales = ts.bond_scales()
        self.n_atoms = len(ts.elements)
        survey = AdjacencyStructure.from_adjlist(self.adjlist)
        self.rotatable = survey.rotatable_bonds()
        self.torsion_keys = sorted(effective_torsions(survey))
        options = axial_options(survey)
        self.axial_labels = sorted(options)
        self.axial_grid = list(product(*(options[label] for label in self.axial_labels)))

        # real atoms, and the ones free to swing: neither a site nor bonded to one
        self.keep = [i for i, element in enumerate(ts.elements) if element != "X"]
        binders = {b for x in sites for b in ts.adj[x]}
        self.tail = [i for i in self.keep if i not in binders]

    @property
    def n_builds(self):
        return self.torsion_steps ** len(self.torsion_keys) * len(self.axial_grid)

    def axial_index(self, axial):
        """Position of an axial choice in ``axial_grid``."""
        return self.axial_grid.index(tuple(axial[label] for label in self.axial_labels))

    def _positions(self, angles, axial, scale):
        torsions = {key: angle for key, angle in zip(self.torsion_keys, angles)}
        return build_at(self.adjlist, self.site_elements, scale, self.n_atoms,
                        torsions, axial, self.bond_scales)

    def survivors(self):
        """Stage one: every torsion x axial combination whose fitted X-X
        span lands within ``span_tolerance`` of the O-O distance, as
        :class:`Survivor` records sorted best fit first."""
        step = 360.0 / self.torsion_steps
        survivors = []
        for combo in product(range(self.torsion_steps), repeat=len(self.torsion_keys)):
            torsions = {key: k * step for key, k in zip(self.torsion_keys, combo)}
            angles = tuple(round(angle) for angle in torsions.values())
            for axial_combo in self.axial_grid:
                axial = {label: pair for label, pair in zip(self.axial_labels, axial_combo)}
                scale, span, error = fit_site_scale(
                    self.adjlist, self.site_elements, self.n_atoms,
                    self.site_a, self.site_b, self.target_a, self.target_b,
                    torsions, axial, self.bond_scales, self.scale_range)
                if error <= self.span_tolerance:
                    survivors.append(Survivor(angles, axial, scale, span, error))
        survivors.sort(key=lambda s: s.error)
        return survivors

    def rolls(self, survivor):
        """Stage two: seat one survivor on the oxygens and roll it about
        the O-O axis. Returns [(roll_deg, clearance, molecule)] over every
        roll, where ``molecule`` holds the real atoms (no X) in the
        framework's cell and ``molecule.info`` records the guess."""
        positions = self._positions(survivor.angles, survivor.axial, survivor.scale)
        cell, pbc = self.framework.cell, self.framework.pbc
        symbols = [self.ts.elements[i] for i in self.keep]
        out = []
        for k in range(self.roll_steps):
            roll = k * (360.0 / self.roll_steps)
            placed = seat(positions, self.site_a, self.site_b,
                          self.target_a, self.target_b, roll)
            clearance = min_clearance(placed[self.tail], self.framework.positions,
                                      cell, pbc)
            molecule = Atoms(symbols, positions=placed[self.keep], cell=cell, pbc=pbc)
            molecule.info.update({"angles": list(survivor.angles),
                                  "axial": str(survivor.axial),
                                  "scale": survivor.scale,
                                  "span": round(survivor.span, 3),
                                  "roll": round(roll),
                                  "clearance": round(clearance, 3)})
            out.append((roll, clearance, molecule))
        return out
