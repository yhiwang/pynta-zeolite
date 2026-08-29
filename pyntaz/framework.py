"""Build the bare zeolite: an IZA framework from maze, repeated to a
supercell, one T-site substituted by Al and centred, and the adsorption sites
around that Al (its first-shell oxygens) worked out from connectivity.

The result, a :class:`ZeoliteFramework`, is what every placement step
receives. Sites are plain dicts so they can be dumped to json unchanged.
"""

from itertools import combinations

import numpy as np
import ase
from ase.neighborlist import NeighborList, natural_cutoffs
from maze.zeolite import Zeolite

from . import config
from .geometry import unit


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


def make_supercell(zeolite, min_length=config.SUPERCELL_MIN_LENGTH):
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


def substitute_al_and_center(atoms, index):
    """Turn atom ``index`` into Al and move it to the middle of the cell."""
    atoms[index].symbol = "Al"
    atoms.translate(atoms.cell.sum(axis=0) / 2 - atoms[index].position)
    atoms.wrap()


# --------------------------------------------------------------------------
# the framework object
# --------------------------------------------------------------------------

class ZeoliteFramework:
    """Al-substituted zeolite supercell plus the sites around the Al.

    Attributes
    ----------
    code : str          IZA code ("MOR")
    t_label : str       substituted T-site label ("T4")
    atoms : ase.Atoms   the supercell with the Al in place
    al_index : int      index of the Al atom
    repeats : list      supercell repeats
    t_sites, o_sites    {label: [supercell indices]}
    mono_sites : list   [site dict] one per first-shell oxygen (after find_sites)
    bi_sites : list     [(site, site)] every pair of first-shell oxygens
    """

    def __init__(self, code, atoms, t_label, al_index, repeats,
                 t_sites, o_sites):
        self.code = code
        self.atoms = atoms
        self.t_label = t_label
        self.al_index = al_index
        self.repeats = repeats
        self.t_sites = t_sites
        self.o_sites = o_sites
        self.mono_sites = []
        self.bi_sites = []

    # -- connectivity around the Al ---------------------------------------

    def _neighbor_list(self):
        nl = NeighborList(natural_cutoffs(self.atoms),
                          bothways=True, self_interaction=False)
        nl.update(self.atoms)
        return nl

    def _label_of(self, index, table):
        for label, indices in table.items():
            if index in indices:
                return label
        return None

    def first_shell_oxygens(self, nl=None):
        """{o_index: O label} for the oxygens bonded to the Al."""
        nl = nl or self._neighbor_list()
        neighbors, _ = nl.get_neighbors(self.al_index)
        return {int(i): self._label_of(int(i), self.o_sites)
                for i in neighbors if self.atoms[i].symbol == "O"}

    def first_shell_silicons(self, nl=None):
        """{o_index: si_index} -- the Si on the far side of each first-shell
        oxygen."""
        nl = nl or self._neighbor_list()
        silicon_of = {}
        for o_index in self.first_shell_oxygens(nl):
            neighbors, _ = nl.get_neighbors(o_index)
            for j in neighbors:
                if self.atoms[j].symbol == "Si":
                    silicon_of[o_index] = int(j)
                    break
        return silicon_of

    def second_shell_oxygens(self, nl=None):
        """{si_index: [o_index, ...]} for the framework oxygens on each
        first-shell Si, excluding the first-shell oxygen it came from."""
        nl = nl or self._neighbor_list()
        first_shell = set(self.first_shell_oxygens(nl))
        oxygens_of = {}
        for si_index in set(self.first_shell_silicons(nl).values()):
            neighbors, _ = nl.get_neighbors(si_index)
            oxygens_of[si_index] = sorted(
                int(j) for j in neighbors
                if self.atoms[j].symbol == "O" and int(j) not in first_shell)
        return oxygens_of

    def shell_indices(self, cutoff=5.0):
        """Framework atoms within ``cutoff`` A of the Al (minimum image).
        The complement is what stays frozen in a shell relaxation."""
        distances = self.atoms.get_distances(self.al_index,
                                             range(len(self.atoms)), mic=True)
        return [i for i in range(len(self.atoms)) if distances[i] <= cutoff]

    # -- sites --------------------------------------------------------------

    def monodentate_sites(self):
        """One site dict per first-shell oxygen.

        ``normal`` points out of the framework at the oxygen (bisector of the
        Al->O and Si->O directions); ``tangent_ref`` is the O->Al direction
        projected into the plane perpendicular to the normal and defines
        zero degrees for the monodentate spin sweep."""
        nl = self._neighbor_list()
        oxygens = self.first_shell_oxygens(nl)
        silicon_of = self.first_shell_silicons(nl)
        al_position = self.atoms[self.al_index].position

        sites = []
        for o_index, o_label in oxygens.items():
            o_position = self.atoms[o_index].position
            al_to_o = unit(o_position - al_position)
            si_to_o = unit(o_position - self.atoms[silicon_of[o_index]].position)
            normal = unit(al_to_o + si_to_o)

            o_to_al = al_position - o_position
            tangent_ref = unit(o_to_al - np.dot(o_to_al, normal) * normal)

            sites.append({
                "site": o_label,
                "position": o_position,
                "normal": normal,
                "tangent_ref": tangent_ref,
                "indices": (o_index,),
                "si_index": silicon_of[o_index],
                "al_index": self.al_index,
                "morphology": self.t_label,
                "surface": self.code,
            })
        return sites

    def bridge_sites(self):
        """Midpoint geometry for each pair of first-shell oxygens.

        Not used for placement -- ``place_bidentate`` works from the two
        oxygen positions directly. This is for inspecting a candidate pair:
        the span and the outward direction at its midpoint."""
        al_position = self.atoms[self.al_index].position
        sites = []
        for site_a, site_b in combinations(self.monodentate_sites(), 2):
            position = 0.5 * (site_a["position"] + site_b["position"])
            sites.append({
                "site": "%s-%s" % (site_a["site"], site_b["site"]),
                "position": position,
                "normal": unit(position - al_position),
                "indices": (site_a["indices"][0], site_b["indices"][0]),
                "span": float(np.linalg.norm(site_b["position"]
                                             - site_a["position"])),
            })
        return sites

    def find_sites(self):
        """Fill ``mono_sites`` and ``bi_sites``; returns self."""
        self.mono_sites = self.monodentate_sites()
        self.bi_sites = list(combinations(self.mono_sites, 2))
        return self

    def describe(self):
        for i, site in enumerate(self.mono_sites):
            print("site %02d: O%d (%s) at %s"
                  % (i, site["indices"][0], site["site"],
                     np.round(site["position"], 3)))


def build_framework(code, t_label, min_length=config.SUPERCELL_MIN_LENGTH):
    """Bare zeolite ``code`` with T-site ``t_label`` replaced by Al, sites found.

    maze reads ``./data/<code>.cif`` if it exists and downloads it from the
    IZA database otherwise, so keep ``data/`` next to where you run.
    """
    zeolite = Zeolite.make(code)
    n_unit = len(zeolite)

    t_sites, o_sites = split_site_labels(zeolite)
    supercell, repeats = make_supercell(zeolite, min_length)

    t_sites = expand_site_indices(t_sites, n_unit, len(supercell))
    o_sites = expand_site_indices(o_sites, n_unit, len(supercell))

    if t_label not in t_sites:
        raise ValueError("%s is not a T-site in %s; available: %s"
                         % (t_label, code, list(t_sites)))

    al_index = t_sites[t_label][0]
    substitute_al_and_center(supercell, al_index)

    framework = ZeoliteFramework(code, supercell, t_label, al_index, repeats,
                                 t_sites, o_sites)
    return framework.find_sites()
