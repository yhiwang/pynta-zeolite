import numpy as np
import ase
from itertools import combinations
from maze.zeolite import Zeolite
from ase.neighborlist import NeighborList, natural_cutoffs


def split_site_labels(z):
    t_sites = {}
    o_sites = {}
    for label in z.get_site_types():
        if label.startswith("T"):
            t_sites[label] = z.site_to_atom_indices[label]
        elif label.startswith("O"):
            o_sites[label] = z.site_to_atom_indices[label]
    return t_sites, o_sites


def make_supercell(z, min_length=12.0):
    reps = [int(np.ceil(min_length / L)) for L in z.cell.cellpar()[:3]]
    return ase.Atoms(z).repeat(reps), reps


def map_indices_to_supercell(sites, n_unit, n_super):
    n_copies = n_super // n_unit
    expanded = {}
    for label, indices in sites.items():
        expanded[label] = sorted(i + k * n_unit
                                 for i in indices
                                 for k in range(n_copies))
    return expanded


def Al_substitute_and_center(atoms, index):
    atoms[index].symbol = "Al"
    atoms.translate(atoms.cell.sum(axis=0) / 2 - atoms[index].position)
    atoms.wrap()


class ZeoliteBare:
    def __init__(self, code, atoms, label, al_index, reps,
                 t_sites_super, o_sites_super):
        self.code = code
        self.atoms = atoms
        self.label = label
        self.al_index = al_index
        self.reps = reps
        self.t_sites_super = t_sites_super
        self.o_sites_super = o_sites_super

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

    def get_first_degree_oxygens(self, nl=None):
        nl = nl or self._neighbor_list()
        neighbors, _ = nl.get_neighbors(self.al_index)
        return {int(i): self._label_of(int(i), self.o_sites_super)
                for i in neighbors if self.atoms[i].symbol == "O"}

    def get_first_degree_silicon(self, nl=None):
        nl = nl or self._neighbor_list()
        result = {}
        for o_index in self.get_first_degree_oxygens(nl):
            neighbors, _ = nl.get_neighbors(o_index)
            for j in neighbors:
                if self.atoms[j].symbol == "Si":
                    result[o_index] = int(j)
                    break
        return result

    def get_second_degree_oxygens(self, nl=None):
        """{si_index: [o_index, ...]} for the framework oxygens on each
        first-shell Si, excluding the first-shell oxygen it came from."""
        nl = nl or self._neighbor_list()
        first = set(self.get_first_degree_oxygens(nl))
        result = {}
        for si_index in set(self.get_first_degree_silicon(nl).values()):
            neighbors, _ = nl.get_neighbors(si_index)
            result[si_index] = sorted(int(j) for j in neighbors
                                      if self.atoms[j].symbol == "O"
                                      and int(j) not in first)
        return result

    def shell_indices(self, cutoff=5.0):
        """Framework atoms within `cutoff` of the Al, minimum-image.

        The complement is what stays frozen during a shell relaxation."""
        d = self.atoms.get_distances(self.al_index,
                                     range(len(self.atoms)), mic=True)
        return [i for i in range(len(self.atoms)) if d[i] <= cutoff]

    def get_sites(self):
        nl = self._neighbor_list()
        oxygens = self.get_first_degree_oxygens(nl)
        silicons = self.get_first_degree_silicon(nl)
        al_pos = self.atoms[self.al_index].position

        sites = []
        for o_index, o_label in oxygens.items():
            o_pos = self.atoms[o_index].position

            al_to_o = o_pos - al_pos
            al_to_o /= np.linalg.norm(al_to_o)

            si_to_o = o_pos - self.atoms[silicons[o_index]].position
            si_to_o /= np.linalg.norm(si_to_o)

            normal = al_to_o + si_to_o
            normal /= np.linalg.norm(normal)

            o_to_al = al_pos - o_pos
            tangent_ref = o_to_al - np.dot(o_to_al, normal) * normal
            tangent_ref /= np.linalg.norm(tangent_ref)

            sites.append({
                "site": o_label,
                "position": o_pos,
                "normal": normal,
                "tangent_ref": tangent_ref,
                "indices": (o_index,),
                "si_index": silicons[o_index],
                "al_index": self.al_index,
                "morphology": self.label,
                "surface": self.code,
            })
        return sites

    def get_bridge_sites(self):
        """Midpoint geometry for each pair of first-shell oxygens.

        Not used for placement -- add_adsorbate_to_double_site_z works from
        the two oxygen positions directly. This is for inspecting a candidate
        pair: the span and the outward direction at its midpoint."""
        al_pos = self.atoms[self.al_index].position
        sites = []
        for a, b in combinations(self.get_sites(), 2):
            position = 0.5 * (a["position"] + b["position"])
            normal = position - al_pos
            sites.append({
                "site": "%s-%s" % (a["site"], b["site"]),
                "position": position,
                "normal": normal / np.linalg.norm(normal),
                "indices": (a["indices"][0], b["indices"][0]),
                "span": float(np.linalg.norm(b["position"] - a["position"])),
            })
        return sites

    def generate_unique_placements(self):
        single_sites = self.get_sites()
        single_sites_lists = [[s] for s in single_sites]
        double_sites_lists = [[a, b] for a, b in combinations(single_sites, 2)]
        return single_sites_lists, double_sites_lists

    def analyze_zeolite(self):
        (self.single_sites_lists,
         self.double_sites_lists) = self.generate_unique_placements()


def make_zeolite_bare(code, label, min_length=12.0):
    z = Zeolite.make(code)
    n_unit = len(z)

    t_sites, o_sites = split_site_labels(z)
    supercell, reps = make_supercell(z, min_length)

    t_sites_super = map_indices_to_supercell(t_sites, n_unit, len(supercell))
    o_sites_super = map_indices_to_supercell(o_sites, n_unit, len(supercell))

    if label not in t_sites_super:
        raise ValueError("%s not a T-site in %s; available: %s"
                         % (label, code, list(t_sites_super)))

    al_index = t_sites_super[label][0]
    Al_substitute_and_center(supercell, al_index)

    return ZeoliteBare(code, supercell, label, al_index, reps,
                       t_sites_super, o_sites_super)