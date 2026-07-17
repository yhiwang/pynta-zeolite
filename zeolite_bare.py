import numpy as np
import ase
from copy import deepcopy
from maze.zeolite import Zeolite
from ase.visualize import view
from ase.neighborlist import NeighborList, natural_cutoffs


def get_sites(z):
    t_sites = {}
    o_sites = {}
    for label in z.get_site_types():
        if label.startswith("T"):
            t_sites[label] = z.site_to_atom_indices[label]
        elif label.startswith("O"):
            o_sites[label] = z.site_to_atom_indices[label]
    return t_sites, o_sites


def make_supercell(z, min_length=12.0):
    lengths = z.cell.cellpar()[:3]
    reps = [int(np.ceil(min_length / L)) for L in lengths]
    atoms = ase.Atoms(z).repeat(reps)
    return atoms, reps


def map_indices_to_supercell(sites, n_unit, n_super):
    n_copies = n_super // n_unit
    expanded = {}
    for label, indices in sites.items():
        all_copies = []
        for i in indices:
            for k in range(n_copies):
                all_copies.append(i + k * n_unit)
        expanded[label] = sorted(all_copies)
    return expanded


def Al_substitute_and_center(atoms, index):
    atoms[index].symbol = "Al"
    shift = atoms.cell.sum(axis=0) / 2 - atoms[index].position
    atoms.translate(shift)
    atoms.wrap()


class ZeoliteBare:
    def __init__(self, code, atoms, label, al_index, t_sites_super, o_sites_super):
        self.code = code
        self.atoms = atoms
        self.label = label
        self.al_index = al_index
        self.t_sites_super = t_sites_super
        self.o_sites_super = o_sites_super

    def get_first_degree_oxygens(self):
        nl = NeighborList(natural_cutoffs(self.atoms), bothways=True, self_interaction=False)
        nl.update(self.atoms)
        neighbors, _ = nl.get_neighbors(self.al_index)

        result = {}
        for i in neighbors:
            if self.atoms[i].symbol != "O":
                continue
            for label, indices in self.o_sites_super.items():
                if i in indices:
                    result[int(i)] = label
                    break
        return result

    def get_first_degree_silicon(self):
        # Self
        nl = NeighborList(natural_cutoffs(self.atoms), bothways=True, self_interaction=False)
        nl.update(self.atoms)

        oxygens = self.get_first_degree_oxygens()
        result = {}
        for o_index in oxygens:
            neighbors, _ = nl.get_neighbors(o_index)
            for j in neighbors:
                if self.atoms[j].symbol == "Si":
                    result[o_index] = int(j)
                    break
        return result

    def get_sites(self):
        # ACAT
        oxygens = self.get_first_degree_oxygens()
        silicons = self.get_first_degree_silicon()
        al_pos = self.atoms[self.al_index].position

        sites = []
        for o_index, o_label in oxygens.items():
            o_pos = self.atoms[o_index].position

            al_to_o = o_pos - al_pos
            al_to_o = al_to_o / np.linalg.norm(al_to_o)

            si_pos = self.atoms[silicons[o_index]].position
            si_to_o = o_pos - si_pos
            si_to_o = si_to_o / np.linalg.norm(si_to_o)

            normal = al_to_o + si_to_o
            normal = normal / np.linalg.norm(normal)

            sites.append({
                "site": o_label,
                "position": o_pos,
                "normal": normal,
                "indices": (o_index,),
                "morphology": self.label,
                "surface": self.code,
                "composition": None,
                "subsurf_index": None,
                "subsurf_element": None,
                "label": None,
            })
        return sites
    
    def get_bridge_sites(self):
        # ACAT
        oxygens = self.get_first_degree_oxygens()
        al_pos = self.atoms[self.al_index].position
        o_items = list(oxygens.items())

        sites = []
        for a in range(len(o_items)):
            for b in range(a + 1, len(o_items)):
                o1, label1 = o_items[a]
                o2, label2 = o_items[b]
                p1 = self.atoms[o1].position
                p2 = self.atoms[o2].position

                position = (p1 + p2) / 2
                normal = position - al_pos
                normal = normal / np.linalg.norm(normal)

                sites.append({
                    "site": f"{label1}-{label2}",
                    "position": position,
                    "normal": normal,
                    "indices": (o1, o2),
                    "morphology": self.label,
                    "surface": self.code,
                    "composition": None,
                    "subsurf_index": None,
                    "subsurf_element": None,
                    "label": None,
                })
        return sites
    
    def build_adjacency(self, sites, cutoff):
        # ACAT
        positions = [s["position"] for s in sites]
        adjacency = {}
        for i in range(len(sites)):
            neighbors = []
            for j in range(len(sites)):
                if i == j:
                    continue
                d = np.linalg.norm(positions[i] - positions[j])
                if d <= cutoff:
                    neighbors.append(j)
            adjacency[i] = neighbors
        return adjacency
    
    def get_single_adjacency(self, cutoff=3.0):
        # ACAT but instead of list, we only consider same type
        sites = self.get_sites()
        return self.build_adjacency(sites, cutoff)

    def get_bridge_adjacency(self, cutoff=1.5):
        # ACAT but instead of list, we only consider same type
        # 1.5 -> only bridges sharing an oxygen (gap is between 1.32 and 1.86 Å)
        sites = self.get_bridge_sites()
        return self.build_adjacency(sites, cutoff)
    
    def generate_unique_placements(self):
        # PYNTA but modified, because I don't need the unique sites part
        single_sites = self.get_sites()
        bridge_sites = self.get_bridge_sites()

        by_index = {s["indices"][0]: s for s in single_sites} #Builds a new dictionary with index as key for ez look up

        single_sites_lists = [[s] for s in single_sites]

        double_sites_lists = []
        for bridge in bridge_sites:
            o1, o2 = bridge["indices"]
            double_sites_lists.append([by_index[o1], by_index[o2]])

        single_site_bond_params_lists = []
        for site_list in single_sites_lists:
            pos = deepcopy(site_list[0]["position"])
            single_site_bond_params_lists.append([{"site_pos": pos, "ind": None, "k": 100.0, "deq": 0.0}])

        double_site_bond_params_lists = []
        for pair in double_sites_lists:
            params = []
            for site in pair:
                pos = deepcopy(site["position"])
                params.append({"site_pos": pos, "ind": None, "k": 100.0, "deq": 0.0})
            double_site_bond_params_lists.append(params)

        return single_sites_lists, double_sites_lists, single_site_bond_params_lists, double_site_bond_params_lists
    
    def analyze_zeolite(self):
        # PYNTA analyze_slab mirror: compute the six site outputs and store on self
        self.sites = self.get_sites()
        self.site_adjacency = self.get_single_adjacency()

        (self.single_sites_lists,
         self.double_sites_lists,
         self.single_site_bond_params_lists,
         self.double_site_bond_params_lists) = self.generate_unique_placements()


def make_zeolite_bare(code, label, min_length=12.0):
    z = Zeolite.make(code)
    n_unit = len(z)

    t_sites, o_sites = get_sites(z)
    supercell, reps = make_supercell(z, min_length)

    t_sites_super = map_indices_to_supercell(t_sites, n_unit, len(supercell))
    o_sites_super = map_indices_to_supercell(o_sites, n_unit, len(supercell))

    if label not in t_sites_super:
        raise ValueError(f"{label} not a T-site in {code}; available: {list(t_sites_super)}")

    al_index = t_sites_super[label][0]
    Al_substitute_and_center(supercell, al_index)

    return ZeoliteBare(code, supercell, label, al_index, t_sites_super, o_sites_super)
