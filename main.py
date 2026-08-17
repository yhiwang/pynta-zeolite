from generate_mol_dict import load_reactions, generate_mol_dict
from adsorbate import construct_initial_guess_files


class PyntaZ:
    def __init__(self, zeolite_bare, rxns_file,
                 calculate_thermodynamic_references=False):
        self.zeolite = zeolite_bare
        self.rxns_file = rxns_file
        self.calculate_thermodynamic_references = calculate_thermodynamic_references
        self.rxns_list, self.spcs_list = load_reactions(rxns_file)

        self.mol_dict = None
        self.name_to_adjlist_dict = None
        self.adsorbate_xyz_dict = None

    def generate_mol_dict(self):
        (self.mol_dict,
         self.name_to_adjlist_dict,
         self.rxns_list) = generate_mol_dict(
            self.rxns_list, self.spcs_list,
            self.calculate_thermodynamic_references)

    def setup_adsorbates(self, pynta_path):
        if self.mol_dict is None:
            self.generate_mol_dict()

        self.zeolite.analyze_zeolite()
        slab = self.zeolite.atoms

        self.adsorbate_xyz_dict = {}
        for name, mol in self.mol_dict.items():
            self.adsorbate_xyz_dict[name] = construct_initial_guess_files(
                mol, name, pynta_path, slab,
                self.zeolite.single_sites_lists,
                self.zeolite.double_sites_lists)
        return self.adsorbate_xyz_dict