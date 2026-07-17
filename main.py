import yaml
from zeolite_bare import ZeoliteBare, make_zeolite_bare
from molecule.molecule import Molecule
from pynta.mol import get_name, get_adsorbate
from adsorbate import construct_initial_guess_files


class PyntaZ:
    def __init__(self, zeolite_bare, rxns_file, calculate_thermodynamic_references=False):
        self.zeolite = zeolite_bare
        self.rxns_file = rxns_file
        self.calculate_thermodynamic_references = calculate_thermodynamic_references

        with open(rxns_file) as f:
            targets = yaml.safe_load(f)
        self.rxns_list = [v for v in targets if "reactant" in v]
        self.spcs_list = [v for v in targets if "reactant" not in v]
        for i, r in enumerate(self.rxns_list):
            r["index"] = i
    
    def generate_mol_dict(self):
        # This is the exact same as pynta
        """
        generates all unique Molecule objects based on the reactions and generates a dictionary
        mapping smiles to each unique Molecule object
        also updates self.rxns_list and self.spcs_list with addtional useful information for each reaction
        """
        if self.calculate_thermodynamic_references: #force inclusion of H2, H2O, CH4 and NH3 for thermochemistry referencing
            mols = [Molecule().from_smiles(sm) for sm in ["[H][H]","O","C","N"]]
        else:
            mols = []
        
        for r in self.spcs_list:
            mol = Molecule().from_adjacency_list(r["molecule"])
            mol.multiplicity = mol.get_radical_count() + 1
            mols.append(mol)
        
        for r in self.rxns_list:
            r["reactant_mols"] = []
            r["product_mols"] = []
            react = Molecule().from_adjacency_list(r["reactant"])
            prod = Molecule().from_adjacency_list(r["product"])
            react.clear_labeled_atoms()
            prod.clear_labeled_atoms()
            for mol in react.split():
                mol.multiplicity = mol.get_radical_count() + 1
                if not mol.is_surface_site():
                    mols.append(mol)
                    r["reactant_mols"].append(mol)
            for mol in prod.split():
                mol.multiplicity = mol.get_radical_count() + 1
                if not mol.is_surface_site():
                    mols.append(mol)
                    r["product_mols"].append(mol)

        unique_mols = []
        for mol in mols:
            for m in unique_mols:
                if mol.is_isomorphic(m):
                    break
            else:
                unique_mols.append(mol)

        for mol in unique_mols:
            mol.multiplicity = mol.get_radical_count() + 1

        mol_dict = {get_name(mol):mol for mol in unique_mols}
        self.mol_dict = mol_dict
        self.name_to_adjlist_dict = {sm:mol.to_adjacency_list() for sm,mol in mol_dict.items()}


        for r in self.rxns_list:
            r["reactant_names"] = []
            r["product_names"] = []

            for i,rmol in enumerate(r["reactant_mols"]):
                for sm,mol in mol_dict.items():
                    if mol is rmol or mol.is_isomorphic(rmol,save_order=True):
                        r["reactant_mols"][i] = mol
                        r["reactant_names"].append(sm)
                        break
                else:
                    print("rmol")
                    print(rmol.to_adjacency_list())
                    raise ValueError

            for i,rmol in enumerate(r["product_mols"]):
                for sm,mol in mol_dict.items():
                    if mol is rmol or mol.is_isomorphic(rmol,save_order=True):
                        r["product_mols"][i] = mol
                        r["product_names"].append(sm)
                        break
                else:
                    print("rmol")
                    print(rmol.to_adjacency_list())
                    raise ValueError

            r["reactant_mols"] = [x.to_adjacency_list() for x in r["reactant_mols"]]
            r["product_mols"] = [x.to_adjacency_list() for x in r["product_mols"]]
    
    def generate_atom_maps(self):
        # This is the exact same as pynta
        gratom_to_molecule_atom_maps = dict()
        gratom_to_molecule_surface_atom_maps = dict()
        for sm,mol in self.mol_dict.items():
            ads,mol_to_atoms_map = get_adsorbate(mol)
            
            gratom_to_molecule_atom_maps[sm] = {val:key for key,val in mol_to_atoms_map.items()}

            surf_index_atom_map = dict()
            for i,atm in enumerate(mol.atoms):
                if atm.is_bonded_to_surface():
                    surf_index_atom_map[mol_to_atoms_map[i]] = i

            gratom_to_molecule_surface_atom_maps[sm] = surf_index_atom_map

        self.gratom_to_molecule_atom_maps = gratom_to_molecule_atom_maps
        self.gratom_to_molecule_surface_atom_maps = gratom_to_molecule_surface_atom_maps
    
    def setup_adsorbates(self, pynta_path, Eharmtol=3.0, Eharmfiltertol=30.0, Nharmmin=6,
                         harm_f_software="TBLite", harm_f_software_kwargs=None, nprocs=1):

        if harm_f_software_kwargs is None:
            harm_f_software_kwargs = dict()

        # pull the analyzed site data from the ZeoliteBare
        self.zeolite.analyze_zeolite()
        slab = self.zeolite.atoms
        nslab = len(slab)
        pbc = slab.pbc

        self.adsorbate_xyz_dict = dict()
        for sm, mol in self.mol_dict.items():
            xyzs = construct_initial_guess_files(
                mol, sm, pynta_path, slab,
                self.zeolite.single_site_bond_params_lists,
                self.zeolite.single_sites_lists,
                self.zeolite.double_site_bond_params_lists,
                self.zeolite.double_sites_lists,
                Eharmtol, Eharmfiltertol, Nharmmin,
                self.zeolite.sites, self.zeolite.site_adjacency,
                pbc, nslab,
                harm_f_software, harm_f_software_kwargs,
                nprocs,
            )
            self.adsorbate_xyz_dict[sm] = xyzs