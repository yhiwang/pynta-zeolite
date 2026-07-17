from molecule.molecule import Molecule
from pynta.mol import get_name


def generate_mol_dict(rxns_list, spcs_list, calculate_thermodynamic_references=True):
    if calculate_thermodynamic_references: #force inclusion of H2, H2O, CH4 and NH3 for thermochemistry referencing
        mols = [Molecule().from_smiles(sm) for sm in ["[H][H]","O","C","N"]]
    else:
        mols = []

    for r in spcs_list:
        mol = Molecule().from_adjacency_list(r["molecule"])
        mol.multiplicity = mol.get_radical_count() + 1
        mols.append(mol)

    for r in rxns_list:
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
    name_to_adjlist_dict = {sm:mol.to_adjacency_list() for sm,mol in mol_dict.items()}

    for r in rxns_list:
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

    return mol_dict, name_to_adjlist_dict, rxns_list