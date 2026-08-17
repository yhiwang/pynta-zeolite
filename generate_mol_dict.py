import yaml

from molecule.molecule import Molecule
from pynta.mol import get_name


def load_reactions(path):
    with open(path) as f:
        targets = yaml.safe_load(f)

    rxns_list = [v for v in targets if "reactant" in v]
    spcs_list = [v for v in targets if "reactant" not in v]
    for i, r in enumerate(rxns_list):
        r["index"] = i
    return rxns_list, spcs_list


def match_name(mol, mol_dict):
    for name, unique in mol_dict.items():
        if unique is mol or unique.is_isomorphic(mol, save_order=True):
            return name, unique
    raise ValueError("no unique species matches\n%s" % mol.to_adjacency_list())


def generate_mol_dict(rxns_list, spcs_list,
                      calculate_thermodynamic_references=False):
    if calculate_thermodynamic_references:
        mols = [Molecule().from_smiles(sm) for sm in ["[H][H]", "O", "C", "N"]]
    else:
        mols = []

    for r in spcs_list:
        mol = Molecule().from_adjacency_list(r["molecule"])
        mol.multiplicity = mol.get_radical_count() + 1
        mols.append(mol)

    for r in rxns_list:
        r["reactant_mols"] = []
        r["product_mols"] = []
        for key, side in (("reactant_mols", r["reactant"]),
                          ("product_mols", r["product"])):
            whole = Molecule().from_adjacency_list(side)
            whole.clear_labeled_atoms()
            for mol in whole.split():
                mol.multiplicity = mol.get_radical_count() + 1
                if not mol.is_surface_site():
                    mols.append(mol)
                    r[key].append(mol)

    unique_mols = []
    for mol in mols:
        if not any(mol.is_isomorphic(m) for m in unique_mols):
            unique_mols.append(mol)

    mol_dict = {get_name(mol): mol for mol in unique_mols}
    name_to_adjlist_dict = {name: mol.to_adjacency_list()
                            for name, mol in mol_dict.items()}

    for r in rxns_list:
        for key, name_key in (("reactant_mols", "reactant_names"),
                              ("product_mols", "product_names")):
            r[name_key] = []
            for i, mol in enumerate(r[key]):
                name, unique = match_name(mol, mol_dict)
                r[key][i] = unique
                r[name_key].append(name)
            r[key] = [m.to_adjacency_list() for m in r[key]]

    return mol_dict, name_to_adjlist_dict, rxns_list