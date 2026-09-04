"""Helpers vendored from pynta.mol (get_adsorbate, get_name).

Copied verbatim from pynta/mol.py so that pyntaz has no pynta dependency.
Only the imports differ: acat, pynta.utils and pynta.calculator are not
needed by these functions, and rdkit.AllChem is imported explicitly rather
than relied on being pulled in by another module.

Do not "improve" these -- get_adsorbate's mol_to_atoms_map is written into
info.json as atom_to_molecule_atom_map and read back by step 6, and
get_name's output is the on-disk directory name.
"""

from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem


def get_desorbed_with_map(mol):
    molcopy = mol.copy(deep=True)
    init_map = {i: a for i, a in enumerate(molcopy.atoms)}
    for bd in molcopy.get_all_edges():
        if bd.atom1.is_surface_site():
            bd.atom2.radical_electrons += round(bd.order)
            molcopy.remove_bond(bd)
            molcopy.remove_atom(bd.atom1)
        elif bd.atom2.is_surface_site():
            bd.atom1.radical_electrons += round(bd.order)
            molcopy.remove_bond(bd)
            molcopy.remove_atom(bd.atom2)
    molcopy.sort_atoms()
    out_map = {i: molcopy.atoms.index(a)
               for i, a in init_map.items() if a in molcopy.atoms}
    return molcopy, out_map


def get_conformer(desorbed):
    try:
        rdmol, rdmap = desorbed.to_rdkit_mol(remove_h=False, return_mapping=True)
    except Exception as e:
        syms = [a.symbol for a in desorbed.atoms]
        indmap = {i: i for i in range(len(desorbed.atoms))}
        if len(desorbed.atoms) == 1:
            atoms = Atoms(syms[0], positions=[(0, 0, 0)])
            return atoms, indmap
        elif len(desorbed.atoms) == 2:
            atoms = Atoms(syms[0] + syms[1], positions=[(0, 0, 0), (1.3, 0, 0)])
            return atoms, indmap
        else:
            raise e
    indmap = {i: rdmap[a] for i, a in enumerate(desorbed.atoms)}
    AllChem.EmbedMultipleConfs(rdmol, numConfs=1, randomSeed=1)
    conf = rdmol.GetConformer()
    pos = conf.GetPositions()
    syms = [a.GetSymbol() for a in rdmol.GetAtoms()]
    atoms = Atoms(symbols=syms, positions=pos)
    return atoms, indmap


def get_adsorbate(mol):
    desorbed, mol_to_desorbed_map = get_desorbed_with_map(mol)
    atoms, desorbed_to_atoms_map = get_conformer(desorbed)
    mol_to_atoms_map = {key: desorbed_to_atoms_map[val]
                        for key, val in mol_to_desorbed_map.items()}
    return atoms, mol_to_atoms_map


def get_name(mol):
    try:
        return mol.to_smiles()
    except Exception:
        return mol.to_adjacency_list().replace("\n", " ")[:-1].replace(' ', '')