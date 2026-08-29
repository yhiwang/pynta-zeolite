"""Read ``reaction.yaml`` into RMG molecules.

The yaml holds reactions (RMG adjacency lists for the whole reactant and
product side, ``X`` marking a framework site) and, optionally, standalone
species. Every distinct molecule that appears on any side becomes a species
named by pynta's ``get_name`` (e.g. ``C[CH2][Pt]``); each reaction records
which species make up its sides.
"""

from collections import namedtuple

import yaml
from molecule.molecule import Molecule
from pynta.mol import get_name

# species:   {name: Molecule}           one per unique molecule
# adjlists:  {name: adjacency list str} the same, serialised
# reactions: [dict]                     yaml records plus reactant_names/product_names,
#                                       reactant_mols/product_mols (adjacency lists)
ReactionSet = namedtuple("ReactionSet", "species adjlists reactions")

THERMO_REFERENCE_SMILES = ("[H][H]", "O", "C", "N")


# --------------------------------------------------------------------------
# molecule helpers
# --------------------------------------------------------------------------

def molecule_from_adjlist(adjlist):
    """Molecule from an adjacency list with the multiplicity set from its
    radical count (RMG leaves it at the default otherwise)."""
    mol = Molecule().from_adjacency_list(adjlist)
    mol.multiplicity = mol.get_radical_count() + 1
    return mol


def is_surface_species(mol):
    """True when the molecule contains at least one framework site X."""
    return any(atom.is_surface_site() for atom in mol.atoms)


def count_surface_sites(mol):
    return sum(1 for atom in mol.atoms if atom.is_surface_site())


def surface_atom_indices(mol):
    """Molecule indices of the atoms bonded to a framework site, in atom
    order (the "adatoms")."""
    return [mol.atoms.index(atom) for atom in mol.get_adatoms()]


# --------------------------------------------------------------------------
# yaml -> ReactionSet
# --------------------------------------------------------------------------

def read_reactions_yaml(path):
    """(reactions, standalone species) records from the yaml. Reactions get
    an ``index`` in file order."""
    with open(path) as handle:
        records = yaml.safe_load(handle)
    reactions = [record for record in records if "reactant" in record]
    species = [record for record in records if "reactant" not in record]
    for i, reaction in enumerate(reactions):
        reaction["index"] = i
    return reactions, species


def _match_species(mol, species):
    """(name, unique molecule) in ``species`` isomorphic to ``mol``."""
    for name, candidate in species.items():
        if candidate is mol or candidate.is_isomorphic(mol, save_order=True):
            return name, candidate
    raise ValueError("no unique species matches\n%s" % mol.to_adjacency_list())


def build_reaction_set(reactions, standalone_species,
                       thermodynamic_references=False):
    """Collect every molecule on every side, deduplicate by isomorphism, name
    them, and annotate each reaction with the names of its sides."""
    molecules = []
    if thermodynamic_references:
        molecules += [Molecule().from_smiles(smiles)
                      for smiles in THERMO_REFERENCE_SMILES]

    for record in standalone_species:
        molecules.append(molecule_from_adjlist(record["molecule"]))

    for reaction in reactions:
        reaction["reactant_mols"] = []
        reaction["product_mols"] = []
        for key, side in (("reactant_mols", reaction["reactant"]),
                          ("product_mols", reaction["product"])):
            whole_side = Molecule().from_adjacency_list(side)
            whole_side.clear_labeled_atoms()
            for fragment in whole_side.split():
                fragment.multiplicity = fragment.get_radical_count() + 1
                if not fragment.is_surface_site():
                    molecules.append(fragment)
                    reaction[key].append(fragment)

    unique = []
    for mol in molecules:
        if not any(mol.is_isomorphic(seen) for seen in unique):
            unique.append(mol)

    species = {get_name(mol): mol for mol in unique}
    adjlists = {name: mol.to_adjacency_list() for name, mol in species.items()}

    for reaction in reactions:
        for key, name_key in (("reactant_mols", "reactant_names"),
                              ("product_mols", "product_names")):
            reaction[name_key] = []
            for i, mol in enumerate(reaction[key]):
                name, unique_mol = _match_species(mol, species)
                reaction[key][i] = unique_mol
                reaction[name_key].append(name)
            reaction[key] = [mol.to_adjacency_list() for mol in reaction[key]]

    return ReactionSet(species, adjlists, reactions)


def load_reaction_set(path, thermodynamic_references=False):
    """``read_reactions_yaml`` + ``build_reaction_set`` in one call."""
    reactions, standalone = read_reactions_yaml(path)
    return build_reaction_set(reactions, standalone, thermodynamic_references)
