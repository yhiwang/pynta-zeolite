"""Step 1 of the workflow: put one species of the reaction set on every site
of the framework it fits and hand back the initial guesses.

A monodentate species (one X in its adjacency list) goes on every entry of
``framework.mono_sites``, a bidentate one (two X) on every pair in
``framework.bi_sites``; a gas-phase species (no X) is just centred in a box.
Each guess comes with a *tag* (see :mod:`pyntaz.placement`) naming its
orientation and the clearance score it was ranked by.

:func:`species_info` builds the bookkeeping record that later steps need to
map molecule atoms onto atoms of a stored structure (read back by
:func:`pyntaz.filtering.binder_indices`); how and where that record is
stored is up to the caller.
"""

from collections import namedtuple

from ase.data import atomic_numbers, covalent_radii

from .placement import place_monodentate, place_bidentate
from .pynta_mol import get_adsorbate
from .reactions import surface_atom_indices

GAS_VACUUM = 10.0       # A of vacuum around a gas-phase molecule

# structures:       [ase.Atoms] framework + adsorbate (or the bare gas molecule)
# site_ids:         [int or None] index into mono_sites / bi_sites, None for gas
# tags:             [dict] orientation tag of each structure (+ "site_indices")
# scores:           [float] tail clearance of each structure, A
# mol_to_atoms_map: {molecule atom index: adsorbate atom index}
SpeciesGuesses = namedtuple("SpeciesGuesses",
                            "structures site_ids tags scores mol_to_atoms_map")


def estimate_bond_length(framework_atoms, adsorbate, site, binder):
    """Sum of covalent radii of the site atom and the binding atom."""
    site_symbol = framework_atoms[site["indices"][0]].symbol
    binder_symbol = adsorbate[binder].symbol
    return (covalent_radii[atomic_numbers[site_symbol]]
            + covalent_radii[atomic_numbers[binder_symbol]])


def place_on_sites(adsorbate, framework_atoms, binders, sites):
    """Full orientation sweep of ``adsorbate`` on one site (or site pair).
    Returns (structures, tags, scores)."""
    if len(binders) == 1:
        bond_length = estimate_bond_length(framework_atoms, adsorbate,
                                           sites[0], binders[0])
        return place_monodentate(framework_atoms, adsorbate, binders[0], sites[0],
                                 bond_length, best_only=False)
    bond_lengths = (estimate_bond_length(framework_atoms, adsorbate,
                                         sites[0], binders[0]),
                    estimate_bond_length(framework_atoms, adsorbate,
                                         sites[1], binders[1]))
    return place_bidentate(framework_atoms, adsorbate, binders, sites,
                           bond_lengths, best_only=False)


def generate_guesses(mol, adsorbate, framework, mol_to_atoms_map):
    """(structures, site_ids, tags, scores) over every site the species
    fits: monodentate species go on ``framework.mono_sites``, bidentate on
    ``framework.bi_sites``. Every tag also records the framework oxygen
    indices of its site as ``site_indices``."""
    binders = [mol_to_atoms_map[i] for i in surface_atom_indices(mol)]

    if len(binders) == 1:
        sites_lists = [[site] for site in framework.mono_sites]
    elif len(binders) == 2:
        sites_lists = [list(pair) for pair in framework.bi_sites]
    else:
        raise ValueError("only monodentate and bidentate are supported, got %d"
                         % len(binders))

    structures, site_ids, tags, scores = [], [], [], []
    for site_id, sites in enumerate(sites_lists):
        placed, placed_tags, placed_scores = place_on_sites(
            adsorbate, framework.atoms, binders, sites)
        for structure, tag, score in zip(placed, placed_tags, placed_scores):
            structures.append(structure)
            site_ids.append(site_id)
            tag = dict(tag)
            tag["site_indices"] = [int(site["indices"][0]) for site in sites]
            tags.append(tag)
            scores.append(score)
    return structures, site_ids, tags, scores


def species_guesses(mol, framework, gas_vacuum=GAS_VACUUM):
    """Every initial guess for one species as a :class:`SpeciesGuesses`.

    The 3D adsorbate comes from pynta's ``get_adsorbate`` (an RDKit
    conformer of the desorbed molecule). A gas-phase species gives a single
    structure, centred in ``gas_vacuum`` A of vacuum with the framework's
    periodicity, tagged ``{}`` with ``site_id`` None.
    """
    adsorbate, mol_to_atoms_map = get_adsorbate(mol)

    if not mol.get_surface_sites():
        adsorbate.pbc = framework.atoms.pbc
        adsorbate.center(vacuum=gas_vacuum)
        return SpeciesGuesses([adsorbate], [None], [{}], [0.0], mol_to_atoms_map)

    structures, site_ids, tags, scores = generate_guesses(
        mol, adsorbate, framework, mol_to_atoms_map)
    return SpeciesGuesses(structures, site_ids, tags, scores, mol_to_atoms_map)


# --------------------------------------------------------------------------
# the species record
# --------------------------------------------------------------------------

def species_info(name, mol, mol_to_atoms_map, n_framework, configs):
    """The bookkeeping record of one species, json-able.

    Keys: ``name``; ``adjlist``; ``atom_to_molecule_atom_map`` (adsorbate
    index -> molecule index); ``gratom_to_molecule_surface_atom_map``
    (adsorbate index of each binder -> molecule index); ``nslab`` (framework
    atom count, the adsorbate starts after it in every combined structure);
    ``configs`` ({config key: tag}, as given).
    """
    binder_to_mol_atom = {mol_to_atoms_map[i]: i
                          for i, atom in enumerate(mol.atoms)
                          if atom.is_bonded_to_surface()}
    return {"name": name,
            "adjlist": mol.to_adjacency_list(),
            "atom_to_molecule_atom_map": {v: k for k, v in mol_to_atoms_map.items()},
            "gratom_to_molecule_surface_atom_map": binder_to_mol_atom,
            "nslab": n_framework,
            "configs": configs}
