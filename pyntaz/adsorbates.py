"""Step 1 of the workflow: put every species of the reaction set on every
site of the framework and write the initial guesses.

Output layout (see :mod:`pyntaz.runtree`)::

    <run_dir>/Adsorbates/<species>/<site>/<stem>/<stem>_init.xyz
    <run_dir>/Adsorbates/<species>/info.json

``<site>`` is the zero-padded index into ``framework.mono_sites`` (or
``bi_sites``), ``<stem>`` the orientation (``degrees_045``,
``flip0_phi105_psi240``) and ``0/gas`` for a gas-phase molecule.
"""

import os

from ase.data import atomic_numbers, covalent_radii
from ase.io import write
from pynta.mol import get_adsorbate

from . import config, runtree
from .placement import place_monodentate, place_bidentate, orientation_stem
from .reactions import surface_atom_indices


def estimate_bond_length(framework_atoms, adsorbate, site, binder):
    """Sum of covalent radii of the site atom and the binding atom."""
    site_symbol = framework_atoms[site["indices"][0]].symbol
    binder_symbol = adsorbate[binder].symbol
    return (covalent_radii[atomic_numbers[site_symbol]]
            + covalent_radii[atomic_numbers[binder_symbol]])


def place_on_sites(adsorbate, framework_atoms, binders, sites):
    """Full orientation sweep of ``adsorbate`` on one site (or site pair).
    Returns (structures, tags)."""
    if len(binders) == 1:
        bond_length = estimate_bond_length(framework_atoms, adsorbate,
                                           sites[0], binders[0])
        structures, tags, scores = place_monodentate(
            framework_atoms, adsorbate, binders[0], sites[0], bond_length,
            best_only=False)
    else:
        bond_lengths = (estimate_bond_length(framework_atoms, adsorbate,
                                             sites[0], binders[0]),
                        estimate_bond_length(framework_atoms, adsorbate,
                                             sites[1], binders[1]))
        structures, tags, scores = place_bidentate(
            framework_atoms, adsorbate, binders, sites, bond_lengths,
            best_only=False)

    label = "-".join("O%d" % site["indices"][0] for site in sites)
    print("  %-12s %3d orientations  tail clearance %.2f-%.2f A"
          % (label, len(structures), min(scores), max(scores)))
    return structures, tags


def generate_guesses(mol, adsorbate, framework, mol_to_atoms_map):
    """(structures, site_ids, tags) over every site the species fits:
    monodentate species go on ``framework.mono_sites``, bidentate on
    ``framework.bi_sites``."""
    binders = [mol_to_atoms_map[i] for i in surface_atom_indices(mol)]

    if len(binders) == 1:
        sites_lists = [[site] for site in framework.mono_sites]
    elif len(binders) == 2:
        sites_lists = [list(pair) for pair in framework.bi_sites]
    else:
        raise ValueError("only monodentate and bidentate are supported, got %d"
                         % len(binders))

    structures, site_ids, tags = [], [], []
    for site_id, sites in enumerate(sites_lists):
        placed, placed_tags = place_on_sites(adsorbate, framework.atoms,
                                             binders, sites)
        for structure, tag in zip(placed, placed_tags):
            structures.append(structure)
            site_ids.append(site_id)
            tag = dict(tag)
            tag["site_indices"] = [int(site["indices"][0]) for site in sites]
            tags.append(tag)
    return structures, site_ids, tags


def write_species_guesses(mol, name, run_dir, framework):
    """Write every initial guess for one species and its ``info.json``.

    If the species folder already exists it is reused untouched and the
    existing xyz files are returned. Returns the list of xyz paths.
    """
    layout = config.RunLayout(run_dir)
    species_dir = os.path.join(layout.adsorbates, name)
    if os.path.exists(species_dir):
        print("%s: reusing %s" % (name, species_dir))
        return [path for _, path in runtree.initial_guess_files(species_dir)]

    adsorbate, mol_to_atoms_map = get_adsorbate(mol)

    if not mol.get_surface_sites():
        adsorbate.pbc = framework.atoms.pbc
        adsorbate.center(vacuum=config.GAS_VACUUM)
        structures, site_ids, tags = [adsorbate], [None], [{}]
    else:
        print("\n%s" % name)
        structures, site_ids, tags = generate_guesses(mol, adsorbate, framework,
                                                      mol_to_atoms_map)

    binder_to_mol_atom = {}
    for i, atom in enumerate(mol.atoms):
        if atom.is_bonded_to_surface():
            binder_to_mol_atom[mol_to_atoms_map[i]] = i

    xyz_paths, manifest = [], {}
    for structure, site_id, tag in zip(structures, site_ids, tags):
        if site_id is None:
            relative = os.path.join("0", config.GAS_STEM)
        else:
            relative = os.path.join("%02d" % site_id, orientation_stem(tag))
        config_dir = os.path.join(species_dir, relative)
        os.makedirs(config_dir, exist_ok=True)
        xyz = os.path.join(config_dir,
                           os.path.basename(relative) + config.INITIAL_GUESS_SUFFIX)
        write(xyz, structure)
        xyz_paths.append(xyz)
        manifest[relative] = tag

    info = {"name": name,
            "adjlist": mol.to_adjacency_list(),
            "atom_to_molecule_atom_map": {v: k for k, v in mol_to_atoms_map.items()},
            "gratom_to_molecule_surface_atom_map": binder_to_mol_atom,
            "nslab": len(framework.atoms),
            "configs": manifest}
    runtree.write_species_info(layout.adsorbates, name, info)
    return xyz_paths


def write_all_guesses(reaction_set, run_dir, framework):
    """{species name: [xyz paths]} for every species in the reaction set."""
    return {name: write_species_guesses(mol, name, run_dir, framework)
            for name, mol in reaction_set.species.items()}
