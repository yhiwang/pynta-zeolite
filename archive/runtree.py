"""Reading and walking a run directory.

This is the only module that knows how ``<run_dir>/Adsorbates*/`` is laid
out on disk (``<species>/<site>/<stem>``), so every stage discovers configs
the same way. ``os.listdir`` / ``os.walk`` are used rather than ``glob``:
species names like ``C[CH2][Pt]`` contain brackets, which glob reads as
character classes.
"""

import json
import os
import re
import shutil

from ase.io import read

from . import config


def safe_filename(name):
    """Species names contain ``= [ ]`` -- keep them out of filenames."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


# --------------------------------------------------------------------------
# walking
# --------------------------------------------------------------------------

def species_dirs(tree):
    """[(species name, path)] for every species folder in a tree."""
    if not os.path.isdir(tree):
        return []
    return [(name, os.path.join(tree, name)) for name in sorted(os.listdir(tree))
            if os.path.isdir(os.path.join(tree, name))]


def site_dirs(species_dir):
    """[(site, path)] for the numeric site folders of one species."""
    return [(site, os.path.join(species_dir, site))
            for site in sorted(os.listdir(species_dir))
            if site.isdigit() and os.path.isdir(os.path.join(species_dir, site))]


def config_dirs(species_dir, marker=config.RELAX_TRAJECTORY):
    """[(site, stem, path)] for every ``<site>/<stem>/`` folder holding
    ``marker`` (relaxation layout, one folder per config)."""
    configs = []
    for dirpath, _, files in os.walk(species_dir):
        if marker not in files:
            continue
        relative = os.path.relpath(dirpath, species_dir).split(os.sep)
        if len(relative) != 2:
            continue
        configs.append((relative[0], relative[1], dirpath))
    return sorted(configs)


def config_files(species_dir):
    """[(site, stem, path)] for every ``<site>/<stem>.xyz`` (flat layout of
    the filtered / unique trees)."""
    configs = []
    for site, site_path in site_dirs(species_dir):
        for filename in sorted(os.listdir(site_path)):
            if filename.endswith(".xyz"):
                configs.append((site, os.path.splitext(filename)[0],
                                os.path.join(site_path, filename)))
    return configs


def initial_guess_files(root):
    """[(relative config dir, xyz path)] for every ``*_init.xyz`` under
    ``root``. Raises when a folder holds more than one."""
    found = []
    for dirpath, _, files in os.walk(root):
        inits = [name for name in sorted(files)
                 if name.endswith(config.INITIAL_GUESS_SUFFIX)]
        if not inits:
            continue
        if len(inits) > 1:
            raise RuntimeError("%s holds %d %s files, expected 1"
                               % (dirpath, len(inits), config.INITIAL_GUESS_SUFFIX))
        found.append((os.path.relpath(dirpath, root),
                      os.path.join(dirpath, inits[0])))
    return sorted(found)


def stems_in(site_dir):
    """{stem} of the ``.xyz`` files in a flat site folder."""
    return {os.path.splitext(name)[0] for name in os.listdir(site_dir)
            if name.endswith(".xyz")}


# --------------------------------------------------------------------------
# species info.json
# --------------------------------------------------------------------------

def species_info_path(tree, species):
    return os.path.join(tree, species, config.SPECIES_INFO)


def load_species_info(tree, species):
    """The ``info.json`` written by placement for ``species`` in ``tree``.

    Keys: name, adjlist, atom_to_molecule_atom_map (adsorbate index ->
    molecule index), gratom_to_molecule_surface_atom_map (adsorbate index of
    each binder -> molecule index), nslab (framework atom count), configs
    ({"<site>/<stem>": tag}).
    """
    with open(species_info_path(tree, species)) as handle:
        return json.load(handle)


def write_species_info(tree, species, info):
    os.makedirs(os.path.join(tree, species), exist_ok=True)
    with open(species_info_path(tree, species), "w") as handle:
        json.dump(info, handle, indent=2)


def copy_species_info(source_tree, target_tree, species):
    """Copy ``info.json`` alongside a species in another tree (only if that
    species folder exists there)."""
    target = os.path.join(target_tree, species)
    if os.path.isdir(target):
        shutil.copy2(species_info_path(source_tree, species),
                     os.path.join(target, config.SPECIES_INFO))


def binder_indices(info, n_framework=None):
    """Absolute indices of the binding atoms in a combined structure
    (framework atoms first, then the adsorbate). ``n_framework`` defaults to
    ``info["nslab"]``; pass the count found by connectivity when you have it."""
    if n_framework is None:
        n_framework = info["nslab"]
    return sorted(n_framework + int(k)
                  for k in info["gratom_to_molecule_surface_atom_map"])


# --------------------------------------------------------------------------
# structures
# --------------------------------------------------------------------------

def read_with_energy(path):
    """(atoms, energy) -- energy is None when the file carries none."""
    atoms = read(path)
    try:
        return atoms, atoms.get_potential_energy()
    except (RuntimeError, AttributeError):
        return atoms, None
