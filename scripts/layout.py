"""Where everything lives on disk. This is the only file that knows the
run-directory layout and the file names, so every step finds things the
same way and changing the layout means changing this file::

    <run_dir>/bare.xyz, framework.json, <CODE>_pairs.json        step 0
    <run_dir>/Adsorbates/<species>/info.json                     step 1
    <run_dir>/Adsorbates/<species>/<site>/<stem>/<stem>_init.xyz
    <run_dir>/Adsorbates_relax/<species>/<site>/<stem>/relax.*   step 2
    <run_dir>/Adsorbates_relax_filtered/<species>/<site>/<stem>.xyz   step 3
    <run_dir>/Adsorbates_relax_unique/<species>/<site>/<stem>.xyz
    <run_dir>/<species>_sweep.png                                step 4
    <run_dir>/TS_guesses/<i>_rxn/info.json                       step 5
    <run_dir>/TS_guesses/<i>_rxn/pair_<k>/<stem>/<stem>_init.xyz
    <run_dir>/TS_guesses/<i>_rxn/pair_<k>/sweep.traj

``<site>`` is the zero-padded index into the framework's ``mono_sites`` (or
``bi_sites``), ``<stem>`` the orientation name from ``pyntaz.placement``
(``degrees_045``, ``flip0_phi105_psi240``) or ``pyntaz.ts_graph``; a
gas-phase molecule sits at ``0/gas``.

``os.listdir`` / ``os.walk`` are used rather than ``glob``: species names
like ``C[CH2][Pt]`` contain brackets, which glob reads as character classes.
"""

import json
import os
import re
import shutil

import _common  # noqa: F401  -- repo on sys.path
from ase.io import read, write

# -- file names --------------------------------------------------------------

BARE_XYZ = "bare.xyz"                 # the framework atoms
FRAMEWORK_JSON = "framework.json"     # ZeoliteFramework.to_dict()
PAIRS_JSON = "%s_pairs.json"          # unique second-order T pairs of a code
INITIAL_GUESS_SUFFIX = "_init.xyz"    # <stem>_init.xyz, one per config folder
RELAXED_STRUCTURE = "relax.xyz"       # written by relax_one.py, carries energy + forces
RELAX_TRAJECTORY = "relax.traj"
RELAX_LOG = "relax.log"
JOB_SCRIPT = "job.sh"
SPECIES_INFO = "info.json"            # adsorbates.species_info() record
REACTION_INFO = "info.json"           # TS_guesses/<i>_rxn/: reaction + pair manifest
SWEEP_TRAJECTORY = "sweep.traj"       # every roll of every survivor, for ase gui
GAS_SITE = "0"                        # site folder of a gas-phase species


class RunLayout:
    """The sub-directories of one run directory."""

    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.adsorbates = os.path.join(run_dir, "Adsorbates")
        self.relaxed = os.path.join(run_dir, "Adsorbates_relax")
        self.filtered = os.path.join(run_dir, "Adsorbates_relax_filtered")
        self.unique = os.path.join(run_dir, "Adsorbates_relax_unique")
        self.ts_guesses = os.path.join(run_dir, "TS_guesses")
        self.plots = run_dir

    def __repr__(self):
        return "RunLayout(%r)" % self.run_dir


def safe_filename(name):
    """Species names contain ``= [ ]`` -- keep them out of filenames."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def site_dirname(site_id):
    return "%02d" % site_id


def reaction_dirname(index):
    return "%d_rxn" % index


def pair_dirname(pair_id):
    return "pair_%02d" % pair_id


def initial_guess_path(config_dir):
    """``<config_dir>/<stem>_init.xyz`` where stem is the folder's own name."""
    return os.path.join(config_dir,
                        os.path.basename(config_dir) + INITIAL_GUESS_SUFFIX)


# -- walking a species tree --------------------------------------------------

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


def config_dirs(species_dir, marker=RELAX_TRAJECTORY):
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
        inits = [name for name in sorted(files) if name.endswith(INITIAL_GUESS_SUFFIX)]
        if not inits:
            continue
        if len(inits) > 1:
            raise RuntimeError("%s holds %d %s files, expected 1"
                               % (dirpath, len(inits), INITIAL_GUESS_SUFFIX))
        found.append((os.path.relpath(dirpath, root), os.path.join(dirpath, inits[0])))
    return sorted(found)


def stems_in(site_dir):
    """{stem} of the ``.xyz`` files in a flat site folder."""
    return {os.path.splitext(name)[0] for name in os.listdir(site_dir)
            if name.endswith(".xyz")}


# -- species info.json --------------------------------------------------------

def species_info_path(tree, species):
    return os.path.join(tree, species, SPECIES_INFO)


def load_species_info(tree, species):
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
                     os.path.join(target, SPECIES_INFO))


def write_species_guesses(tree, name, mol, guesses, n_framework):
    """Lay the :class:`pyntaz.adsorbates.SpeciesGuesses` of one species out
    as ``<tree>/<name>/<site>/<stem>/<stem>_init.xyz`` and write its
    ``info.json``. Returns the manifest {"<site>/<stem>": tag}."""
    from pyntaz.adsorbates import species_info
    from pyntaz.placement import orientation_stem
    manifest = {}
    for structure, site_id, tag in zip(guesses.structures, guesses.site_ids,
                                       guesses.tags):
        site = GAS_SITE if site_id is None else site_dirname(site_id)
        relative = os.path.join(site, orientation_stem(tag))
        config_dir = os.path.join(tree, name, relative)
        os.makedirs(config_dir, exist_ok=True)
        write(initial_guess_path(config_dir), structure)
        manifest[relative] = tag
    write_species_info(tree, name, species_info(
        name, mol, guesses.mol_to_atoms_map, n_framework, manifest))
    return manifest


# -- framework, reactions, structures -----------------------------------------
# (pyntaz imported inside the functions: relax_one.py imports this module on
# a compute node where maze and RMG may not be installed)

def save_framework(framework, run_dir):
    """``bare.xyz`` + ``framework.json`` into ``run_dir``."""
    os.makedirs(run_dir, exist_ok=True)
    write(os.path.join(run_dir, BARE_XYZ), framework.atoms)
    with open(os.path.join(run_dir, FRAMEWORK_JSON), "w") as handle:
        json.dump(framework.to_dict(), handle, indent=2)


def load_framework(run_dir):
    """The framework saved by step 0 in ``run_dir``."""
    from pyntaz.framework import ZeoliteFramework
    atoms = read(os.path.join(run_dir, BARE_XYZ))
    with open(os.path.join(run_dir, FRAMEWORK_JSON)) as handle:
        return ZeoliteFramework.from_dict(json.load(handle), atoms)


def save_pairs(pairs, run_dir, code):
    """The unique-pair table of ``code`` (step 0) next to the framework."""
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, PAIRS_JSON % code), "w") as handle:
        json.dump(pairs, handle, indent=2)


def load_reactions(path):
    """ReactionSet from a reaction yaml file."""
    from pyntaz.reactions import reaction_set_from_yaml
    with open(path) as handle:
        return reaction_set_from_yaml(handle.read())


def read_with_energy(path):
    """(atoms, energy) -- energy is None when the file carries none."""
    atoms = read(path)
    try:
        return atoms, atoms.get_potential_energy()
    except (RuntimeError, AttributeError):
        return atoms, None
