"""Where everything lives on disk. This is the only file that knows the
run-directory layout and the file names, so every step finds things the
same way and changing the layout means changing this file.

One run directory per model, round, framework and site (run_dirs)::

    runs/<MODEL>/<ROUND>/<FRAMEWORK>/<site>/      e.g. runs/mpa0/round0/MOR/T1/

and inside each one::

    <run_dir>/bare.xyz, framework.json, <CODE>_pairs.json        step 0
    <run_dir>/TS_guesses/<i>_rxn/info.json                       step 1
    <run_dir>/TS_guesses/<i>_rxn/pair_<k>/<stem>/<stem>_init.xyz
    <run_dir>/TS_guesses/<i>_rxn/pair_<k>/sweep.traj
    <run_dir>/TS_unique/<i>_rxn/pair_<k>/<stem>/<stem>_init.xyz    step 2
    <run_dir>/TS_unique/<i>_rxn/pair_counts.png                  analysis/plot_pair_counts.py
    <run_dir>/TS_harmonic/<i>_rxn/info.json                      step 3
    <run_dir>/TS_harmonic/<i>_rxn/pair_<k>/<stem>/<stem>_init.xyz, ts_bonds.json
    <run_dir>/TS_harmonic/<i>_rxn/pair_<k>/<stem>/ts_harmonic.*
    <run_dir>/TS_harmonic/<i>_rxn/pair_<k>/<stem>/<state>/<state>.*, endpoint.json   (initial, final)
    <run_dir>/TS_sella/<i>_rxn/pair_<k>/<stem>/<start>/start.xyz, ts_bonds.json   step 4
    <run_dir>/TS_sella/<i>_rxn/pair_<k>/<stem>/<start>/ts_sella.xyz, sella.*, vib*, result.json
    <run_dir>/TS_sella/ts_summary.csv, ts_summary.png             analysis/summarize_ts.py
    <run_dir>/DFT_sp/<i>_rxn/pair_<k>/<stem>/<start>/f<frame>/config.extxyz, source.json   dft/01
    <run_dir>/DFT_sp/manifest.csv, all_configs.extxyz              dft/01

``<stem>`` is the guess name from ``pyntaz.ts_graph``.

``os.listdir`` / ``os.walk`` are used rather than ``glob``: species names
like ``C[CH2][Pt]`` contain brackets, which glob reads as character classes.
"""

import json
import os

import _common  # noqa: F401  -- repo on sys.path
from ase.io import read, write

# -- file names --------------------------------------------------------------

BARE_XYZ = "bare.xyz"                 # the framework atoms
FRAMEWORK_JSON = "framework.json"     # ZeoliteFramework.to_dict()
PAIRS_JSON = "%s_pairs.json"          # unique second-order T pairs of a code
INITIAL_GUESS_SUFFIX = "_init.xyz"    # <stem>_init.xyz, one per config folder
JOB_SCRIPT = "job.sh"
JOB_ERR = "job.err"                   # SLURM stderr of a job
JOB_ID = "job.id"                     # SLURM id, written by steps 3-4 after sbatch
REACTION_INFO = "info.json"           # TS_guesses/<i>_rxn/: reaction + pair manifest
SWEEP_TRAJECTORY = "sweep.traj"       # every roll of every survivor, for ase gui
PAIR_TABLE = "pair_counts.png"        # TS_unique/<i>_rxn/: plot_pair_counts.py table
TS_BONDS_JSON = "ts_bonds.json"       # TS_harmonic/.../<stem>/: reacting bonds as atom indices
TS_HARMONIC_STRUCTURE = "ts_harmonic.xyz"   # written by ts_harmonic_one.py, MACE energy + forces
TS_HARMONIC_TRAJECTORY = "ts_harmonic.traj"
TS_HARMONIC_LOG = "ts_harmonic.log"
ENDPOINT_RESULT = "endpoint.json"     # TS_harmonic/.../<stem>/<state>/: bond check of initial / final
TS_START = "start.xyz"                # TS_sella/.../<start>/: the structure Sella starts from
TS_SELLA_STRUCTURE = "ts_sella.xyz"   # written by ts_sella_one.py, MACE energy + forces
TS_SELLA_TRAJECTORY = "sella.traj"
TS_SELLA_LOG = "sella.log"
TS_SELLA_VIB = "vib"                  # vibration cache dir; the first mode goes to vib.0.traj
TS_SELLA_RESULT = "result.json"       # verdict, energies, frequencies; written last
TS_SUMMARY_CSV = "ts_summary.csv"     # TS_sella/: summarize_ts.py table, one row per guess and start
TS_SUMMARY_PNG = "ts_summary.png"     # TS_sella/: summarize_ts.py figure
ENDPOINT_SUMMARY_CSV = "endpoint_summary.csv"   # TS_harmonic/: summarize_endpoints.py table, one row per guess
ENDPOINT_SUMMARY_PNG = "endpoint_summary.png"   # TS_harmonic/: summarize_endpoints.py figure
NEB_RECORD = "neb.json"               # NEB/.../<stem>/: sources, MACE energies, bonds (step 5)
NEB_STATES = {"initial": "initial.xyz", "ts": "ts.xyz", "final": "final.xyz"}   # NEB/.../<stem>/
NEB_BAND = "band.traj"                # NEB/.../<stem>/: starting band, FixAtoms on every image (step 5)
NEB_BAND_INFO = "band.json"           # NEB/.../<stem>/: how the band was built, free atoms (step 5)
NEB_RESULT = "neb_result.json"        # NEB/.../<stem>/: written last by the step 6 job
NEB_TRAJECTORY = "neb.traj"           # NEB/.../<stem>/: every image of every NEB step (step 6)
NEB_LOG = "neb.log"                   # NEB/.../<stem>/: endpoint relax + FIRE log, both stages
NEB_FINAL_BAND = "neb_band.traj"      # NEB/.../<stem>/: the band where the NEB stopped, MACE energy + forces
NEB_PLOT = "neb.png"                  # NEB/.../<stem>/: energy along the final band
DFT_CONFIG = "config.extxyz"          # DFT_sp/.../f<frame>/: one structure, MACE labels kept as mace_*
DFT_SOURCE = "source.json"            # DFT_sp/.../f<frame>/: where the frame came from
DFT_MANIFEST = "manifest.csv"         # DFT_sp/: one row per collected config
DFT_ALL_CONFIGS = "all_configs.extxyz"   # DFT_sp/: every collected config, for ase gui
DFT_SP = "DFT_sp"                     # <framework folder>/DFT_sp/: the pool and the picks of all sites
DFT_POOL = "pool"                     # DFT_sp/pool/<i>_rxn/: every structure dft/01 collected
DFT_POOL_CONFIGS = "pool.extxyz"      # DFT_sp/pool/<i>_rxn/: the structures, MACE labels kept as mace_*
DFT_POOL_TABLE = "pool.csv"           # DFT_sp/pool/<i>_rxn/: one row per structure, same order
DFT_SELECTED = "selected.csv"         # DFT_sp/<i>_rxn/: the picks of dft/02, in pick order
DFT_REJECTED_CONFIGS = "rejected.extxyz"   # DFT_sp/pool/<i>_rxn/: the structures a dft/01 check removed
DFT_REJECTED_TABLE = "rejected.csv"        # DFT_sp/pool/<i>_rxn/: one row per removed structure, with the reason
DFT_ATOMS = "DFT_atoms"               # runs/DFT_atoms/<element>/: the isolated atoms of dft/04, one copy for all runs
DFT_TRAINING = "training"             # DFT_sp/training/: the data split of dft/05
DFT_MODEL_PREFIX = "model_"           # DFT_sp/training/model_<k>/: the files of one model (model_dirname)
DFT_TEST = "test.extxyz"              # DFT_sp/training/: the test set, the same for every model
DFT_TRAIN = "train.extxyz"            # DFT_sp/training/model_<k>/: the train set, with the isolated atoms
DFT_VALID = "valid.extxyz"            # DFT_sp/training/model_<k>/: the valid set
DFT_SPLIT = "split.csv"               # DFT_sp/training/: one row per structure, its set in each model
MACE_FT = "MACE_ft"                   # <framework folder>/MACE_ft/model_<k>/: the fine-tune run of one model
TS_COLLECT_STATES = {"initial": "initial.xyz", "ts": "ts.xyz", "final": "final.xyz"}  # DFT_TS/.../<stem>/ (step 7)
TS_COLLECT_INFO = "ts_info.json"      # DFT_TS/.../<stem>/: reaction, site, bonds, NEB quality, MACE energies
TS_COLLECT_CSV = "ts_collected.csv"   # DFT_TS/: one row per collected TS, with the ranking
DFT_SELLA_DIR = "dft_sella"            # DFT_TS/.../<stem>/dft_sella/: the step 8 job runs here
DFT_SELLA_STRUCTURE = "ts_dft.xyz"     # DFT_TS/.../<stem>/dft_sella/: where Sella stopped, DFT energy + forces
DFT_SELLA_VASP = "vasp"                # DFT_TS/.../<stem>/dft_sella/vasp/: VASP's files, WAVECAR between steps
DFT_SELLA_PROGRESS = "progress.json"   # DFT_TS/.../<stem>/dft_sella/: steps so far + last accepted geometry
QUASI_IRC_DIR = "quasi_irc"           # DFT_TS/.../<stem>/quasi_irc/: start.xyz + ts_info.json (step 9); steps 10-11 run in its subfolders
DFT_VIB_DIR = "vib"                   # DFT_TS/.../<stem>/quasi_irc/vib/: the step 10 job runs here
DFT_VIB_CACHE = "displacements"       # quasi_irc/vib/displacements/: ASE's forces, one file per displacement (kept for --resume)
DFT_VIB_VASP = "vasp"                 # quasi_irc/vib/vasp/: VASP's files, WAVECAR between displacements
DFT_VIB_MODE = "mode.xyz"             # quasi_irc/vib/: start.xyz + the first mode as the per-atom array "mode" (step 11)
DFT_VIB_BULKY = ("WAVECAR", "CHG", "CHGCAR", "vasprun.xml", "vaspout.h5", "DOSCAR",
                 "EIGENVAL", "PROCAR", "XDATCAR", "PCDAT")   # quasi_irc/vib/vasp/: the big VASP files,
                                       # deleted after every job unless settings.DFT_VIB_KEEP_WAVECAR


RUNS = "runs"                         # top folder of all runs, relative to the repo


def run_dirs(model, round_name, framework, sites):
    """[(site, run_dir)] with run_dir = runs/<model>/<round>/<framework>/<site>,
    in the order of ``sites``."""
    return [(site, os.path.join(RUNS, model, round_name, framework, site))
            for site in sites]

def framework_dir(model, round_name, framework):
    """runs/<model>/<round>/<framework>, the folder that holds the site folders."""
    return os.path.join(RUNS, model, round_name, framework)

class RunLayout:
    """The sub-directories of one run directory."""

    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.ts_guesses = os.path.join(run_dir, "TS_guesses")
        self.ts_unique = os.path.join(run_dir, "TS_unique")
        self.ts_harmonic = os.path.join(run_dir, "TS_harmonic")
        self.ts_sella = os.path.join(run_dir, "TS_sella")
        self.neb = os.path.join(run_dir, "NEB")
        self.dft_sp = os.path.join(run_dir, "DFT_sp")
        self.dft_ts = os.path.join(run_dir, "DFT_TS")
        
    def __repr__(self):
        return "RunLayout(%r)" % self.run_dir


def reaction_dirname(index):
    return "%d_rxn" % index


def pair_dirname(pair_id):
    return "pair_%02d" % pair_id


def model_dirname(number):
    """model_1, model_2, ...: the folder of one of the models trained on the same round."""
    return "%s%d" % (DFT_MODEL_PREFIX, number)


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


# -- framework, reactions, structures -----------------------------------------
# (pyntaz imported inside the functions: the worker scripts import this module
# on a compute node where maze and RMG may not be installed)

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


def harmonic_job_dir(config_dir, state):
    """Folder of one step 3 job: the guess folder itself for "ts" (where
    step 4 looks), a ``<state>/`` subfolder for any other state."""
    return config_dir if state == "ts" else os.path.join(config_dir, state)


def harmonic_files(state):
    """(structure, trajectory, log) file names of one step 3 state."""
    if state == "ts":
        return TS_HARMONIC_STRUCTURE, TS_HARMONIC_TRAJECTORY, TS_HARMONIC_LOG
    return state + ".xyz", state + ".traj", state + ".log"

POOL_FILES = ("raw", "ts_harmonic", "initial", "final", "ts_sella", "band", "neb_start", "neb_band")
POOL_FIRST_BAND = ("neb_start",)      # files that hold every band of the NEB: dft/01 takes only the
                                      # first one, the start band (band.json gives its size)


def pool_files(run, reaction, pair, stem):
    """{name: path} of the files dft/01 can collect from, for one guess of
    the RunLayout ``run``: the start file and the result file of steps 3,
    4 and 6. ``reaction`` and ``pair`` are the folder names."""
    guess = os.path.join(run.ts_harmonic, reaction, pair, stem)
    neb = os.path.join(run.neb, reaction, pair, stem)
    return {
        "raw": initial_guess_path(guess),
        "ts_harmonic": os.path.join(guess, harmonic_files("ts")[0]),
        "initial": os.path.join(harmonic_job_dir(guess, "initial"), harmonic_files("initial")[0]),
        "final": os.path.join(harmonic_job_dir(guess, "final"), harmonic_files("final")[0]),
        "ts_sella": os.path.join(run.ts_sella, reaction, pair, stem, "harmonic", TS_SELLA_STRUCTURE),
        "band": os.path.join(neb, NEB_BAND),
        "neb_start": os.path.join(neb, NEB_TRAJECTORY),
        "neb_band": os.path.join(neb, NEB_FINAL_BAND),
    }