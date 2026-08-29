"""Everything machine- or run-specific lives here: paths on HPC2, the names of
the sub-directories inside a run directory, and the numeric thresholds the
workflow uses. Nothing else in ``pyntaz`` hard-codes a path or a cutoff.

Machine paths can be overridden without editing this file through the
``PYNTAZ_*`` environment variables listed next to each value.
"""

import os

# --------------------------------------------------------------------------
# machine paths (HPC2 defaults, override with environment variables)
# --------------------------------------------------------------------------

# folder that contains the ``pyntaz/`` package; SLURM jobs put it on PYTHONPATH
REPO = os.environ.get(
    "PYNTAZ_REPO",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# python interpreter used inside SLURM jobs (the conda env python directly --
# ``conda activate`` fails silently in batch jobs on HPC2)
PYTHON = os.environ.get(
    "PYNTAZ_PYTHON",
    "/nfs/hpc2/apo/kulkarnilab/ivanwang/miniconda3/envs/pyntaz-env/bin/python")

# MACE model file. A bare model name would make MACE try to download it,
# which fails on a compute node without internet.
MACE_MODEL = os.environ.get(
    "PYNTAZ_MODEL",
    os.path.join(REPO, "models", "mace-mpa-0-medium.model"))

SLURM_ACCOUNT = os.environ.get("PYNTAZ_SLURM_ACCOUNT", "ark245grp")
SLURM_PARTITION = os.environ.get("PYNTAZ_SLURM_PARTITION", "high")

# per relaxation job
SLURM_CORES = 4
SLURM_MEMORY = "16G"
SLURM_TIME = "04:00:00"


def check_machine_paths():
    """Print every path above that does not exist; True when all exist."""
    missing = [(name, path) for name, path in
               (("PYTHON", PYTHON), ("MACE_MODEL", MACE_MODEL), ("REPO", REPO))
               if not os.path.exists(path)]
    for name, path in missing:
        print("%s does not exist: %s" % (name, path))
    return not missing


# --------------------------------------------------------------------------
# run directory layout
# --------------------------------------------------------------------------

DEFAULT_RUN_DIR = "test_run"
DEFAULT_REACTIONS_FILE = "reaction.yaml"

INITIAL_GUESS_SUFFIX = "_init.xyz"   # <stem>_init.xyz written by placement
RELAXED_STRUCTURE = "relax.xyz"      # written by relax_one.py
RELAX_TRAJECTORY = "relax.traj"
RELAX_LOG = "relax.log"
SPECIES_INFO = "info.json"
GAS_STEM = "gas"                     # stem used for gas-phase species


class RunLayout:
    """The sub-directories of one run directory.

    ``<run_dir>/Adsorbates/<species>/<site>/<stem>/<stem>_init.xyz``  initial guesses
    ``<run_dir>/Adsorbates_relax/<species>/<site>/<stem>/relax.*``    MACE relaxations
    ``<run_dir>/Adsorbates_relax_filtered/<species>/<site>/<stem>.xyz`` survivors
    ``<run_dir>/Adsorbates_relax_unique/<species>/<site>/<stem>.xyz``   deduplicated minima
    ``<run_dir>/ts_guesses/<i>_rxn/pair_NNNN/``                       TS endpoint pairs
    """

    def __init__(self, run_dir=DEFAULT_RUN_DIR):
        self.run_dir = run_dir
        self.adsorbates = os.path.join(run_dir, "Adsorbates")
        self.relaxed = os.path.join(run_dir, "Adsorbates_relax")
        self.filtered = os.path.join(run_dir, "Adsorbates_relax_filtered")
        self.unique = os.path.join(run_dir, "Adsorbates_relax_unique")
        self.ts_guesses = os.path.join(run_dir, "ts_guesses")
        self.plots = run_dir

    def __repr__(self):
        return "RunLayout(%r)" % self.run_dir


# --------------------------------------------------------------------------
# numeric settings
# --------------------------------------------------------------------------

# framework construction
SUPERCELL_MIN_LENGTH = 12.0     # A, repeat the unit cell until every edge is at least this

# placement sweep
MONODENTATE_ANGLES = 24         # spin steps about the site normal (15 deg)
BIDENTATE_PHI_STEPS = 24        # swing about the O-O axis (15 deg)
BIDENTATE_PSI_STEPS = 12        # roll about the binder-binder axis (30 deg)
GAS_VACUUM = 10.0               # A of vacuum around a gas-phase molecule

# relaxation
RELAX_FMAX = 0.05               # eV/A
RELAX_MAX_STEPS = 100

# filtering
BOND_CHANGE_THRESHOLD = 0.5     # A, a bond that moved more than this has broken
RMSD_THRESHOLD = 2.0            # A, below this two relaxed configs are the same minimum
CLASH_TOLERANCE = 0.7           # fraction of the covalent-radius sum below which two atoms clash

# transition-state guess
FRAMEWORK_DRIFT_TOLERANCE = 1e-6  # A, both endpoints must share the same frozen framework
TS_STRETCH_START = 0.4            # A, initial pull on the breaking bond
TS_STRETCH_STEP = 0.2             # A, added per retry when the inserted gas still clashes
TS_STRETCH_MAX = 2.0              # A, past this the guess is dissociated, not stretched
TS_ROLL_STEPS = 24                # roll steps of the inserted gas about the forming-bond axis
