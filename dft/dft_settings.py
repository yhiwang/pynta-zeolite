"""How the DFT stage is tuned. The run itself (CODE, SITES, RUN_DIR) comes
from scripts/settings.py, so the DFT stage always works on the same run as
the pipeline. Edit here, not in the dft scripts."""

import os

# --------------------------------------------------------------------------
# dft step 1: collect configs from the Sella trajectories
# --------------------------------------------------------------------------

COLLECT_WRITE = True           # False: count and report only (dry run)
COLLECT_REACTIONS = None        # None: every reaction in TS_sella; or e.g. [0, 3]
COLLECT_STARTS = ("harmonic", "raw")    # which Sella starts to take frames from
COLLECT_VERDICTS = None         # None: every verdict (failed runs are useful data);
                                # or e.g. ("ts", "wrong_mode")
COLLECT_STRIDE = 1              # look at every Nth Sella step (1: all of them)
COLLECT_MIN_DISP = 0.1          # A, keep a frame once some atom has moved this far
                                # since the last kept frame
COLLECT_MAX_FRAMES = 3          # per trajectory, end points included; None: no cap
COLLECT_MIN_DIST = 0.7          # A, frames with two atoms closer than this are dropped

# --------------------------------------------------------------------------
# dft step 2: VASP single points on the collected configs (one SLURM job per
# config). Every config of a training set must use the same values below.
# --------------------------------------------------------------------------

SP_SUBMIT = True               # False: write the inputs + job.sh only (dry run)
SP_LIMIT = None                 # submit at most this many new configs per run;
                                # set 2-3 for a timing test first
SP_LANES = 6                    # at most this many jobs running at a time (6 x 32 = 192 cores)

VASP_PP_PATH = os.path.expanduser("~/vasp_pp")   # holds potpaw_PBE/ (a link to the
                                # lab's grp_shared/PBE64 or PBE54 -- never mix them)
VASP_MODULE = "vasp/6.5.1%nvhpc@24.9"
VASP_BINARY = "vasp_gam"        # Gamma-point only build, matches kpts below

# the lab's settings (RPBE-D3(BJ), 400 eV) as single points, closed shell
SP_VASP = dict(
    xc="rpbe",                  # GGA = RP with PBE POTCARs
    ivdw=12,                    # D3 with Becke-Johnson damping
    encut=400,
    prec="Normal",
    ediff=1e-6,
    ismear=0,
    sigma=0.05,
    ispin=1,
    nsw=0,
    ibrion=-1,
    algo="Fast",
    lreal="Auto",
    lasph=True,
    isym=0,
    nelm=200,
    nelmin=4,
    ncore=4,
    lwave=False,
    lcharg=False,
    kpts=(1, 1, 1),
    gamma=True,
)

SP_SLURM_ACCOUNT = "ark245grp"
SP_SLURM_PARTITION = "high"
SP_SLURM_TASKS = 32             # MPI ranks on one node
SP_SLURM_MEMORY = "64G"
SP_SLURM_TIME = "00:30:00"

# --------------------------------------------------------------------------
# dft step 3: training set from the finished single points
# --------------------------------------------------------------------------

TRAIN_SPLIT_BY = "config"       # "config": each config at random; "run": whole Sella runs
TRAIN_VALID_FRACTION = 0.1      # share going to valid   <- set your proportions here
TRAIN_TEST_FRACTION = 0.1       # share going to test       (the rest is train)
TRAIN_HOLDOUT_REACTIONS = []    # e.g. [6]: these reactions go to test entirely
TRAIN_SEED = 0                  # fixes the random draw (same seed = same split)
TRAIN_MAX_FORCE = 20.0          # eV/A, configs with any larger DFT force are dropped

# --------------------------------------------------------------------------
# dft step 4: isolated atoms for MACE's E0s (same settings as step 2, plus
# spin polarization -- free atoms are open shell)
# --------------------------------------------------------------------------

ATOM_SUBMIT = True             # False: write the inputs + job.sh only (dry run)
ATOM_ELEMENTS = ("H", "C", "O", "Al", "Si")
ATOM_MAGMOM = {"H": 1, "C": 2, "O": 2, "Al": 1, "Si": 2}   # ground-state unpaired
                                # electrons, fixed with NUPDOWN
ATOM_BOX = (20.0, 20.5, 21.0)   # A, slightly unequal so the box does not impose symmetry
ATOM_VASP_CHANGES = dict(       # applied on top of SP_VASP
    ispin=2,
    sigma=0.01,
    algo="Normal",
    nelm=300,
    lreal=False,
)
ATOM_SLURM_TASKS = 16
ATOM_SLURM_TIME = "02:00:00"
