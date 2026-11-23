"""How the DFT stage is tuned. The run itself (CODE, SITES, RUN_DIR) comes
from scripts/settings.py, so the DFT stage always works on the same run as
the pipeline. Edit here, not in the dft scripts."""

import sys
import settings                 # the dft scripts put scripts/ on sys.path first

# --------------------------------------------------------------------------
# dft step 1: collect configs (sella / harmonic / neb)
# --------------------------------------------------------------------------

COLLECT_WRITE = True           # False: count and report only (dry run)
COLLECT_REACTIONS = None        # None: every reaction; or e.g. [0, 3]
COLLECT_FILES = (               # names from layout.pool_files
    "ts_harmonic",              # ts_harmonic.xyz: step 3 TS, where step 4 starts
    "initial",                  # initial/initial.xyz: step 3 initial state
    "final",                    # final/final.xyz: step 3 final state
    "ts_sella",                 # harmonic/ts_sella.xyz: step 4 result
    "neb_start",                # neb.traj, first band only: the band step 6 starts from
    "neb_band",                 # neb_band.traj: the band where step 6 stopped
)                               # also there, without MACE values: "raw" (<stem>_init.xyz), "band" (band.traj)
COLLECT_MIN_DIST = 0.7          # A, a structure with two atoms closer than this is dropped
COLLECT_MAX_FORCE = 20.0        # eV/A, a structure with a larger stored MACE force on any atom
                                # is dropped (a structure with no stored force: distance check only)
COLLECT_ALONE = 1.5             # bond check: an adsorbate atom with no atom within this x (r_i + r_j)
                                # is alone. Bonded is settings.TS_HARMONIC_BOND_CUTOFF x (r_i + r_j)

SELECT_WRITE = True            # False: pick and report only (dry run)
SELECT_N = 150                 # picks for each reaction in this round: you choose it
SELECT_SOAP = dict(             # DScribe SOAP on the adsorbate, the site Al and its first-shell O
    r_cut=5.0,                  # A, how far each atom looks
    n_max=8,                    # radial functions
    l_max=6,                    # angular functions
    sigma=0.5,                  # A, width of the atom Gaussians
)
SELECT_SAME = 1e-6              # SOAP distance below which a structure is a copy of a picked one

# --------------------------------------------------------------------------
# dft step 2: VASP single points on the collected configs (one SLURM job per
# config). Every config of a training set must use the same values below.
# --------------------------------------------------------------------------

SP_SUBMIT = True               # False: write the inputs + job.sh only (dry run)
SP_LIMIT = None                 # submit at most this many new configs per run;
                                # set 2-3 for a timing test first
SP_LANES = 6                    # at most this many jobs running at a time (6 x 32 = 192 cores)

DFT_DEVICE = "cpu"              # "cpu" or "gpu": the VASP build dft/02 runs; the values of
                                # both are in scripts/machine.py. dft/03 is always CPU
if DFT_DEVICE not in ("cpu", "gpu"):
    sys.exit('dft_settings.py: DFT_DEVICE = %r is not "cpu" or "gpu"' % DFT_DEVICE)

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

GPU_VASP_CHANGES = dict(        # applied on top of SP_VASP when DFT_DEVICE = "gpu"
    nsim=16,                    # bands worked on at one time: speed only, same energy
)

SP_SLURM_TASKS = 32             # MPI ranks on one node
SP_SLURM_MEMORY = "64G"
SP_SLURM_TIME = "00:30:00"

# --------------------------------------------------------------------------
# dft step 3: isolated atoms for MACE's E0s (same settings as step 2, plus
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

# --------------------------------------------------------------------------
# dft step 4: training set from the finished single points
# --------------------------------------------------------------------------

TRAIN_MODELS = 2                # how many models to train on this round: model_1 .. model_<N>
TRAIN_TEST_FRACTION = 0.1       # share of all structures in the test set (the same for every model)
TRAIN_VALID_FRACTION = 0.1      # share of all structures each model holds out as valid
                                # (a different draw for each model); the rest is train
TRAIN_SEED = 0                  # fixes the draws: the same seed gives the same split
TRAIN_MAX_FORCE = 20.0          # eV/A, a structure with a larger DFT force on any atom is
                                # left out; None: no limit