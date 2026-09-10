"""What this run is and how every step is tuned. Edit here, not in the
step scripts: which framework and Al sites, which reactions, the sweep
sizes and thresholds, and the machine the jobs run on.

The values below equal the defaults inside ``pyntaz``; the scripts pass
them explicitly so a change here always takes effect. Machine paths can
also be overridden through the ``PYNTAZ_*`` environment variables.
"""

import os

from _common import REPO

# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

CODE = "MOR"
SITES = "T2-T4_5.65"       # one T label ("T4") for a single Al, or a pair name
                           # from the table step 0 prints ("T2-T4_5.65") for two
RUN_DIR = os.path.join("runs", "%s_%s" % (CODE, SITES))
REACTIONS = "reaction.yaml"

# --------------------------------------------------------------------------
# step 0: framework and sites
# --------------------------------------------------------------------------

SUPERCELL_MIN_LENGTH = 12.0     # A, repeat the unit cell until every edge is at least this
BIDENTATE_MAX_SPAN = 3.5        # A, longest O-O distance that counts as a site pair

# which oxygens are sites when the framework has two Al (a single Al is the
# same either way). Recorded in framework.json, so every later step follows it.
#   "cross"  only pairs with one O on each Al (the O-Si-O bridges) and their
#            oxygens -- the same-Al chemistry is left to that Al's single run
#   "all"    every first-shell O of both Al and every pair within the span,
#            for single-T reactions followed by double-T ones in one framework
SITE_RULE = "all"

# --------------------------------------------------------------------------
# step 1: placement
# --------------------------------------------------------------------------

GAS_VACUUM = 10.0               # A of vacuum around a gas-phase molecule
# sweep sizes (24 spin angles, 24 phi x 12 psi) are set in pyntaz/placement.py

# --------------------------------------------------------------------------
# step 2: relaxation and SLURM
# --------------------------------------------------------------------------

SUBMIT = True                   # False: write job dirs + job.sh only (dry run)
SKIP = ()                       # species names to leave out
RELAX_FMAX = 0.05               # eV/A
RELAX_MAX_STEPS = 100

# the conda env python directly -- `conda activate` fails silently in batch jobs
PYTHON = os.environ.get(
    "PYNTAZ_PYTHON", os.path.expanduser("~/.conda/envs/pyntaz-slim/bin/python"))
# a model *file*: a bare model name makes MACE try to download it, which
# fails on a compute node without internet
MACE_MODEL = os.environ.get(
    "PYNTAZ_MODEL", os.path.join(REPO, "models", "mace-mpa-0-medium.model"))

SLURM_ACCOUNT = os.environ.get("PYNTAZ_SLURM_ACCOUNT", "ark245grp")
SLURM_PARTITION = os.environ.get("PYNTAZ_SLURM_PARTITION", "high")
SLURM_CORES = 4
SLURM_MEMORY = "16G"
SLURM_TIME = "04:00:00"

# --------------------------------------------------------------------------
# step 3: filtering
# --------------------------------------------------------------------------

BOND_CHANGE_THRESHOLD = 0.5     # A, a bond that moved more than this has broken
RMSD_THRESHOLD = 2.0            # A, below this two relaxed configs are the same minimum

# --------------------------------------------------------------------------
# step 5: TS guesses from the reaction graph
# --------------------------------------------------------------------------

TS_MAX_SPAN = 3.5               # A, O-O pairs further apart than this are not tried
                                # (which pairs: the SITE_RULE saved with the framework)
TS_TORSION_STEPS = 24           # steps over 360 deg for every free torsion
TS_ROLL_STEPS = 24              # steps over 360 deg about the O-O axis
TS_SPAN_TOLERANCE = 0.2         # A, |fitted X-X span - d(O-O)| allowed
TS_SITE_SCALE_RANGE = (1.0, 2.0, 0.05)   # X-atom bond scale tried: start, stop, step
TS_BOND_STRETCH = {"form": 1.0, "break": 1.0}   # length factors for changing bonds
TS_CLEARANCE_MIN = 1.5          # A, a guess whose best roll clears less is not written
