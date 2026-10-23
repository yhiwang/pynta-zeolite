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
SITES = "T4"       # one T label ("T4") for a single Al, or a pair name
                           # from the table step 0 prints ("T2-T4_5.65") for two
RUN_DIR = os.path.join("runs", "%s_%s" % (CODE, SITES))
REACTIONS = "reaction.yaml"

# Which sampling -> DFT -> fine-tune round this is. Used by dft/01 (configs go
# to DFT_sp/<ROUND>/<source>/), scripts/archive_round.py (TS_harmonic,
# TS_sella, NEB move to <RUN_DIR>/<ROUND>/) and mace_ft (FT_NAME). To start
# the next round: set ROUND and point MACE_MODEL at the new fine-tuned model.
ROUND = "round3"

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
# MACE_MODEL = os.environ.get(
#     "PYNTAZ_MODEL", os.path.join(REPO, "models", "mace-mpa-0-medium.model"))

# the model steps 02-10 sample with: the fine-tuned model of the previous
# round (copy it into models/ and change this line when a round ends)
MACE_MODEL = os.environ.get(
    "PYNTAZ_MODEL", os.path.join(REPO, "models", "mace-mpa0-ft-MOR_T4-round2-e1f10.model"))

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
TS_TORSION_STEPS = 6           # steps over 360 deg for every free torsion
TS_ROLL_STEPS = 24              # steps over 360 deg about the O-O axis
TS_SPAN_TOLERANCE = 0.4         # A, |fitted X-X span - d(O-O)| allowed
TS_SITE_SCALE_RANGE = (1.0, 1.2, 0.1)   # X-atom bond scale tried: start, stop, step
TS_BOND_STRETCH = {"form": 1.2, "break": 1.2}   # length factors for changing bonds
TS_CLEARANCE_MIN = 1.5          # A, a guess whose best roll clears less is not written

# --------------------------------------------------------------------------
# step 6: filtering the TS guesses
# --------------------------------------------------------------------------

TS_RMSD_THRESHOLD = 2.0         # A, guesses closer than this (adsorbate RMSD, same
                                # pair and direction) are one guess; the roomiest
                                # member stands for the cluster. ~0.8 merges a 60 deg
                                # tail rotation, ~2 would also merge the C=C swaps
TS_KEEP_PER_DIRECTION = None    # cap on clusters kept per pair and direction, or
                                # None for no cap; 1 keeps only the roomiest guess
                                # of flip0 and of flip1

# --------------------------------------------------------------------------
# step 7: harmonic pre-relax of the TS guesses (one SLURM job per guess;
# SUBMIT, PYTHON, MACE_MODEL, account, partition and memory as in step 2)
# --------------------------------------------------------------------------

TS_HARMONIC_REACTIONS = None    # None: every reaction in TS_unique; or e.g. [0, 3]
TS_HARMONIC_STATES = ("ts", "initial", "final")   # one job per guess and state; drop any
TS_HARMONIC_MODES = {           # per state: which bonds get springs, spring target
                                # (r_i + r_j) x mult (r = covalent radius), and whether
                                # a spring-free relax follows; "order" bonds never get one
    "ts":      {"springs": ("form", "break"), "mult": {"form": 1.2, "break": 1.2},
                "free_relax": False},
    "initial": {"springs": ("break",), "mult": {"form": 1.0, "break": 1.0},
                "free_relax": True},
    "final":   {"springs": ("form",), "mult": {"form": 1.0, "break": 1.0},
                "free_relax": True},
}
TS_HARMONIC_FREE_STEPS = 300    # BFGS cap of the spring-free relax (initial / final)
TS_HARMONIC_BOND_CUTOFF = 1.25  # bonded if r < (r_i + r_j) x this, for the endpoint check
TS_HARMONIC_K = 30.0            # eV/A^2, spring constant
TS_HARMONIC_FRAMEWORK_RADIUS = 4.0   # A, framework atoms this close to the adsorbate
                                # relax in the framework stage; use the same free
                                # region in the saddle search after this
TS_HARMONIC_STEPS = {"framework": 100, "spectators": 20, "reacting": 100}  # BFGS cap per stage
TS_HARMONIC_MAX_CYCLES = 3      # framework -> spectators -> reacting cycles at most
TS_HARMONIC_E_TOL = 0.01        # eV, stop cycling when a cycle changes E by less
TS_HARMONIC_FMAX = 0.05         # eV/A, per stage
TS_HARMONIC_SLURM_CORES = 4
TS_HARMONIC_SLURM_TIME = "01:00:00"

# --------------------------------------------------------------------------
# step 8: Sella saddle search on the TS guesses (one SLURM job per guess and
# start; SUBMIT, PYTHON, MACE_MODEL, account, partition and memory as in step 2)
# --------------------------------------------------------------------------

TS_SELLA_REACTIONS = None       # None: every reaction in TS_harmonic; or e.g. [0, 3]
TS_SELLA_STARTS = ("harmonic", "raw")   # "harmonic": step 7's ts_harmonic.xyz,
                                # "raw": the unrelaxed step 5 guess; drop one to skip it
TS_SELLA_FMAX = 0.05            # eV/A, converged when every free atom is below this
TS_SELLA_STEPS = 300            # Sella steps at most
TS_SELLA_INTERNAL = False       # Sella internal coordinates; try True if Cartesian stalls
TS_SELLA_FREE_RADIUS = TS_HARMONIC_FRAMEWORK_RADIUS   # A, the same region step 7 relaxed
TS_SELLA_VIB_DELTA = 0.01       # A, finite-difference step of the frequency check
TS_SELLA_IMAG_CUTOFF = 50.0     # cm-1, smaller imaginary modes count as noise
TS_SELLA_MODE_MIN = 0.2         # the one imaginary mode must stretch a reacting bond by
                                # at least this (unit mode) for the verdict "ts"
TS_SELLA_SLURM_CORES = 4
TS_SELLA_SLURM_TIME = "03:00:00"

# --------------------------------------------------------------------------
# step 9: the starting band for the NEB (built on the login node, next to
# initial.xyz / ts.xyz / final.xyz in NEB/.../<stem>/)
# --------------------------------------------------------------------------

NEB_IMAGES_BEFORE = 2           # images between the initial state and the TS
NEB_IMAGES_AFTER = 2            # images between the TS and the final state
                                # (band = IS + before + TS + after + FS)
NEB_INTERPOLATION = "idpp"      # "idpp" or "linear", for each half-band
NEB_FREE_RADIUS = TS_HARMONIC_FRAMEWORK_RADIUS   # A, framework atoms this close to
                                # the adsorbate in IS, TS or FS move in the NEB
# changing any of the four above rebuilds the band of every guess whose NEB
# has not been submitted yet; bands with a job.id or a result are left alone

# --------------------------------------------------------------------------
# step 10: climbing-image NEB with MACE (one SLURM job per guess; SUBMIT,
# PYTHON, MACE_MODEL, account, partition and memory as in step 2)
# --------------------------------------------------------------------------

NEB_REACTIONS = None            # None: every reaction in NEB/; or e.g. [0, 3]
NEB_MAX_PER_REACTION = None     # submit only the N lowest-barrier guesses of each
                                # reaction (Ea_fwd from step 9), or None for all
NEB_RELAX_ENDPOINTS = True      # relax IS and FS once more with the band's free atoms
                                # and this model before the NEB (to NEB_FMAX)
NEB_SPRING = 0.1                # eV/A^2, spring constant between images
NEB_METHOD = "improvedtangent"  # ASE NEB tangent
NEB_FMAX_ROUGH = 0.25            # eV/A, first stage: band without climbing
NEB_STEPS_ROUGH = 300
NEB_CLIMB = True                # second stage: climbing image
NEB_FMAX = 0.05                 # eV/A, converged when the climbing band is below this
NEB_STEPS = 1000
NEB_TS_RMSD = 0.25               # A, adsorbate RMSD below which the highest image
                                # counts as the Sella TS (verdict "connected")
NEB_TS_BOND = 0.25              # A, every reacting bond at the top must also be this close
                                # to its Sella TS length (verdict "connected")
NEB_MIN_DEPTH = 0.05            # eV, an interior image this far below both
                                # neighbours is an intermediate minimum
NEB_SLURM_CORES = 4
NEB_SLURM_TIME = "12:00:00"

# --------------------------------------------------------------------------
# step 11: collect the NEB TSs for DFT (login node, submits nothing)
# --------------------------------------------------------------------------

TS_COLLECT_VERDICTS = ("connected", "different_ts")   # neb_result.json verdicts to collect
TS_COLLECT_REACTIONS = None     # None: every reaction in NEB/; or e.g. [0, 3]
TS_DUPLICATE_RMSD = 0.1         # A, adsorbate RMSD below which two TSs of a reaction ...
TS_DUPLICATE_ENERGY = 0.02      # eV, ... and energy gap below which they are the same TS

# --------------------------------------------------------------------------
# step 12: DFT Sella on the TSs step 11 collected (one SLURM job per TS;
# VASP settings, module, POTCARs, account, partition, cores and memory from
# dft/dft_settings.py, so the energies match the training data)
# --------------------------------------------------------------------------

DFT_SELLA_FMAX = 0.05           # eV/A, converged when every free atom is below this
DFT_SELLA_STEPS = 200           # Sella steps at most, all resumed parts together
DFT_SELLA_INTERNAL = TS_SELLA_INTERNAL
DFT_SELLA_VASP_CHANGES = dict(lwave=True)   # on top of SP_VASP: reuse WAVECAR each step
DFT_SELLA_KEEP_WAVECAR = False  # delete WAVECAR and other bulky VASP files once converged
DFT_SELLA_LANES = 8             # at most this many running at once (None: no limit)
DFT_SELLA_SLURM_TIME = "24:00:00"   # a job that runs out continues with --resume