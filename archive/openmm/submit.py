#!/usr/bin/env python
"""Submit one SLURM job per reaction that runs the openmm/ steps in order
for that reaction only:

  01  python 01_make_ts_guess.py <i>             -> guesses/<i>_rxn/
  03  python 03_mace_fixed.py guesses/<i>_rxn    -> mace_relax/<i>_rxn/
  04  python 04_sella_ts.py mace_relax/<i>_rxn   -> sella/<i>_rxn/

STAGES picks which of them the job runs (e.g. only ("04",) to redo Sella
on existing mace_relax/ output). A stage that fails stops the job.

Reads  reaction.yaml   (next to this file); REACTIONS picks which, or all
Writes jobs/<i>_rxn/job.sh, and SLURM writes job.out / job.err there.
Run on the login node from anywhere; SUBMIT = False only writes job.sh.
"""

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from pyntaz.reactions import parse_reactions

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS = os.path.join(HERE, "jobs")

REACTIONS = None                # None: every reaction in reaction.yaml; or e.g. [0, 3]
STAGES = ("01", "03", "04")     # run in this order inside each job
SUBMIT = True                   # False: write job.sh only (dry run)

# machine (same defaults and PYNTAZ_* overrides as scripts/settings.py)
PYTHON = os.environ.get("PYNTAZ_PYTHON", os.path.expanduser("~/.conda/envs/pyntaz-slim/bin/python"))
SLURM_ACCOUNT = os.environ.get("PYNTAZ_SLURM_ACCOUNT", "ark245grp")
SLURM_PARTITION = os.environ.get("PYNTAZ_SLURM_PARTITION", "high")
SLURM_CORES = 8
SLURM_MEMORY = "16G"
SLURM_TIME = "12:00:00"         # 03 + two Sella runs + two frequency checks

COMMANDS = {"01": "%(python)s %(here)s/01_make_ts_guess.py %(index)d",
            "03": "%(python)s %(here)s/03_mace_fixed.py %(here)s/guesses/%(index)d_rxn",
            "04": "%(python)s %(here)s/04_sella_ts.py %(here)s/mace_relax/%(index)d_rxn"}

JOB_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=%(name)s
#SBATCH --account=%(account)s
#SBATCH --partition=%(partition)s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=%(cores)d
#SBATCH --mem=%(memory)s
#SBATCH --time=%(time)s
#SBATCH --output=job.out
#SBATCH --error=job.err

set -e                          # a failed stage stops the job
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=%(repo)s:$PYTHONPATH

# the env python directly: `conda activate` fails silently in batch jobs
%(body)s
"""

if not os.path.exists(PYTHON):
    raise SystemExit("PYTHON does not exist: %s (set PYNTAZ_PYTHON)" % PYTHON)
unknown = [stage for stage in STAGES if stage not in COMMANDS]
if unknown:
    raise SystemExit("unknown stage(s) %s; choose from %s" % (unknown, sorted(COMMANDS)))

with open(os.path.join(HERE, "reaction.yaml")) as handle:
    reactions, _ = parse_reactions(handle.read())
indices = [reaction["index"] for reaction in reactions]
if REACTIONS is not None:
    missing = set(REACTIONS) - set(indices)
    if missing:
        raise SystemExit("no reaction with index %s in reaction.yaml" % sorted(missing))
    indices = [i for i in indices if i in REACTIONS]

for i in indices:
    job_dir = os.path.join(JOBS, "%d_rxn" % i)
    os.makedirs(job_dir, exist_ok=True)
    body = "\n".join("echo \"== %s  $(date)\"\n" % stage
                     + COMMANDS[stage] % {"python": PYTHON, "here": HERE, "index": i}
                     for stage in STAGES)
    with open(os.path.join(job_dir, "job.sh"), "w") as handle:
        handle.write(JOB_TEMPLATE % {"name": "ts%d_%s" % (i, "-".join(STAGES)),
                                     "account": SLURM_ACCOUNT, "partition": SLURM_PARTITION,
                                     "cores": SLURM_CORES, "memory": SLURM_MEMORY,
                                     "time": SLURM_TIME, "repo": REPO, "body": body})
    if SUBMIT:
        result = subprocess.run(["sbatch", "--chdir=" + job_dir, "job.sh"],
                                capture_output=True, text=True)
        print("%d_rxn  %s  ->  %s" % (i, "+".join(STAGES), result.stdout.strip() or result.stderr.strip()))
        if result.returncode != 0:
            raise SystemExit("sbatch failed, stopping")
    else:
        print("%d_rxn  %s  ->  would submit %s" % (i, "+".join(STAGES), os.path.join(job_dir, "job.sh")))

print("\n%d job(s) %s; output in %s/<i>_rxn/job.out"
      % (len(indices), "submitted" if SUBMIT else "written", JOBS))