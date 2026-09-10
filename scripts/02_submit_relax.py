#!/usr/bin/env python
"""Step 2 -- mirror Adsorbates/ into Adsorbates_relax/ and submit one MACE
relaxation job per config.

settings.SUBMIT = False writes job dirs + job.sh only (dry run); True runs
sbatch. Configs that already have a relax.xyz are skipped. Each job runs
scripts/relax_one.py from this checkout, so keep it where it is until the
jobs have started.
"""

import os
import shutil
import subprocess
import sys

import _common  # noqa: F401
import layout
import settings

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "relax_one.py")

JOB_TEMPLATE = """#!/bin/bash
#SBATCH --account=%(account)s
#SBATCH --partition=%(partition)s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=%(cores)d
#SBATCH --mem=%(memory)s
#SBATCH --time=%(time)s
#SBATCH --output=job.out
#SBATCH --error=job.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=%(repo)s:$PYTHONPATH

# the env python directly: `conda activate` fails silently in batch jobs
%(python)s %(worker)s %(xyz)s
"""


def job_script(xyz_name):
    """Text of job.sh for one relaxation; ``xyz_name`` is relative to the
    job directory."""
    return JOB_TEMPLATE % {"account": settings.SLURM_ACCOUNT,
                           "partition": settings.SLURM_PARTITION,
                           "cores": settings.SLURM_CORES,
                           "memory": settings.SLURM_MEMORY,
                           "time": settings.SLURM_TIME,
                           "repo": _common.REPO, "python": settings.PYTHON,
                           "worker": WORKER, "xyz": xyz_name}


def missing_machine_paths():
    """[(name, path)] of the settings paths that do not exist."""
    return [(name, path) for name, path in
            (("PYTHON", settings.PYTHON), ("MACE_MODEL", settings.MACE_MODEL),
             ("WORKER", WORKER))
            if not os.path.exists(path)]


run = layout.RunLayout(settings.RUN_DIR)

for name, path in missing_machine_paths():
    print("%s does not exist: %s" % (name, path))
if missing_machine_paths():
    sys.exit("fix scripts/settings.py or set the PYNTAZ_* environment variables")
if not os.path.isdir(run.adsorbates):
    sys.exit("%s not found -- run step 1 first" % run.adsorbates)

n_jobs = 0
for relative, xyz in layout.initial_guess_files(run.adsorbates):
    if relative.split(os.sep)[0] in settings.SKIP:
        continue

    job_dir = os.path.join(run.relaxed, relative)
    os.makedirs(job_dir, exist_ok=True)
    shutil.copy2(xyz, os.path.join(job_dir, os.path.basename(xyz)))

    if os.path.exists(os.path.join(job_dir, layout.RELAXED_STRUCTURE)):
        print("already done  ", relative)
        continue

    with open(os.path.join(job_dir, layout.JOB_SCRIPT), "w") as handle:
        handle.write(job_script(os.path.basename(xyz)))

    if settings.SUBMIT:
        result = subprocess.run(["sbatch", "--chdir=" + job_dir, layout.JOB_SCRIPT],
                                capture_output=True, text=True)
        print(relative, "->", result.stdout.strip() or result.stderr.strip())
        if result.returncode != 0:
            sys.exit("sbatch failed, stopping")
    else:
        print("would submit  ", relative)
    n_jobs += 1

print("\n%d configs %s" % (n_jobs, "submitted" if settings.SUBMIT else "would be submitted"))
