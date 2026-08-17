#!/usr/bin/env python
"""
Mirror test_run/Adsorbates into test_run/Adsorbates_relax and submit one
relax job per config. Set SUBMIT = False for a dry run.
"""

import os
import shutil
import subprocess
import sys

import paths

SUBMIT = True
SRC = os.path.join("test_run", "Adsorbates")
DST = os.path.join("test_run", "Adsorbates_relax")
SKIP = []
CORES = 4
MEM = "16G"
TIME = "04:00:00"

JOB = """#!/bin/bash
#SBATCH --account=%(account)s
#SBATCH --partition=%(partition)s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=%(cores)d
#SBATCH --mem=%(mem)s
#SBATCH --time=%(time)s
#SBATCH --output=job.out
#SBATCH --error=job.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=%(repo)s:$PYTHONPATH

# the env python directly -- this conda.sh points at an old install path and
# `conda activate` fails silently
%(python)s mace_relax.py %(xyz)s
"""


def job_script(xyz):
    return JOB % {"account": paths.SLURM_ACCOUNT,
                  "partition": paths.SLURM_PARTITION,
                  "cores": CORES, "mem": MEM, "time": TIME,
                  "repo": paths.REPO, "python": paths.PYTHON, "xyz": xyz}


if not paths.check():
    sys.exit("fix paths.py or set the PYNTAZ_* environment variables")
if not os.path.isdir(SRC):
    sys.exit("%s not found -- run from the repo root" % SRC)

worker = os.path.join(paths.REPO, "mace_relax.py")
n = 0

for dirpath, dirs, files in os.walk(SRC):
    inits = [f for f in sorted(files) if f.endswith("_init.xyz")]
    if not inits:
        continue
    if len(inits) > 1:
        sys.exit("%s holds %d _init.xyz files, expected 1" % (dirpath, len(inits)))

    rel = os.path.relpath(dirpath, SRC)
    if rel.split(os.sep)[0] in SKIP:
        continue

    dst = os.path.join(DST, rel)
    os.makedirs(dst, exist_ok=True)
    shutil.copy2(os.path.join(dirpath, inits[0]), os.path.join(dst, inits[0]))
    shutil.copy2(worker, os.path.join(dst, "mace_relax.py"))

    if os.path.exists(os.path.join(dst, "relax.xyz")):
        print("already done  ", rel)
        continue

    with open(os.path.join(dst, "job.sh"), "w") as f:
        f.write(job_script(inits[0]))

    if SUBMIT:
        # list argv, never a shell: species names contain [ ] =
        r = subprocess.run(["sbatch", "--chdir=" + dst, "job.sh"],
                           capture_output=True, text=True)
        print(rel, "->", r.stdout.strip() or r.stderr.strip())
        if r.returncode != 0:
            sys.exit("sbatch failed, stopping")
    else:
        print("would submit  ", rel)
    n += 1

print("\n%d configs %s" % (n, "submitted" if SUBMIT else "would be submitted"))