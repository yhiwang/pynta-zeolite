#!/usr/bin/env python
"""Step 2 -- mirror Adsorbates/ into Adsorbates_relax/ and submit one MACE
relaxation job per config.

SUBMIT = False writes job dirs + job.sh only (dry run); True runs sbatch.
Configs that already have a relax.xyz are skipped.
"""

import os
import shutil
import subprocess
import sys

from _common import config
from pyntaz import runtree
from pyntaz.relax import slurm_job_script, RELAX_WORKER

RUN_DIR = os.path.join("runs", "MOR_T2-T4_5.65")
SUBMIT = True
SKIP = ()


layout = config.RunLayout(RUN_DIR)

if not config.check_machine_paths():
    sys.exit("fix pyntaz/config.py or set the PYNTAZ_* environment variables")
if not os.path.isdir(layout.adsorbates):
    sys.exit("%s not found -- run step 1 first" % layout.adsorbates)

worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), RELAX_WORKER)
n_jobs = 0
for relative, xyz in runtree.initial_guess_files(layout.adsorbates):
    if relative.split(os.sep)[0] in SKIP:
        continue

    job_dir = os.path.join(layout.relaxed, relative)
    os.makedirs(job_dir, exist_ok=True)
    shutil.copy2(xyz, os.path.join(job_dir, os.path.basename(xyz)))
    shutil.copy2(worker, os.path.join(job_dir, RELAX_WORKER))

    if os.path.exists(os.path.join(job_dir, config.RELAXED_STRUCTURE)):
        print("already done  ", relative)
        continue

    with open(os.path.join(job_dir, "job.sh"), "w") as handle:
        handle.write(slurm_job_script(os.path.basename(xyz)))

    if SUBMIT:
        result = subprocess.run(["sbatch", "--chdir=" + job_dir, "job.sh"],
                                capture_output=True, text=True)
        print(relative, "->", result.stdout.strip() or result.stderr.strip())
        if result.returncode != 0:
            sys.exit("sbatch failed, stopping")
    else:
        print("would submit  ", relative)
    n_jobs += 1

print("\n%d configs %s" % (n_jobs, "submitted" if SUBMIT else "would be submitted"))