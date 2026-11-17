#!/usr/bin/env python
"""Step 5 as a job -- submit scripts/05_collect_neb.py for the current
MODEL, ROUND and FRAMEWORK (every site of it) instead of running it on the
login node.

    runs/<MODEL>/<ROUND>/<FRAMEWORK>/collect_neb.sh    the job script
    runs/<MODEL>/<ROUND>/<FRAMEWORK>/collect_neb.out   what step 5 prints
    runs/<MODEL>/<ROUND>/<FRAMEWORK>/collect_neb.err   errors, if any

The job script holds MODEL, ROUND and FRAMEWORK (PYNTAZ_JOB_MODEL_NAME,
PYNTAZ_JOB_ROUND, PYNTAZ_JOB_FRAMEWORK; settings.py reads them), so MODEL
can change for the next submission while this job waits in the queue. Every
other setting is read when the job starts.

settings.SUBMIT = False writes collect_neb.sh but submits nothing (dry run).
Do not submit a second job for the same MODEL and FRAMEWORK while the first
one runs: both would write the same folders.
"""

import os
import subprocess
import sys

import _common
import layout
import machine
import settings

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "05_collect_neb.py")
JOB_SCRIPT = "collect_neb.sh"
JOB_OUT = "collect_neb.out"
JOB_ERR = "collect_neb.err"

JOB_TEMPLATE = """#!/bin/bash
%(header)s

# the run this job was submitted for (settings.py reads these)
export PYNTAZ_JOB_MODEL_NAME=%(model)s
export PYNTAZ_JOB_ROUND=%(round)s
export PYNTAZ_JOB_FRAMEWORK=%(framework)s

# the env python directly: `conda activate` fails silently in batch jobs;
# -u: every line goes to collect_neb.out as soon as it is printed
%(python)s -u %(worker)s
"""

folder = os.path.join(_common.REPO, layout.RUNS, settings.MODEL, settings.ROUND,
                      settings.FRAMEWORK)
if not os.path.isdir(folder):
    sys.exit("%s not found -- run steps 0-4 first" % folder)
for name, path in (("PYTHON", machine.PYTHON), ("WORKER", WORKER)):
    if not os.path.exists(path):
        sys.exit("%s does not exist: %s" % (name, path))

header = machine.sbatch_header(settings.NEB_COLLECT_SLURM_CORES,
                               settings.NEB_COLLECT_SLURM_MEMORY,
                               settings.NEB_COLLECT_SLURM_TIME,
                               name="collect_%s" % settings.MODEL)
script_path = os.path.join(folder, JOB_SCRIPT)
with open(script_path, "w") as handle:
    handle.write(JOB_TEMPLATE % {"header": header, "model": settings.MODEL,
                                 "round": settings.ROUND,
                                 "framework": settings.FRAMEWORK,
                                 "python": machine.PYTHON, "worker": WORKER})

label = "%s %s %s (sites %s)" % (settings.MODEL, settings.ROUND, settings.FRAMEWORK,
                                 ", ".join(settings.SITES))
if settings.SUBMIT:
    # step 5 finds the runs relative to the repo, so the job runs there; the
    # options here win over the job.out / job.err lines of the header
    out = subprocess.run(["sbatch", "--chdir=" + _common.REPO,
                          "--output=" + os.path.join(folder, JOB_OUT),
                          "--error=" + os.path.join(folder, JOB_ERR), script_path],
                         capture_output=True, text=True)
    print("%s -> %s" % (label, out.stdout.strip() or out.stderr.strip()))
    if out.returncode != 0:
        sys.exit("sbatch failed")
    print("output: %s" % os.path.join(folder, JOB_OUT))
else:
    print("would submit   %s" % label)
    print("job script: %s" % script_path)