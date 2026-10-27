"""The cluster the jobs run on: one block of values per machine, picked by
MACHINE in settings.py, and the #SBATCH lines every job.sh starts with.

An empty value ("" or []) writes no #SBATCH line. To add a machine, copy a
block, give it a new name and set MACHINE to that name in settings.py.
"""

import os
import sys

from settings import MACHINE

if MACHINE == "hive":
    # the conda env python directly -- `conda activate` fails silently in batch jobs
    PYTHON = os.path.expanduser("~/.conda/envs/pyntaz-slim/bin/python")
    CPU_ACCOUNT = "ark245grp"
    CPU_PARTITION = "high"
    CPU_QOS = ""
    CPU_CONSTRAINT = ""
    CPU_EXTRA = []              # more #SBATCH options, one string each
    GPU_ACCOUNT = "ark245grp"
    GPU_PARTITION = "gpu-6000-blackwell"
    GPU_QOS = ""
    GPU_CONSTRAINT = ""
    GPU_REQUEST = "--gres=gpu:1"
    GPU_EXTRA = []
elif MACHINE == "perlmutter":
    PYTHON = "SET_ME"
    CPU_ACCOUNT = "SET_ME"
    CPU_PARTITION = "SET_ME"
    CPU_QOS = "SET_ME"
    CPU_CONSTRAINT = "SET_ME"
    CPU_EXTRA = []
    GPU_ACCOUNT = "SET_ME"
    GPU_PARTITION = "SET_ME"
    GPU_QOS = "SET_ME"
    GPU_CONSTRAINT = "SET_ME"
    GPU_REQUEST = "SET_ME"
    GPU_EXTRA = []
else:
    sys.exit('MACHINE = "%s" in scripts/settings.py is not in scripts/machine.py '
             '(known: hive, perlmutter)' % MACHINE)


def sbatch_header(cores, memory, time, name=None):
    """The #SBATCH lines of a one-task CPU job, without a final newline."""
    options = []
    if name:
        options.append("--job-name=%s" % name)
    for option, value in (("--account", CPU_ACCOUNT), ("--partition", CPU_PARTITION),
                          ("--qos", CPU_QOS), ("--constraint", CPU_CONSTRAINT)):
        if value:
            options.append("%s=%s" % (option, value))
    options += CPU_EXTRA
    options += ["--nodes=1", "--ntasks=1", "--cpus-per-task=%d" % cores,
                "--mem=%s" % memory, "--time=%s" % time,
                "--output=job.out", "--error=job.err"]
    return "\n".join("#SBATCH " + option for option in options)