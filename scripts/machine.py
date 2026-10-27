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
    # the mace-ft env's trainer, called directly (mace_ft/01_train.py)
    FT_MACE_RUN_TRAIN = os.path.expanduser("~/.conda/envs/mace-ft/bin/mace_run_train")
    # VASP (dft/02, dft/04, scripts/12): lines job.sh runs first, the MPI launch
    # (the binary follows it), CPUs per MPI rank, and the folder with potpaw_PBE/
    VASP_SETUP = ["module load vasp/6.5.1%nvhpc@24.9"]
    VASP_LAUNCH = "mpirun -np $SLURM_NTASKS"
    VASP_CPUS_PER_RANK = 1
    VASP_PP_PATH = os.path.expanduser("~/vasp_pp")   # potpaw_PBE/ is a link to the
                                # lab's grp_shared/PBE64 or PBE54 -- never mix them
elif MACHINE == "perlmutter":
    PYTHON = "/global/common/software/m3548/ivan1007/envs/pyntaz-slim/bin/python"
    CPU_ACCOUNT = "m3548"
    CPU_PARTITION = ""
    CPU_QOS = "shared"
    CPU_CONSTRAINT = "cpu"
    CPU_EXTRA = []
    GPU_ACCOUNT = "SET_ME"
    GPU_PARTITION = "SET_ME"
    GPU_QOS = "SET_ME"
    GPU_CONSTRAINT = "SET_ME"
    GPU_REQUEST = "SET_ME"
    GPU_EXTRA = []
    FT_MACE_RUN_TRAIN = "SET_ME"
    VASP_SETUP = ["SET_ME"]
    VASP_LAUNCH = "SET_ME"
    VASP_CPUS_PER_RANK = "SET_ME"
    VASP_PP_PATH = "SET_ME"
else:
    sys.exit('MACHINE = "%s" in scripts/settings.py is not in scripts/machine.py '
             '(known: hive, perlmutter)' % MACHINE)


def sbatch_header(cores, memory, time, name=None, gpu=False, tasks=1):
    """The #SBATCH lines of a CPU job (``tasks`` tasks, ``cores`` CPUs each),
    or of a one-GPU job with gpu=True, without a final newline."""
    if gpu:
        account, partition, qos, constraint, extra = (
            GPU_ACCOUNT, GPU_PARTITION, GPU_QOS, GPU_CONSTRAINT, GPU_EXTRA)
    else:
        account, partition, qos, constraint, extra = (
            CPU_ACCOUNT, CPU_PARTITION, CPU_QOS, CPU_CONSTRAINT, CPU_EXTRA)
    options = []
    if name:
        options.append("--job-name=%s" % name)
    for option, value in (("--account", account), ("--partition", partition),
                          ("--qos", qos), ("--constraint", constraint)):
        if value:
            options.append("%s=%s" % (option, value))
    options += extra
    options.append("--nodes=1")
    if not gpu:
        options.append("--ntasks=%s" % tasks)
    elif GPU_REQUEST:
        options.append(GPU_REQUEST)
    options += ["--cpus-per-task=%s" % cores, "--mem=%s" % memory, "--time=%s" % time,
                "--output=job.out", "--error=job.err"]
    return "\n".join("#SBATCH " + option for option in options)
