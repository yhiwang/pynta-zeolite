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
    CPU_EXTRA = ["--exclude=hive-as-11-2-39"]   # bad node, Nov 2026: jobs fail to start there
    GPU_ACCOUNT = "ark245grp"
    GPU_PARTITION = "gpu-6000-blackwell"
    GPU_QOS = ""
    GPU_CONSTRAINT = ""
    GPU_REQUEST = "--gres=gpu:1"
    GPU_EXTRA = []
    GPU_CPUS = 8                # CPUs and memory of a one-GPU job (mace_ft/01_train.py)
    GPU_MEMORY = "64G"
    # the mace-ft env's trainer, called directly (mace_ft/01_train.py)
    FT_MACE_RUN_TRAIN = os.path.expanduser("~/.conda/envs/mace-ft/bin/mace_run_train")
    # VASP (dft/02, dft/03, scripts/08): lines job.sh runs first, the MPI launch
    # (the binary follows it), CPUs per MPI rank, and the folder with potpaw_PBE/
    VASP_SETUP = ["module load vasp/6.5.1%nvhpc@24.9"]
    VASP_LAUNCH = "mpirun -np $SLURM_NTASKS"
    VASP_CPUS_PER_RANK = 1
    VASP_PP_PATH = os.path.expanduser("~/vasp_pp")   # potpaw_PBE/ is a link to the
                                # lab's grp_shared/PBE64 or PBE54 -- never mix them
    # GPU VASP (DFT_DEVICE = "gpu" in dft/dft_settings.py): not set up on Hive
    VASP_GPU_SETUP = ["SET_ME"]
    VASP_GPU_LAUNCH = "SET_ME"
    VASP_GPU_THREADS = "SET_ME"
elif MACHINE == "perlmutter":
    PYTHON = "/global/common/software/m3548/ivan1007/envs/pyntaz-slim/bin/python"
    CPU_ACCOUNT = "m3548"
    CPU_PARTITION = ""
    CPU_QOS = "shared"
    CPU_CONSTRAINT = "cpu"
    CPU_EXTRA = []
    GPU_ACCOUNT = "m3548_g"     # NERSC GPU jobs use the project's _g account
    GPU_PARTITION = ""
    GPU_QOS = "shared"          # NERSC: jobs with 1 or 2 GPUs use the shared QOS
    GPU_CONSTRAINT = "gpu"
    GPU_REQUEST = "--gpus=1"
    GPU_EXTRA = []
    GPU_CPUS = 32               # the shared GPU queue requires exactly 32 cores per GPU
    GPU_MEMORY = "48G"          # Slurm counts memory as cores too: about 54G at most for 32
    FT_MACE_RUN_TRAIN = "/global/common/software/m3548/ivan1007/envs/mace-ft/bin/mace_run_train"
    # CPU VASP, tested Nov 2026: 32 ranks, 114 s for a 289-atom single point
    VASP_SETUP = ["module load vasp/6.4.3-cpu",
                  "export OMP_PLACES=threads", "export OMP_PROC_BIND=spread"]
    VASP_LAUNCH = "srun -n $SLURM_NTASKS -c 2 --cpu-bind=cores"
    VASP_CPUS_PER_RANK = 2      # NERSC: never less than 2 on a CPU node
    VASP_PP_PATH = "/global/common/software/nersc9/vasp/dependencies/pseudopotentials/PBE"
    # GPU VASP (DFT_DEVICE = "gpu" in dft/dft_settings.py), tested Nov 2026: 86 s
    # for the same single point. One MPI rank on one GPU; the threads are for
    # the parts of VASP that stay on the CPU, and 16 is every real core of the
    # 32 CPUs of a one-GPU job (32 threads were slower)
    VASP_GPU_SETUP = ["module load vasp/6.4.3-gpu",
                      "export OMP_PLACES=threads", "export OMP_PROC_BIND=spread"]
    VASP_GPU_LAUNCH = "srun -n 1 -c $SLURM_CPUS_PER_TASK -G 1 --cpu-bind=cores --gpu-bind=none"
    VASP_GPU_THREADS = 16
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
