#!/usr/bin/env python
"""DFT step 2: one VASP single point per config collected by step 1.

    python dft/02_submit_sp.py        # SP_SUBMIT = False: write inputs + job.sh only

For every ``<RUN_DIR>/DFT_sp/.../f<frame>/config.extxyz`` it writes, into
the same folder:

    INCAR POSCAR KPOINTS POTCAR   from SP_VASP in dft/dft_settings.py
    ase-sort.dat                  VASP's element-sorted order -> config order
                                  (step 3 needs it to map forces back)
    job.sh                        loads VASP_MODULE, runs VASP_BINARY with mpirun

and submits it (``job.id`` holds the SLURM id, so a rerun never submits a
queued job twice). With SP_LANES = N the jobs share N names and carry
--dependency=singleton, so all of them can be queued at once while at most
N run at the same time. Every config reports one of:

    done          OUTCAR finished (has the timing block)
    in the queue  job.id present and the job is in squeue
    failed        job.id present, job gone from squeue, OUTCAR not finished
                  -> look at job.err / vasp.out, then delete job.id to resubmit
    would submit  inputs written, SP_SUBMIT = False
    submitted     sbatch accepted it (or "not yet": over SP_LIMIT this run)

Runs on the login node. Needs ase only (the POTCARs are concatenated here,
from VASP_PP_PATH).
"""

import os
import subprocess
import sys
from collections import Counter

import _common  # noqa: F401
import dft_settings as ds
import layout
import settings
from ase.io import read

OUTCAR = "OUTCAR"
VASP_INPUTS = ("INCAR", "POSCAR", "KPOINTS", "POTCAR")
FINISHED_MARK = "General timing and accounting"   # last block of a finished OUTCAR

JOB_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks={tasks}
#SBATCH --cpus-per-task=1
#SBATCH --mem={memory}
#SBATCH --time={time}
#SBATCH --output=job.out
#SBATCH --error={err}
{dependency}
module load {module}
export OMP_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
mpirun -np $SLURM_NTASKS {binary} > vasp.out
"""


def config_folders(root):
    """[(relative, folder)] for every folder holding config.extxyz."""
    found = []
    for dirpath, _, files in os.walk(root):
        if layout.DFT_CONFIG in files:
            found.append((os.path.relpath(dirpath, root), dirpath))
    return sorted(found)


def finished(folder):
    path = os.path.join(folder, OUTCAR)
    if not os.path.isfile(path):
        return False
    with open(path, "rb") as handle:          # the mark is near the end
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell() - 20000))
        return FINISHED_MARK.encode() in handle.read()


def queued_ids():
    """SLURM ids of this user's jobs, or None when squeue is unavailable."""
    try:
        out = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""),
                              "-o", "%i"], capture_output=True, text=True,
                             check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return {line.strip() for line in out.splitlines() if line.strip()}


def job_name(relative):
    """sp_R0_p03_f0012 -- short enough for squeue."""
    rxn, pair, _, start, frame = relative.split(os.sep)
    return "sp_R%s_p%s_%s_%s" % (rxn.split("_")[0], pair.split("_")[-1],
                                start[0], frame)


def write_inputs(folder, relative, lane=None):
    """VASP inputs + job.sh, unless the inputs are already there. With a
    ``lane`` the job is named sp_lane<NN> and waits for the previous job of
    that name (--dependency=singleton), so at most SP_LANES run at once."""
    from ase.calculators.vasp import Vasp
    if not all(os.path.isfile(os.path.join(folder, name)) for name in VASP_INPUTS):
        atoms = read(os.path.join(folder, layout.DFT_CONFIG))
        atoms.pbc = True
        Vasp(directory=folder, **ds.SP_VASP).write_input(atoms)
    with open(os.path.join(folder, layout.JOB_SCRIPT), "w") as handle:
        handle.write(JOB_TEMPLATE.format(
            name=job_name(relative) if lane is None else "sp_lane%02d" % lane,
            dependency="" if lane is None else "#SBATCH --dependency=singleton\n",
            account=ds.SP_SLURM_ACCOUNT,
            partition=ds.SP_SLURM_PARTITION, tasks=ds.SP_SLURM_TASKS,
            memory=ds.SP_SLURM_MEMORY, time=ds.SP_SLURM_TIME, err=layout.JOB_ERR,
            module=ds.VASP_MODULE, binary=ds.VASP_BINARY))


def submit(folder):
    out = subprocess.run(["sbatch", layout.JOB_SCRIPT], cwd=folder,
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError("sbatch failed in %s:\n%s" % (folder, out.stderr))
    job_id = out.stdout.strip().split()[-1]
    with open(os.path.join(folder, layout.JOB_ID), "w") as handle:
        handle.write(job_id + "\n")
    return job_id


def main():
    run = layout.RunLayout(settings.RUN_DIR)
    if not os.path.isdir(run.dft_sp):
        sys.exit("%s not found -- run dft/01_collect_configs.py first" % run.dft_sp)
    potpaw = os.path.join(ds.VASP_PP_PATH, "potpaw_PBE")
    if not os.path.isdir(potpaw):
        sys.exit("no POTCARs at %s -- link the lab set there first" % potpaw)
    os.environ["VASP_PP_PATH"] = ds.VASP_PP_PATH

    configs = config_folders(run.dft_sp)
    if not configs:
        sys.exit("no %s under %s" % (layout.DFT_CONFIG, run.dft_sp))
    in_queue = queued_ids()
    status = Counter()
    n_new = 0
    for relative, folder in configs:
        if finished(folder):
            status["done"] += 1
            continue
        id_path = os.path.join(folder, layout.JOB_ID)
        if os.path.isfile(id_path):
            with open(id_path) as handle:
                job_id = handle.read().strip()
            if in_queue is None or job_id in in_queue:
                status["in the queue"] += 1
            else:
                status["failed"] += 1
                print("failed   %s  (job %s; see job.err / vasp.out, delete job.id "
                      "to resubmit)" % (relative, job_id))
            continue
        if ds.SP_LIMIT is not None and n_new >= ds.SP_LIMIT:
            status["not yet (over SP_LIMIT)"] += 1
            continue
        lanes = getattr(ds, "SP_LANES", None)
        write_inputs(folder, relative, None if not lanes else n_new % lanes)
        n_new += 1
        if ds.SP_SUBMIT:
            job_id = submit(folder)
            status["submitted"] += 1
            print("submitted %s  job %s" % (relative, job_id))
        else:
            status["would submit"] += 1
            print("would submit %s" % relative)

    print("\n%d configs: " % len(configs)
          + ", ".join("%s %d" % (k, v) for k, v in sorted(status.items())))
    if not ds.SP_SUBMIT and status["would submit"]:
        print("dry run: inputs + job.sh written, nothing submitted. Check one INCAR, "
              "then set SP_SUBMIT = True in dft/dft_settings.py")


if __name__ == "__main__":
    main()