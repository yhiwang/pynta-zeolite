#!/usr/bin/env python
"""DFT step 2: one VASP single point per structure that dft/02_select_configs.py picked.

    python dft/02_submit_sp.py                    # submit new configs, report the rest
    python dft/02_submit_sp.py --resubmit-failed  # ... and submit the failed ones again

For every ``runs/<MODEL>/<ROUND>/<FRAMEWORK>/DFT_sp/<i>_rxn/<site>/<pair>/<stem>/<name>/config.extxyz``
it writes, into the same folder:

    INCAR POSCAR KPOINTS POTCAR   from SP_VASP in dft/dft_settings.py
    ase-sort.dat                  VASP's element-sorted order -> config order
                                  (step 4 needs it to map forces back)
    job.sh                        runs VASP_SETUP, then VASP_BINARY with VASP_LAUNCH,
                                  then, only if the OUTCAR finished, deletes the
                                  bulky outputs step 4 never reads (POTCAR,
                                  vasprun.xml, vaspout.h5, DOSCAR, ...), so the
                                  home quota only ever holds the jobs running now

and submits it (``job.id`` holds the SLURM id, so a rerun never submits a
queued job twice). With SP_LANES = N the jobs share N names and carry
--dependency=singleton, so all of them can be queued at once while at most
N run at the same time. Every config reports one of:

    done          OUTCAR finished (has the timing block)
    in the queue  job.id present and the job is in squeue
    failed        job.id present, job gone from squeue, OUTCAR not finished
                  -> look at job.err / vasp.out, then rerun with --resubmit-failed
                     (or delete that folder's job.id and rerun)
    resubmitted   failed, and --resubmit-failed sent it again: the outputs of the
                  failed attempt (OUTCAR, vasp.out, job.err, ...) are deleted first,
                  the inputs are kept, job.sh is rewritten
    would submit  inputs written, SP_SUBMIT = False (also "would resubmit")
    submitted     sbatch accepted it (or "not yet": over SP_LIMIT this run)

--resubmit-failed resubmits every failed config, whatever the reason, so read
a few job.err / vasp.out first: a job that hit the time limit or a real VASP
error will just fail again. When squeue cannot be reached nothing counts as
failed, so nothing is resubmitted.

Runs on the login node. Needs ase only (the POTCARs are concatenated here,
from VASP_PP_PATH). dft/03 reuses JOB_TEMPLATE with the same format fields,
so the template takes no new ones.
"""

import os
import subprocess
import sys
from collections import Counter

import _common  # noqa: F401
import dft_settings as ds
import layout
import machine
import settings
from ase.io import read

OUTCAR = "OUTCAR"
VASP_INPUTS = ("INCAR", "POSCAR", "KPOINTS", "POTCAR")
FINISHED_MARK = "General timing and accounting"   # last block of a finished OUTCAR
RESUBMIT_FLAG = "--resubmit-failed"
STALE_OUTPUTS = ("OUTCAR", "OSZICAR", "CONTCAR", "vasprun.xml", "vaspout.h5",
                 "vasp.out", "job.out")            # plus layout.JOB_ERR

# the cleanup line repeats FINISHED_MARK and lists the files by hand on purpose:
# dft/03 formats this template with its own fields, so it can take no new ones
JOB_TEMPLATE = """#!/bin/bash
{header}
#SBATCH --no-requeue
{dependency}
{setup}
export OMP_NUM_THREADS={threads}
cd "$SLURM_SUBMIT_DIR"
{launch} {binary} > vasp.out
if grep -q "General timing and accounting" OUTCAR 2>/dev/null; then
    rm -f POTCAR vasprun.xml vaspout.h5 DOSCAR EIGENVAL PROCAR XDATCAR PCDAT CHG CHGCAR WAVECAR
fi
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
    """sp_R0_T1_p03_neb_band_0003 -- reaction, site, pair, structure."""
    rxn, site, pair, _, name = relative.split(os.sep)[-5:]
    return "sp_R%s_%s_p%s_%s" % (rxn.split("_")[0], site, pair.split("_")[-1], name)


def write_inputs(folder, relative, lane=None):
    """VASP inputs + job.sh, unless the inputs are already there. With a
    ``lane`` the job is named sp_lane<NN> and waits for the previous job of
    that name (--dependency=singleton), so at most SP_LANES run at once."""
    from ase.calculators.vasp import Vasp
    gpu = ds.DFT_DEVICE == "gpu"
    if not all(os.path.isfile(os.path.join(folder, name)) for name in VASP_INPUTS):
        atoms = read(os.path.join(folder, layout.DFT_CONFIG))
        atoms.pbc = True
        params = dict(ds.SP_VASP, **ds.GPU_VASP_CHANGES) if gpu else ds.SP_VASP
        Vasp(directory=folder, **params).write_input(atoms)
    name = job_name(relative) if lane is None else "sp_lane%02d" % lane
    if gpu:                 # one GPU, one MPI rank, threads on the CPUs of the job
        header = machine.sbatch_header(machine.GPU_CPUS, machine.GPU_MEMORY,
                                       ds.SP_SLURM_TIME, name=name, gpu=True)
    else:
        header = machine.sbatch_header(machine.VASP_CPUS_PER_RANK, ds.SP_SLURM_MEMORY,
                                       ds.SP_SLURM_TIME, name=name, tasks=ds.SP_SLURM_TASKS)
    with open(os.path.join(folder, layout.JOB_SCRIPT), "w") as handle:
        handle.write(JOB_TEMPLATE.format(
            header=header,
            dependency="" if lane is None else "#SBATCH --dependency=singleton\n",
            setup="\n".join(machine.VASP_GPU_SETUP if gpu else machine.VASP_SETUP),
            threads=machine.VASP_GPU_THREADS if gpu else 1,
            launch=machine.VASP_GPU_LAUNCH if gpu else machine.VASP_LAUNCH,
            binary=ds.VASP_BINARY))


def clear_failed(folder):
    """Delete what a failed attempt left behind; the inputs and job.id stay
    (submit() overwrites job.id)."""
    for name in STALE_OUTPUTS + (layout.JOB_ERR,):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            os.remove(path)


def submit(folder):
    out = subprocess.run(["sbatch", layout.JOB_SCRIPT], cwd=folder,
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError("sbatch failed in %s:\n%s" % (folder, out.stderr))
    job_id = out.stdout.strip().split()[-1]
    with open(os.path.join(folder, layout.JOB_ID), "w") as handle:
        handle.write(job_id + "\n")
    return job_id


def parse_args(argv):
    unknown = [a for a in argv if a != RESUBMIT_FLAG]
    if unknown:
        sys.exit("unknown argument(s): %s   (the only option is %s)"
                 % (" ".join(unknown), RESUBMIT_FLAG))
    return RESUBMIT_FLAG in argv


def main():
    resubmit = parse_args(sys.argv[1:])
    dft_sp = os.path.join(layout.framework_dir(settings.MODEL, settings.ROUND, settings.FRAMEWORK),
                          layout.DFT_SP)
    if not os.path.isdir(dft_sp):
        sys.exit("%s not found -- run dft/01 and dft/02_select_configs.py first" % dft_sp)
    potpaw = os.path.join(machine.VASP_PP_PATH, "potpaw_PBE")
    if not os.path.isdir(potpaw):
        sys.exit("no POTCARs at %s -- link the lab set there first" % potpaw)
    os.environ["VASP_PP_PATH"] = machine.VASP_PP_PATH

    configs = config_folders(dft_sp)
    if not configs:
        sys.exit("no %s under %s" % (layout.DFT_CONFIG, dft_sp))
    in_queue = queued_ids()
    if resubmit and in_queue is None:
        print("squeue unavailable: cannot tell failed from queued, nothing is resubmitted")
    status = Counter()
    n_new = 0
    for relative, folder in configs:
        if finished(folder):
            status["done"] += 1
            continue
        previous = None                              # job id of a failed attempt
        id_path = os.path.join(folder, layout.JOB_ID)
        if os.path.isfile(id_path):
            with open(id_path) as handle:
                job_id = handle.read().strip()
            if in_queue is None or job_id in in_queue:
                status["in the queue"] += 1
                continue
            if not resubmit:
                status["failed"] += 1
                print("failed   %s  (job %s; see job.err / vasp.out, then rerun with %s)"
                      % (relative, job_id, RESUBMIT_FLAG))
                continue
            previous = job_id
        if ds.SP_LIMIT is not None and n_new >= ds.SP_LIMIT:
            status["not yet (over SP_LIMIT)"] += 1
            continue
        if previous is not None and ds.SP_SUBMIT:
            clear_failed(folder)
        lanes = getattr(ds, "SP_LANES", None)
        write_inputs(folder, relative, None if not lanes else n_new % lanes)
        n_new += 1
        again = previous is not None
        if ds.SP_SUBMIT:
            job_id = submit(folder)
            status["resubmitted" if again else "submitted"] += 1
            print("%s %s  job %s%s" % ("resubmitted" if again else "submitted", relative,
                                       job_id, " (was %s)" % previous if again else ""))
        else:
            status["would resubmit" if again else "would submit"] += 1
            print("%s %s" % ("would resubmit" if again else "would submit", relative))

    print("\n%d configs: " % len(configs)
          + ", ".join("%s %d" % (k, v) for k, v in sorted(status.items())))
    if status["failed"]:
        print("%d failed: read a few job.err / vasp.out, then rerun with %s"
              % (status["failed"], RESUBMIT_FLAG))
    if not ds.SP_SUBMIT and (status["would submit"] or status["would resubmit"]):
        print("dry run: inputs + job.sh written, nothing submitted. Check one INCAR, "
              "then set SP_SUBMIT = True in dft/dft_settings.py")


if __name__ == "__main__":
    main()