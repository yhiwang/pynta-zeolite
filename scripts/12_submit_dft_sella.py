#!/usr/bin/env python
"""Step 12 -- one DFT Sella saddle search per TS that step 11 collected,
starting from the MACE TS (see scripts/dft_sella_one.py).

    python scripts/12_submit_dft_sella.py            # submit new ones, report the rest
    python scripts/12_submit_dft_sella.py --resume   # ... and resubmit the stopped ones

Which TSs: every row of DFT_TS/ts_collected.csv (step 11 does the choosing).
Rerun step 11 first if new NEBs finished.

    <RUN_DIR>/DFT_TS/<i>_rxn/pair_<k>/<stem>/job.id                  the SLURM id; step 11
                                                                    never rewrites this folder
    <RUN_DIR>/DFT_TS/<i>_rxn/pair_<k>/<stem>/dft_sella/start.xyz      copy of ts.xyz
    <RUN_DIR>/DFT_TS/<i>_rxn/pair_<k>/<stem>/dft_sella/ts_info.json   copy
    <RUN_DIR>/DFT_TS/<i>_rxn/pair_<k>/<stem>/dft_sella/job.sh         runs dft_sella_one.py
    ... and the worker's sella.traj, sella.log, ts_dft.xyz, vasp/, result.json there

Every selected TS reports one of:

    done          result.json written (converged or not_converged, steps, energies)
    in the queue  job.id present, the job is in squeue
    stopped       job.id present, job gone, no result.json: usually the time limit
                  -> --resume submits it again and it continues from sella.traj
                     (look at job.err first if it stopped after a few minutes)
    submitted     new, sbatch accepted it (would submit: settings.SUBMIT = False)

A search that ended not_converged (DFT_SELLA_STEPS used up) has a result;
to continue it, raise DFT_SELLA_STEPS, delete dft_sella/result.json and rerun
with --resume.

VASP settings, binary, ranks and memory come from dft/dft_settings.py (SP_VASP
+ DFT_SELLA_VASP_CHANGES), and the module, launch, POTCARs, account and
partition from scripts/machine.py, the same ones dft/02 uses, so the DFT TS
energies are on the same footing as the training data. Disk: each running
job keeps a WAVECAR (a few hundred MB) in vasp/ until it converges, so with
DFT_SELLA_LANES = N the jobs share N names and carry --dependency=singleton:
all can be queued at once while at most N run (None: no limit).

settings.SUBMIT = False writes everything but submits nothing (dry run).
Each job runs scripts/dft_sella_one.py from this checkout, so keep it where
it is until the jobs have started.
"""

import csv
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict

import _common  # noqa: F401
import layout
import machine
import settings

sys.path.insert(0, os.path.join(_common.REPO, "dft"))
import dft_settings as ds  # noqa: E402

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dft_sella_one.py")
RESUME_FLAG = "--resume"
REQUIRED_SETTINGS = ("DFT_SELLA_FMAX", "DFT_SELLA_STEPS", "DFT_SELLA_INTERNAL",
                     "DFT_SELLA_VASP_CHANGES", "DFT_SELLA_KEEP_WAVECAR",
                     "DFT_SELLA_LANES", "DFT_SELLA_SLURM_TIME")
REQUIRED_LAYOUT = ("DFT_SELLA_DIR", "DFT_SELLA_STRUCTURE", "DFT_SELLA_VASP",
                   "TS_COLLECT_CSV", "TS_COLLECT_INFO", "TS_COLLECT_STATES")

JOB_TEMPLATE = """#!/bin/bash
%(header)s
%(dependency)s
%(setup)s
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=%(repo)s:$PYTHONPATH
export VASP_PP_PATH=%(pp)s
export ASE_VASP_COMMAND="%(launch)s %(binary)s"
cd "$SLURM_SUBMIT_DIR"

# the env python directly: `conda activate` fails silently in batch jobs
%(python)s %(worker)s %(start)s
"""


def check_setup():
    """Stop before touching anything when a setting, layout name or path is missing."""
    missing = ["settings." + n for n in REQUIRED_SETTINGS if not hasattr(settings, n)]
    missing += ["layout." + n for n in REQUIRED_LAYOUT if not hasattr(layout, n)]
    if missing:
        sys.exit("missing: %s\nadd them first (see the docstring of dft_sella_one.py "
                 "and this script)" % ", ".join(missing))
    for name, path in (("PYTHON", machine.PYTHON), ("WORKER", WORKER),
                       ("VASP_PP_PATH/potpaw_PBE",
                        os.path.join(machine.VASP_PP_PATH, "potpaw_PBE"))):
        if not os.path.exists(path):
            sys.exit("%s does not exist: %s" % (name, path))


def parse_args(argv):
    unknown = [a for a in argv if a != RESUME_FLAG]
    if unknown:
        sys.exit("unknown argument(s): %s   (the only option is %s)"
                 % (" ".join(unknown), RESUME_FLAG))
    return RESUME_FLAG in argv


def job_script(name, lane=None):
    """With a ``lane`` the job is named ds_lane<NN> and waits for the previous
    job of that name (--dependency=singleton): at most DFT_SELLA_LANES run at
    once, which caps the WAVECARs on disk."""
    if lane is not None:
        name = "ds_lane%02d" % lane
    return JOB_TEMPLATE % {"header": machine.sbatch_header(
                               machine.VASP_CPUS_PER_RANK, ds.SP_SLURM_MEMORY,
                               settings.DFT_SELLA_SLURM_TIME, name=name,
                               tasks=ds.SP_SLURM_TASKS),
                           "dependency": "" if lane is None
                           else "#SBATCH --dependency=singleton\n",
                           "setup": "\n".join(machine.VASP_SETUP),
                           "launch": machine.VASP_LAUNCH, "binary": ds.VASP_BINARY,
                           "pp": machine.VASP_PP_PATH, "repo": _common.REPO,
                           "python": machine.PYTHON, "worker": WORKER,
                           "start": layout.TS_START}


def job_name(row):
    """ds_R1_p02_<n> -- short enough for squeue."""
    return "ds_R%s_p%s" % (row["reaction"], row["pair"].split("_")[-1])


def queued_ids():
    """SLURM ids of this user's jobs, or None when squeue is unavailable."""
    try:
        out = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return {line.strip() for line in out.splitlines() if line.strip()}


def prepare(ts_dir, job_dir, row, lane=None):
    """start.xyz, ts_info.json and job.sh in job_dir; an existing start.xyz is
    kept, so a resumed job never gets a new start."""
    os.makedirs(job_dir, exist_ok=True)
    start = os.path.join(job_dir, layout.TS_START)
    if not os.path.isfile(start):
        shutil.copy2(os.path.join(ts_dir, layout.TS_COLLECT_STATES["ts"]), start)
    shutil.copy2(os.path.join(ts_dir, layout.TS_COLLECT_INFO),
                 os.path.join(job_dir, layout.TS_COLLECT_INFO))
    with open(os.path.join(job_dir, layout.JOB_SCRIPT), "w") as handle:
        handle.write(job_script(job_name(row), lane))


def submit(ts_dir, job_dir):
    out = subprocess.run(["sbatch", layout.JOB_SCRIPT], cwd=job_dir,
                         capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit("sbatch failed in %s:\n%s" % (job_dir, out.stderr))
    job_id = out.stdout.strip().split()[-1]
    with open(os.path.join(ts_dir, layout.JOB_ID), "w") as handle:
        handle.write(job_id + "\n")
    return job_id


def describe(result):
    return "%-13s %3d steps  fmax %.3f  E %.4f  moved %.2f A" % (
        result["verdict"], result["n_steps"], result["fmax"], result["energy"],
        result["max_displacement"])


def main():
    resume = parse_args(sys.argv[1:])
    check_setup()
    run = layout.RunLayout(settings.RUN_DIR)
    table = os.path.join(run.dft_ts, layout.TS_COLLECT_CSV)
    if not os.path.isfile(table):
        sys.exit("%s not found -- run scripts/11_collect_ts.py first" % table)
    with open(table) as handle:
        rows = list(csv.DictReader(handle))
    in_queue = queued_ids()
    if resume and in_queue is None:
        print("squeue unavailable: cannot tell stopped from queued, nothing is resumed")

    status = Counter()
    lanes = settings.DFT_SELLA_LANES
    n_new = 0
    by_reaction = defaultdict(list)
    for row in rows:
        by_reaction[int(row["reaction"])].append(row)
    for index in sorted(by_reaction):
        print("R%-3d %s" % (index, by_reaction[index][0]["reaction_name"]))
        for row in by_reaction[index]:
            ts_dir = os.path.join(run.dft_ts, row["id"])
            job_dir = os.path.join(ts_dir, layout.DFT_SELLA_DIR)
            label = "%-52s dE %.2f" % (row["id"], float(row["dE_ts_min"]))
            result_path = os.path.join(job_dir, layout.TS_SELLA_RESULT)
            id_path = os.path.join(ts_dir, layout.JOB_ID)
            if os.path.isfile(result_path):
                with open(result_path) as handle:
                    result = json.load(handle)
                status["done"] += 1
                print("  done          %s  %s" % (label, describe(result)))
                continue
            again = False
            if os.path.isfile(id_path):
                with open(id_path) as handle:
                    job_id = handle.read().strip()
                if in_queue is None or job_id in in_queue:
                    status["in the queue"] += 1
                    print("  in the queue  %s  job %s" % (label, job_id))
                    continue
                if not resume:
                    status["stopped"] += 1
                    print("  stopped       %s  job %s (rerun with %s)"
                          % (label, job_id, RESUME_FLAG))
                    continue
                again = True
            prepare(ts_dir, job_dir, row, None if not lanes else n_new % lanes)
            n_new += 1
            word = "resubmitted" if again else "submitted"
            if settings.SUBMIT:
                job_id = submit(ts_dir, job_dir)
                status[word] += 1
                print("  %-13s %s  job %s" % (word, label, job_id))
            else:
                plain = word.replace("ted", "")          # submit / resubmit
                status["would " + plain] += 1
                print("  would %-8s %s" % (plain, label))

    print("\n%d TSs picked" % len(rows))
    print(", ".join("%s %d" % item for item in sorted(status.items())))
    if status["stopped"]:
        print("%d stopped: check one job.err, then rerun with %s" % (status["stopped"], RESUME_FLAG))
    if not settings.SUBMIT:
        print("dry run: job.sh written, nothing submitted (settings.SUBMIT = False)")


if __name__ == "__main__":
    main()