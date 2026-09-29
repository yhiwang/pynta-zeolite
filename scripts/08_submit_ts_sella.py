#!/usr/bin/env python
"""Step 8 -- set up TS_sella/ from TS_harmonic/ and submit one Sella saddle
search per TS guess and start (see pyntaz/ts_sella.py).

Every guess is searched from each start in settings.TS_SELLA_STARTS, in its
own folder, so the two can be compared side by side:

    harmonic   step 7's ts_harmonic.xyz (framework relaxed, bonds pulled
               to the spring targets)
    raw        the unrelaxed step 5 guess, <stem>_init.xyz

    <RUN_DIR>/TS_sella/<i>_rxn/pair_<k>/<stem>/<start>/start.xyz       copied
    <RUN_DIR>/TS_sella/<i>_rxn/pair_<k>/<stem>/<start>/ts_bonds.json   copied
    <RUN_DIR>/TS_sella/<i>_rxn/pair_<k>/<stem>/<start>/job.sh

A "harmonic" start whose step 7 job has not finished (no ts_harmonic.xyz
yet) is reported as waiting and not submitted; run this again later to pick
it up. Starts that already have result.json are skipped, and so are starts
submitted before whose job has not written result.json yet (job.id is in
their folder; delete the folder to resubmit a failed one). Needs no RMG: the
reacting bonds come from step 7's ts_bonds.json.

settings.SUBMIT = False writes everything but job submission (dry run).
Each job runs scripts/ts_sella_one.py from this checkout, so keep it where
it is until the jobs have started.
"""

import os
import shutil
import subprocess
import sys

import _common  # noqa: F401
import layout
import settings

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ts_sella_one.py")
STARTS = ("harmonic", "raw")

JOB_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=%(name)s
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


def job_script(name):
    """Text of job.sh for one start; the worker reads start.xyz from the
    job directory."""
    return JOB_TEMPLATE % {"name": name,
                           "account": settings.SLURM_ACCOUNT,
                           "partition": settings.SLURM_PARTITION,
                           "cores": settings.TS_SELLA_SLURM_CORES,
                           "memory": settings.SLURM_MEMORY,
                           "time": settings.TS_SELLA_SLURM_TIME,
                           "repo": _common.REPO, "python": settings.PYTHON,
                           "worker": WORKER, "xyz": layout.TS_START}


def start_file(config_dir, start):
    """The structure a start begins from, inside a TS_harmonic config dir."""
    if start == "harmonic":
        return os.path.join(config_dir, layout.TS_HARMONIC_STRUCTURE)
    return layout.initial_guess_path(config_dir)


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
if not os.path.isdir(run.ts_harmonic):
    sys.exit("%s not found -- run step 7 first" % run.ts_harmonic)
unknown = [start for start in settings.TS_SELLA_STARTS if start not in STARTS]
if unknown:
    sys.exit("unknown TS_SELLA_STARTS %s; choose from %s" % (unknown, list(STARTS)))

wanted = settings.TS_SELLA_REACTIONS
print("%s: reactions %s, starts %s"
      % (settings.RUN_DIR, "all" if wanted is None else list(wanted),
         ", ".join(settings.TS_SELLA_STARTS)))

totals = {"submit": 0, "done": 0, "queued": 0, "waiting": 0, "skipped": 0}
for reaction_name, reaction_dir in layout.species_dirs(run.ts_harmonic):
    index = int(reaction_name.split("_")[0])
    if wanted is not None and index not in wanted:
        continue
    print("\n%s" % reaction_name)
    target_reaction = os.path.join(run.ts_sella, reaction_name)
    os.makedirs(target_reaction, exist_ok=True)
    shutil.copy2(os.path.join(reaction_dir, layout.REACTION_INFO),
                 os.path.join(target_reaction, layout.REACTION_INFO))

    counts = {"submit": 0, "done": 0, "queued": 0, "waiting": 0, "skipped": 0}
    for relative, _ in layout.initial_guess_files(reaction_dir):
        config_dir = os.path.join(reaction_dir, relative)
        bonds_json = os.path.join(config_dir, layout.TS_BONDS_JSON)
        pair = relative.split(os.sep)[0]
        if not os.path.isfile(bonds_json):
            print("  skipped        %s  (step 7 did not set it up: no %s)"
                  % (relative, layout.TS_BONDS_JSON))
            counts["skipped"] += 1
            continue
        for start in settings.TS_SELLA_STARTS:
            label = "%s/%s" % (relative, start)
            job_dir = os.path.join(target_reaction, relative, start)
            if os.path.exists(os.path.join(job_dir, layout.TS_SELLA_RESULT)):
                print("  already done   %s" % label)
                counts["done"] += 1
                continue
            if os.path.exists(os.path.join(job_dir, layout.JOB_ID)):
                with open(os.path.join(job_dir, layout.JOB_ID)) as handle:
                    print("  in the queue   %s  (job %s, no result yet)" % (label, handle.read().strip()))
                counts["queued"] += 1
                continue
            source = start_file(config_dir, start)
            if not os.path.isfile(source):
                print("  waiting        %s  (no %s yet)" % (label, os.path.basename(source)))
                counts["waiting"] += 1
                continue

            os.makedirs(job_dir, exist_ok=True)
            shutil.copy2(source, os.path.join(job_dir, layout.TS_START))
            shutil.copy2(bonds_json, os.path.join(job_dir, layout.TS_BONDS_JSON))
            with open(os.path.join(job_dir, layout.JOB_SCRIPT), "w") as handle:
                handle.write(job_script("tss%d_%s_%s" % (index, pair[-2:], start[0])))
            if settings.SUBMIT:
                result = subprocess.run(["sbatch", "--chdir=" + job_dir, layout.JOB_SCRIPT],
                                        capture_output=True, text=True)
                print("  %s -> %s" % (label, result.stdout.strip() or result.stderr.strip()))
                if result.returncode != 0:
                    sys.exit("sbatch failed, stopping")
                with open(os.path.join(job_dir, layout.JOB_ID), "w") as handle:
                    handle.write(result.stdout.strip().split()[-1] + "\n")
            else:
                print("  would submit   %s" % label)
            counts["submit"] += 1

    print("  %d %s, %d already done, %d in the queue, %d waiting for step 7, %d guesses skipped"
          % (counts["submit"], "submitted" if settings.SUBMIT else "to submit",
             counts["done"], counts["queued"], counts["waiting"], counts["skipped"]))
    for key in totals:
        totals[key] += counts[key]

print("\n%d starts %s, %d already done, %d in the queue, %d waiting for step 7, "
      "%d guesses skipped  (%s)"
      % (totals["submit"], "submitted" if settings.SUBMIT else "would be submitted",
         totals["done"], totals["queued"], totals["waiting"], totals["skipped"], run.ts_sella))