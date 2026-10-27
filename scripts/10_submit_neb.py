#!/usr/bin/env python
"""Step 10 -- submit one climbing-image NEB with MACE per band that step 9
built (see pyntaz/neb.py).

    <RUN_DIR>/NEB/<i>_rxn/pair_<k>/<stem>/band.traj   from step 9, the input
    <RUN_DIR>/NEB/<i>_rxn/pair_<k>/<stem>/job.sh      runs scripts/neb_one.py

The job runs in the guess folder and writes neb.traj, neb.log,
neb_band.traj, neb.png and, last, neb_result.json there.

Guesses with neb_result.json are reported with their verdict and skipped;
guesses submitted before whose job has not finished (job.id present) are
left alone -- delete job.id to resubmit a failed one, and rerun step 9 first
if you changed the band settings. NEB_REACTIONS picks reactions and
NEB_MAX_PER_REACTION keeps only the lowest-barrier guesses of each (Ea_fwd
from step 9's neb.json, MACE).

settings.SUBMIT = False writes job.sh but submits nothing (dry run). Each
job runs scripts/neb_one.py from this checkout, so keep it where it is
until the jobs have started.
"""

import json
import os
import subprocess
import sys

import _common  # noqa: F401
import layout
import machine
import settings

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neb_one.py")

JOB_TEMPLATE = """#!/bin/bash
%(header)s

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=%(repo)s:$PYTHONPATH

# the env python directly: `conda activate` fails silently in batch jobs
%(python)s %(worker)s %(band)s
"""


def job_script(name):
    return JOB_TEMPLATE % {"header": machine.sbatch_header(settings.NEB_SLURM_CORES,
                                                     settings.SLURM_MEMORY,
                                                     settings.NEB_SLURM_TIME, name=name),
                           "repo": _common.REPO, "python": machine.PYTHON,
                           "worker": WORKER, "band": layout.NEB_BAND}


def load(path):
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def guesses(reaction_dir):
    """[(relative, folder, neb.json)] of every guess with a band, lowest
    forward barrier first."""
    found = []
    for pair, pair_dir in layout.species_dirs(reaction_dir):
        for stem in sorted(os.listdir(pair_dir)):
            folder = os.path.join(pair_dir, stem)
            record = load(os.path.join(folder, layout.NEB_RECORD))
            if record is None or not os.path.isfile(os.path.join(folder, layout.NEB_BAND)):
                continue
            found.append((os.path.join(pair, stem), folder, record))
    return sorted(found, key=lambda item: item[2]["Ea_fwd"])


def missing_machine_paths():
    return [(name, path) for name, path in
            (("PYTHON", machine.PYTHON), ("MACE_MODEL", settings.MACE_MODEL),
             ("WORKER", WORKER))
            if not os.path.exists(path)]


run = layout.RunLayout(settings.RUN_DIR)

for name, path in missing_machine_paths():
    print("%s does not exist: %s" % (name, path))
if missing_machine_paths():
    sys.exit("fix scripts/settings.py or set the PYNTAZ_* environment variables")
if not os.path.isdir(run.neb):
    sys.exit("%s not found -- run step 9 first" % run.neb)

wanted = settings.NEB_REACTIONS
cap = settings.NEB_MAX_PER_REACTION
print("%s: reactions %s, %s per reaction, model %s"
      % (settings.RUN_DIR, "all" if wanted is None else list(wanted),
         "all guesses" if cap is None else "lowest %d" % cap,
         os.path.basename(settings.MACE_MODEL)))

keys = ("submit", "done", "queued", "capped")
totals = dict.fromkeys(keys, 0)
verdicts = {}
for reaction_name, reaction_dir in layout.species_dirs(run.neb):
    head = reaction_name.split("_")[0]
    if not head.isdigit():
        continue
    index = int(head)
    if wanted is not None and index not in wanted:
        continue
    print("\n%s" % reaction_name)
    counts = dict.fromkeys(keys, 0)
    for rank, (relative, folder, record) in enumerate(guesses(reaction_dir)):
        label = "%s  (Ea %.3f eV)" % (relative, record["Ea_fwd"])
        result = load(os.path.join(folder, layout.NEB_RESULT))
        if result is not None:
            print("  already done   %s  -> %s" % (label, result["verdict"]))
            verdicts[result["verdict"]] = verdicts.get(result["verdict"], 0) + 1
            counts["done"] += 1
            continue
        if os.path.exists(os.path.join(folder, layout.JOB_ID)):
            with open(os.path.join(folder, layout.JOB_ID)) as handle:
                print("  in the queue   %s  (job %s, no result yet)" % (label, handle.read().strip()))
            counts["queued"] += 1
            continue
        if cap is not None and rank >= cap:
            counts["capped"] += 1
            continue

        with open(os.path.join(folder, layout.JOB_SCRIPT), "w") as handle:
            handle.write(job_script("neb%d_%s" % (index, relative.split(os.sep)[0][-2:])))
        if settings.SUBMIT:
            out = subprocess.run(["sbatch", "--chdir=" + folder, layout.JOB_SCRIPT],
                                 capture_output=True, text=True)
            print("  %s -> %s" % (label, out.stdout.strip() or out.stderr.strip()))
            if out.returncode != 0:
                sys.exit("sbatch failed, stopping")
            with open(os.path.join(folder, layout.JOB_ID), "w") as handle:
                handle.write(out.stdout.strip().split()[-1] + "\n")
        else:
            print("  would submit   %s" % label)
        counts["submit"] += 1

    print("  %d %s, %d already done, %d in the queue, %d over NEB_MAX_PER_REACTION"
          % (counts["submit"], "submitted" if settings.SUBMIT else "to submit",
             counts["done"], counts["queued"], counts["capped"]))
    for key in keys:
        totals[key] += counts[key]

print("\n%d NEBs %s, %d already done, %d in the queue, %d over the cap  (%s)"
      % (totals["submit"], "submitted" if settings.SUBMIT else "would be submitted",
         totals["done"], totals["queued"], totals["capped"], run.neb))
if verdicts:
    print("verdicts so far: " + ", ".join("%s %d" % item for item in sorted(verdicts.items())))