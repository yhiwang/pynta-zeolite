#!/usr/bin/env python
"""Step 7 -- mirror TS_unique/ into TS_harmonic/ and submit harmonic relax
jobs (see pyntaz/harmonic_relax.py): one job per TS guess and state.

States (settings.TS_HARMONIC_STATES, defined in settings.TS_HARMONIC_MODES)
all start from the same raw step 5 guess:

    ts        springs on every forming and breaking bond (stretched)
    initial   springs on the breaking bonds only, then a spring-free relax
    final     springs on the forming bonds only, then a spring-free relax

For every reaction in settings.TS_HARMONIC_REACTIONS (None: all of
TS_unique), and every guess step 6 kept:

    <RUN_DIR>/TS_harmonic/<i>_rxn/info.json                         copied from TS_unique
    <RUN_DIR>/TS_harmonic/<i>_rxn/pair_<k>/<stem>/<stem>_init.xyz   copied
    <RUN_DIR>/TS_harmonic/<i>_rxn/pair_<k>/<stem>/ts_bonds.json     reacting bonds
    <RUN_DIR>/TS_harmonic/<i>_rxn/pair_<k>/<stem>/job.sh            ts job
    <RUN_DIR>/TS_harmonic/<i>_rxn/pair_<k>/<stem>/<state>/job.sh    initial / final jobs

The ts job runs in the guess folder itself, where step 8 looks for
ts_harmonic.xyz; the other states each get their own subfolder.

ts_bonds.json holds the forming and breaking bonds as atom indices of that
guess, worked out here from the reaction graph so the job needs no RMG.
Every guess is checked against the graph first (atom count, and the element
at every mapped index); a guess that fails is reported and not submitted.

settings.SUBMIT = False writes everything but job submission (dry run).
A state that already has its result is skipped, and so is one submitted
before whose job has not finished (job.id is in its folder; delete job.id,
or the state's subfolder, to resubmit a failed one). Each job runs
scripts/harmonic_one.py from this checkout, so keep it where it is until
the jobs have started.
"""

import json
import os
import shutil
import subprocess
import sys

import _common  # noqa: F401
import layout
import settings
from ase.io import read
from pyntaz.reactions import molecule_from_adjlist
from pyntaz.ts_graph import TSGraph
from pyntaz.harmonic_relax import check_atom_order, guess_bond_indices

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "harmonic_one.py")

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
%(python)s %(worker)s %(xyz)s %(state)s
"""


def job_script(name, xyz_name, state):
    """Text of job.sh for one guess and state; ``xyz_name`` is relative to
    the job directory."""
    return JOB_TEMPLATE % {"name": name,
                           "account": settings.SLURM_ACCOUNT,
                           "partition": settings.SLURM_PARTITION,
                           "cores": settings.TS_HARMONIC_SLURM_CORES,
                           "memory": settings.SLURM_MEMORY,
                           "time": settings.TS_HARMONIC_SLURM_TIME,
                           "repo": _common.REPO, "python": settings.PYTHON,
                           "worker": WORKER, "xyz": xyz_name, "state": state}


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
if not os.path.isdir(run.ts_unique):
    sys.exit("%s not found -- run steps 5 and 6 first" % run.ts_unique)

n_framework = len(read(os.path.join(settings.RUN_DIR, layout.BARE_XYZ)))
wanted = settings.TS_HARMONIC_REACTIONS
states = list(settings.TS_HARMONIC_STATES)
unknown = [state for state in states if state not in settings.TS_HARMONIC_MODES]
if unknown:
    sys.exit("TS_HARMONIC_STATES has %s, not in TS_HARMONIC_MODES" % unknown)
print("%s: %d framework atoms, reactions %s, states %s"
      % (settings.RUN_DIR, n_framework, "all" if wanted is None else list(wanted), states))

n_jobs = n_done = n_queued = n_bad = 0
for reaction_name, reaction_dir in layout.species_dirs(run.ts_unique):
    index = int(reaction_name.split("_")[0])
    if wanted is not None and index not in wanted:
        continue
    with open(os.path.join(reaction_dir, layout.REACTION_INFO)) as handle:
        info = json.load(handle)
    ts = TSGraph(molecule_from_adjlist(info["reactant"]), molecule_from_adjlist(info["product"]))
    print("\n%s  %s" % (reaction_name, info["reaction"]))
    if len(ts.sites()) != 2:
        print("  %d site(s) -- only two-site reactions are handled, skipped" % len(ts.sites()))
        continue

    target_dir = os.path.join(run.ts_harmonic, reaction_name)
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy2(os.path.join(reaction_dir, layout.REACTION_INFO),
                 os.path.join(target_dir, layout.REACTION_INFO))

    counts = {"submit": 0, "done": 0, "queued": 0, "bad": 0}
    for relative, xyz in layout.initial_guess_files(reaction_dir):
        pair, stem = relative.split(os.sep)
        guess = info["pairs"][pair]["guesses"][stem]
        label = os.path.join(reaction_name, relative)

        guess_dir = os.path.join(target_dir, relative)
        os.makedirs(guess_dir, exist_ok=True)
        xyz_name = os.path.basename(xyz)
        shutil.copy2(xyz, os.path.join(guess_dir, xyz_name))

        todo = []
        for state in states:
            job_dir = layout.harmonic_job_dir(guess_dir, state)
            tag = "%-7s %s" % (state, relative)
            if os.path.exists(os.path.join(job_dir, layout.harmonic_files(state)[0])):
                print("  already done   %s" % tag)
                counts["done"] += 1
            elif os.path.exists(os.path.join(job_dir, layout.JOB_ID)):
                with open(os.path.join(job_dir, layout.JOB_ID)) as handle:
                    print("  in the queue   %s  (job %s, no result yet)" % (tag, handle.read().strip()))
                counts["queued"] += 1
            else:
                todo.append(state)
        if not todo:
            continue

        atoms = read(xyz)
        try:
            check_atom_order(atoms, ts, guess["oxygens"], n_framework)
        except ValueError as error:
            print("  NOT SUBMITTED  %s: %s" % (relative, error))
            counts["bad"] += len(todo)
            continue
        bonds = guess_bond_indices(ts, guess["oxygens"], n_framework)
        for bond in bonds:
            bond["start"] = round(float(atoms.get_distance(*bond["indices"], mic=True)), 3)
        with open(os.path.join(guess_dir, layout.TS_BONDS_JSON), "w") as handle:
            json.dump({"guess": label, "reaction": info["reaction"], "index": index,
                       "n_framework": n_framework, "oxygens": guess["oxygens"],
                       "flip": guess["flip"], "bonds": bonds}, handle, indent=2)

        for state in todo:
            job_dir = layout.harmonic_job_dir(guess_dir, state)
            os.makedirs(job_dir, exist_ok=True)
            tag = "%-7s %s" % (state, relative)
            xyz_arg = os.path.relpath(os.path.join(guess_dir, xyz_name), job_dir)
            with open(os.path.join(job_dir, layout.JOB_SCRIPT), "w") as handle:
                handle.write(job_script("tsh%d_%s%s" % (index, pair[-2:], state[0]), xyz_arg, state))
            if settings.SUBMIT:
                result = subprocess.run(["sbatch", "--chdir=" + job_dir, layout.JOB_SCRIPT],
                                        capture_output=True, text=True)
                print("  %s -> %s" % (tag, result.stdout.strip() or result.stderr.strip()))
                if result.returncode != 0:
                    sys.exit("sbatch failed, stopping")
                with open(os.path.join(job_dir, layout.JOB_ID), "w") as handle:
                    handle.write(result.stdout.strip().split()[-1] + "\n")
            else:
                print("  would submit   %s" % tag)
            counts["submit"] += 1

    print("  %d jobs %s, %d already done, %d in the queue, %d not submitted"
          % (counts["submit"], "submitted" if settings.SUBMIT else "to submit",
             counts["done"], counts["queued"], counts["bad"]))
    n_jobs += counts["submit"]
    n_done += counts["done"]
    n_queued += counts["queued"]
    n_bad += counts["bad"]

print("\n%d jobs %s, %d already done, %d in the queue, %d not submitted  (%s)"
      % (n_jobs, "submitted" if settings.SUBMIT else "would be submitted",
         n_done, n_queued, n_bad, run.ts_harmonic))