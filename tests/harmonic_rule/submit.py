#!/usr/bin/env python
"""Submit test jobs of the step 3 relax: one job per raw guess in SOURCES
and state in STATES. Run on the login node, from anywhere:

    python tests/harmonic_rule/submit.py

Every value of the relax is a constant below. For each raw guess it copies
<stem>_init.xyz and ts_bonds.json into

    tests/harmonic_rule/<TEST>/<the path of the guess, without TS_harmonic>/

and writes <state>/params.json (the values below) and <state>/job.sh there,
then submits it. Each job runs harmonic_test.py with the model and round of
the first two parts of its path, not the MODEL line of settings.py. SLURM
and D3 values are the ones in scripts/settings.py. settings.SUBMIT = False
writes the files without submitting. A second run with the same TEST
overwrites the results; change TEST to keep both.
"""

import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import _common  # noqa: F401
import layout
import machine
import settings

TEST = "new_rule"               # first folder level of the results
SOURCES = [                     # raw guesses: paths under runs/, <model>/<round>/...
    "mpa0/round0/MOR/T1/TS_harmonic/0_rxn/pair_00/flip1_tor000_roll045",
]
# STATES = ("ts", "initial", "final")     # one state needs the comma: ("initial",)
STATES = ("initial",)     # one state needs the comma: ("initial",)

# -- the relax (start values: scripts/settings.py) ---------------------------
STEPS = {"framework": 100, "spectators": 20, "reacting": 100}   # BFGS step cap of each stage
MAX_CYCLES = 3                  # framework -> spectators -> reacting cycles at most
E_TOL = 0.01                    # eV, the cycles stop when one changes E by less
FMAX = 0.05                     # eV/A, force criterion of each stage
FREE_STEPS = 300                # BFGS step cap of the spring-free relax
FREE_FMAX = 0.05                # eV/A, force criterion of the spring-free relax
FRAMEWORK_RADIUS = 4.0          # A, framework atoms this close to the adsorbate relax
BOND_CUTOFF = 1.25              # bonded if r < covalent radii sum x this (bond check)
# per state and bond type: (length factor, k in eV/A^2). The spring target is
# the covalent radii sum x the factor; k = 0 means no spring.
# "free_relax": spring-free relax after the springs. "bonded": the bond types
# that must be bonded at the end (no "bonded": no bond check, as for ts).
MODES = {
    "ts":      {"form": (1.2, 30.0), "break": (1.2, 30.0), "free_relax": False},
    "initial": {"form": (2.0, 0.0), "break": (1.0, 30.0), "free_relax": True, "bonded": ("break",)},
    "final":   {"form": (1.0, 30.0), "break": (2.0, 0.0), "free_relax": True, "bonded": ("form",)},
}

WORKER = os.path.join(HERE, "harmonic_test.py")
PARAMS_JSON = "params.json"     # <guess folder>/<state>/: read by harmonic_test.py

JOB_TEMPLATE = """#!/bin/bash
%(header)s

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=%(repo)s:$PYTHONPATH
# the model of this guess (settings.py reads these)
export PYNTAZ_JOB_MODEL=%(model)s
export PYNTAZ_JOB_HEAD=%(head)s
export PYNTAZ_JOB_D3=%(d3)s

%(python)s %(worker)s %(guess)s %(state)s
"""

unknown = [state for state in STATES if state not in MODES]
if unknown:
    sys.exit('STATES has %s, not in MODES (one state needs the comma: ("initial",))' % unknown)
if not os.path.exists(machine.PYTHON):
    sys.exit("not found: " + machine.PYTHON)

# check every source before anything is submitted
guesses = []
for source in SOURCES:
    parts = source.strip("/").split("/")
    if len(parts) < 3 or parts[0] not in settings.MODELS or parts[1] not in settings.MODELS[parts[0]]:
        sys.exit("%s: a source starts with <model>/<round> of settings.MODELS, e.g. mpa0/round0/..."
                 % source)
    entry = settings.MODELS[parts[0]][parts[1]]
    model = os.path.join(REPO, "models", entry["file"])
    source_dir = os.path.join(REPO, layout.RUNS, *parts)
    inputs = [parts[-1] + layout.INITIAL_GUESS_SUFFIX, layout.TS_BONDS_JSON]
    for path in [os.path.join(source_dir, name) for name in inputs] + [model]:
        if not os.path.exists(path):
            sys.exit("not found: " + path)
    guess_dir = os.path.join(HERE, TEST, *[part for part in parts if part != "TS_harmonic"])
    guesses.append((source, entry, model, source_dir, inputs, guess_dir))

for source, entry, model, source_dir, inputs, guess_dir in guesses:
    os.makedirs(guess_dir, exist_ok=True)
    for name in inputs:
        shutil.copy2(os.path.join(source_dir, name), os.path.join(guess_dir, name))
    print("%s -> %s" % (source, os.path.relpath(guess_dir, REPO)))

    for state in STATES:
        job_dir = os.path.join(guess_dir, state)
        os.makedirs(job_dir, exist_ok=True)
        with open(os.path.join(job_dir, PARAMS_JSON), "w") as handle:
            json.dump({"test": TEST, "source": source, "state": state, "mode": MODES[state],
                       "steps": STEPS, "max_cycles": MAX_CYCLES, "e_tol": E_TOL, "fmax": FMAX,
                       "free_steps": FREE_STEPS, "free_fmax": FREE_FMAX,
                       "framework_radius": FRAMEWORK_RADIUS, "bond_cutoff": BOND_CUTOFF},
                      handle, indent=2)
        job_script = os.path.join(job_dir, layout.JOB_SCRIPT)
        with open(job_script, "w") as handle:
            handle.write(JOB_TEMPLATE % {
                "header": machine.sbatch_header(settings.TS_HARMONIC_SLURM_CORES, settings.SLURM_MEMORY,
                                                settings.TS_HARMONIC_SLURM_TIME, name="rule_" + state),
                "repo": REPO, "model": model, "head": entry["head"] or "",
                "d3": "1" if entry["d3"] else "0", "python": machine.PYTHON,
                "worker": WORKER, "guess": guess_dir, "state": state})
        if not settings.SUBMIT:
            print("  %-7s job.sh written, not submitted" % state)
            continue
        result = subprocess.run(["sbatch", "--chdir=" + job_dir, job_script],
                                capture_output=True, text=True)
        print("  %-7s -> %s" % (state, result.stdout.strip() or result.stderr.strip()))
        if result.returncode != 0:
            sys.exit("sbatch failed, stopping")
        with open(os.path.join(job_dir, layout.JOB_ID), "w") as handle:
            handle.write(result.stdout.strip().split()[-1] + "\n")