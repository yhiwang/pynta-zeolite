#!/usr/bin/env python
"""Harmonic relax of one raw TS guess with MACE, toward one state. This is
the worker that job.sh runs (step 7 writes the job).

    python ts_harmonic_one.py <stem>_init.xyz <state>

<state> is a key of settings.TS_HARMONIC_MODES: "ts" (springs on every
forming and breaking bond), "initial" (breaking bonds only, then a
spring-free relax) or "final" (forming bonds only, then a spring-free
relax). Reads ts_bonds.json next to the input (written by step 7). Writes,
into layout.harmonic_job_dir(<guess folder>, state):

    ts        ts_harmonic.xyz / .traj / .log       in the guess folder
    initial   initial/initial.xyz / .traj / .log, initial/endpoint.json
    final     final/final.xyz / .traj / .log,     final/endpoint.json

The .xyz carries the MACE energy and forces without springs. Needs only
ase, mace and pyntaz.ts_harmonic_relax, so it runs on a compute node
without maze or RMG; the repo must be on PYTHONPATH (job.sh exports it).
"""

import json
import os
import sys

import _common  # noqa: F401
import layout
import settings
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from pyntaz.harmonic_relax import state_relax


def main(xyz_path, state):
    if state not in settings.TS_HARMONIC_MODES:
        sys.exit("unknown state %r, expected one of %s" % (state, list(settings.TS_HARMONIC_MODES)))
    if not os.path.isfile(settings.MACE_MODEL):
        sys.exit("MACE model file not found: " + settings.MACE_MODEL)
    mode = settings.TS_HARMONIC_MODES[state]
    guess_dir = os.path.dirname(os.path.abspath(xyz_path))
    out_dir = layout.harmonic_job_dir(guess_dir, state)
    os.makedirs(out_dir, exist_ok=True)
    structure, trajectory, logfile = layout.harmonic_files(state)
    with open(os.path.join(guess_dir, layout.TS_BONDS_JSON)) as handle:
        record = json.load(handle)

    atoms = read(xyz_path)
    n_framework = record["n_framework"]
    bonds = [(b["indices"][0], b["indices"][1], b["change"]) for b in record["bonds"]]
    tags = {tuple(b["indices"]): "-".join(b["tags"]) for b in record["bonds"]}
    print("%s   %s   state %s" % (record["guess"], record["reaction"], state))
    print("framework %d atoms, adsorbate %d atoms, oxygens %s"
          % (n_framework, len(atoms) - n_framework, record["oxygens"]))

    result = state_relax(
        atoms, bonds, n_framework, settings.MACE_MODEL,
        springs=mode["springs"], mult=mode["mult"], free_relax=mode["free_relax"], sites=record["oxygens"],
        k=settings.TS_HARMONIC_K, framework_radius=settings.TS_HARMONIC_FRAMEWORK_RADIUS,
        steps=settings.TS_HARMONIC_STEPS, max_cycles=settings.TS_HARMONIC_MAX_CYCLES,
        e_tol=settings.TS_HARMONIC_E_TOL, fmax=settings.TS_HARMONIC_FMAX,
        free_steps=settings.TS_HARMONIC_FREE_STEPS, cutoff=settings.TS_HARMONIC_BOND_CUTOFF,
        logfile=os.path.join(out_dir, logfile), trajectory=os.path.join(out_dir, trajectory),
        log=print)

    harmonic = result["harmonic"]
    targets = {(i, j): r0 for i, j, _, _, r0 in harmonic["lengths"]} if harmonic else {}
    print("bonds after (spring target, - = no spring):")
    for i, j, change, r, bonded, expected in result["bonds"]:
        target = "%.3f" % targets[(i, j)] if (i, j) in targets else "  -  "
        check = "" if expected is None else ("  ok" if bonded == expected else "  WRONG")
        print("  %-5s %-12s %-10s %.3f A   (%s)  %s%s"
              % (change, tags[(i, j)], "%d-%d" % (i, j), r, target,
                 "bonded" if bonded else "apart", check))
    if harmonic:
        print("springs settled %s (%d cycle%s, max %d)"
              % (harmonic["settled"], len(harmonic["cycles"]),
                 "" if len(harmonic["cycles"]) == 1 else "s", settings.TS_HARMONIC_MAX_CYCLES))
    if state == "ts":
        print("MACE force left without the springs: reacting atoms %.3f, relaxed framework %.3f eV/A"
              % (harmonic["fmax_reacting"], harmonic["fmax_framework"]))
    for i, j, what in result["other"]:
        print("  other bond %s: %s%d-%s%d" % (what, atoms[i].symbol, i, atoms[j].symbol, j))
    if result["verdict"] is not None:
        print("verdict   %s" % result["verdict"])
    print("energy    %.4f eV" % result["energy"])

    atoms.calc = SinglePointCalculator(atoms, energy=result["energy"], forces=result["forces"])
    write(os.path.join(out_dir, structure), atoms)
    if mode["free_relax"]:
        with open(os.path.join(out_dir, layout.ENDPOINT_RESULT), "w") as handle:
            json.dump({"guess": record["guess"], "reaction": record["reaction"],
                       "index": record["index"], "state": state,
                       "verdict": result["verdict"], "energy": round(result["energy"], 6),
                       "free_steps": result["free_steps"],
                       "free_converged": result["free_converged"],
                       "bonds": [{"change": change, "tags": tags[(i, j)].split("-"),
                                  "indices": [i, j], "r": round(r, 3),
                                  "bonded": bonded, "expected": expected}
                                 for i, j, change, r, bonded, expected in result["bonds"]],
                       "other": [{"indices": [i, j], "change": what}
                                 for i, j, what in result["other"]]},
                      handle, indent=2)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])