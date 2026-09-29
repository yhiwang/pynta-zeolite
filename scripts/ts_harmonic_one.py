#!/usr/bin/env python
"""Harmonic pre-relax of one TS guess with MACE. This is the worker that
job.sh runs inside every TS_harmonic directory (step 7 writes the job).

    python ts_harmonic_one.py <stem>_init.xyz

Reads ts_bonds.json next to the input (written by step 7). Writes
ts_harmonic.xyz (MACE energy and forces, no springs), ts_harmonic.traj
(every stage in order) and ts_harmonic.log (BFGS steps) next to it. Needs
only ase, mace and pyntaz.ts_harmonic_relax, so it runs on a compute node
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
from pyntaz.ts_harmonic_relax import harmonic_relax


def main(xyz_path):
    if not os.path.isfile(settings.MACE_MODEL):
        sys.exit("MACE model file not found: " + settings.MACE_MODEL)
    out_dir = os.path.dirname(os.path.abspath(xyz_path))
    with open(os.path.join(out_dir, layout.TS_BONDS_JSON)) as handle:
        record = json.load(handle)

    atoms = read(xyz_path)
    n_framework = record["n_framework"]
    bonds = [(b["indices"][0], b["indices"][1], b["change"]) for b in record["bonds"]]
    print("%s   %s" % (record["guess"], record["reaction"]))
    print("framework %d atoms, adsorbate %d atoms, oxygens %s"
          % (n_framework, len(atoms) - n_framework, record["oxygens"]))

    result = harmonic_relax(
        atoms, bonds, n_framework, settings.MACE_MODEL,
        mult=settings.TS_HARMONIC_MULT, k=settings.TS_HARMONIC_K,
        framework_radius=settings.TS_HARMONIC_FRAMEWORK_RADIUS,
        steps=settings.TS_HARMONIC_STEPS, max_cycles=settings.TS_HARMONIC_MAX_CYCLES,
        e_tol=settings.TS_HARMONIC_E_TOL, fmax=settings.TS_HARMONIC_FMAX,
        logfile=os.path.join(out_dir, layout.TS_HARMONIC_LOG),
        trajectory=os.path.join(out_dir, layout.TS_HARMONIC_TRAJECTORY),
        log=print)

    print("bonds after (target):")
    tags = {tuple(b["indices"]): b["tags"] for b in record["bonds"]}
    for i, j, change, r, r0 in result["lengths"]:
        print("  %-5s %-12s %-10s %.3f A   (%.3f)"
              % (change, "-".join(tags[(i, j)]), "%d-%d" % (i, j), r, r0))
    print("settled   %s (%d cycle%s, max %d)"
          % (result["settled"], len(result["cycles"]), "" if len(result["cycles"]) == 1 else "s",
             settings.TS_HARMONIC_MAX_CYCLES))
    print("MACE force left without the springs: reacting atoms %.3f, relaxed framework %.3f eV/A"
          % (result["fmax_reacting"], result["fmax_framework"]))
    print("energy    %.4f eV" % result["energy"])

    atoms.calc = SinglePointCalculator(atoms, energy=result["energy"], forces=result["forces"])
    write(os.path.join(out_dir, layout.TS_HARMONIC_STRUCTURE), atoms)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])