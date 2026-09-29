#!/usr/bin/env python
"""Sella saddle search + frequency check of one TS start with MACE. This is
the worker that job.sh runs inside every TS_sella/.../<start>/ directory
(step 8 writes the job).

    python ts_sella_one.py start.xyz

Reads ts_bonds.json next to the input (copied there by step 8). Writes
next to it:

    ts_sella.xyz    where Sella stopped, with MACE energy and forces
    sella.traj      every Sella step, for ase gui
    sella.log       Sella's own log: energy, force, trust radius per step
    vib/, vib.0.traj   frequency cache, and the first mode as an animation
                       (only when the search converged)
    result.json     verdict and numbers; written last, so it also marks
                    the job as finished

Needs only ase, sella, mace and pyntaz.ts_sella, so it runs on a compute
node without maze or RMG; the repo must be on PYTHONPATH (job.sh exports it).
"""

import json
import os
import sys

import _common  # noqa: F401
import layout
import settings
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from pyntaz.ts_sella import (free_atoms, frequency_check, load_mace, saddle_search,
                             verdict)


def lengths(atoms, bonds):
    return [round(float(atoms.get_distance(i, j, mic=True)), 3) for i, j, _ in bonds]


def main(xyz_path):
    if not os.path.isfile(settings.MACE_MODEL):
        sys.exit("MACE model file not found: " + settings.MACE_MODEL)
    out_dir = os.path.dirname(os.path.abspath(xyz_path))
    with open(os.path.join(out_dir, layout.TS_BONDS_JSON)) as handle:
        record = json.load(handle)
    start = os.path.basename(out_dir)          # "harmonic" or "raw"

    atoms = read(xyz_path)
    n_framework = record["n_framework"]
    bonds = [(b["indices"][0], b["indices"][1], b["change"]) for b in record["bonds"]]
    names = ["%s %s" % (b["change"], "-".join(b["tags"])) for b in record["bonds"]]
    free = free_atoms(atoms, n_framework, settings.TS_SELLA_FREE_RADIUS)
    print("%s   start: %s   %s" % (record["guess"], start, record["reaction"]))
    print("free atoms %d (%d framework within %.1f A + %d adsorbate)"
          % (len(free), len(free) - (len(atoms) - n_framework),
             settings.TS_SELLA_FREE_RADIUS, len(atoms) - n_framework))

    calc = load_mace(settings.MACE_MODEL)
    atoms.calc = calc
    start_energy = float(atoms.get_potential_energy())
    start_lengths = lengths(atoms, bonds)

    search = saddle_search(atoms, calc, free, fmax=settings.TS_SELLA_FMAX,
                           max_steps=settings.TS_SELLA_STEPS,
                           internal=settings.TS_SELLA_INTERNAL,
                           logfile=os.path.join(out_dir, layout.TS_SELLA_LOG),
                           trajectory=os.path.join(out_dir, layout.TS_SELLA_TRAJECTORY))
    end_lengths = lengths(atoms, bonds)
    forces = atoms.get_forces()
    print("\n%-22s %8s %8s" % ("bond", "start", "end"))
    for name, before, after in zip(names, start_lengths, end_lengths):
        print("%-22s %8.3f %8.3f" % (name, before, after))
    print("\nenergy  %.4f -> %.4f eV   (%+.4f)"
          % (start_energy, search["energy"], search["energy"] - start_energy))
    print("sella   %d steps, fmax %.3f eV/A, %s"
          % (search["n_steps"], search["fmax"],
             "converged" if search["converged"] else "NOT converged"))

    atoms.calc = SinglePointCalculator(atoms, energy=search["energy"], forces=forces)
    write(os.path.join(out_dir, layout.TS_SELLA_STRUCTURE), atoms)

    check = None
    if search["converged"]:
        atoms.calc = calc
        check = frequency_check(atoms, free, bonds, os.path.join(out_dir, layout.TS_SELLA_VIB),
                                delta=settings.TS_SELLA_VIB_DELTA,
                                imag_cutoff=settings.TS_SELLA_IMAG_CUTOFF)
        print("imaginary modes above %.0f cm-1: %s"
              % (settings.TS_SELLA_IMAG_CUTOFF,
                 "  ".join("%.0fi" % f for f in check["imaginary"]) or "none"))
        if check["stretches"]:
            print("first mode stretches: " + "   ".join(
                "%s %+.2f" % (name, s) for name, s in zip(names, check["stretches"])))
    outcome = verdict(search, check, settings.TS_SELLA_MODE_MIN)
    print("verdict  %s" % outcome)

    with open(os.path.join(out_dir, layout.TS_SELLA_RESULT), "w") as handle:
        json.dump({"guess": record["guess"], "start": start, "verdict": outcome,
                   "energy_start": start_energy, "energy": search["energy"],
                   "converged": search["converged"], "fmax": search["fmax"],
                   "n_steps": search["n_steps"],
                   "imaginary": check["imaginary"] if check else None,
                   "bonds": [{"name": name, "indices": list(bond[:2]), "start": before,
                              "end": after,
                              "stretch": check["stretches"][k]
                              if check and check["stretches"] else None}
                             for k, (name, bond, before, after)
                             in enumerate(zip(names, bonds, start_lengths, end_lengths))],
                   "free_radius": settings.TS_SELLA_FREE_RADIUS,
                   "internal": settings.TS_SELLA_INTERNAL}, handle, indent=2)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])