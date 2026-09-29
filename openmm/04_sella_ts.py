#!/usr/bin/env python
"""Sella saddle search (order 1) with MACE for every directory in
mace_relax/<i>_rxn/, from each start in STARTS so the raw and pre-relaxed
guesses can be compared. Framework fixed except atoms within FREE_RADIUS of
the adsorbate. A converged TS gets a finite-difference frequency check on
the free atoms: how many imaginary modes, and how much the first one
stretches each reacting bond.

Reads  mace_relax/<i>_rxn/ts_guess.json + the STARTS files (from 03)
       or only the directories named on the command line
Copies them into sella/<i>_rxn/ and works there, so mace_relax/ stays as
03 wrote it; writes ts_sella_<tag>.xyz, sella_<tag>.traj, sella_<tag>.log
and vib_<tag>/ next to the copies. A rerun overwrites sella/<i>_rxn/.
"""

import glob
import json
import os
import shutil
import sys

import numpy as np
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.vibrations import Vibrations
from sella import Sella

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MODEL = os.path.join(REPO, "models", "mace-mpa-0-medium.model")
GUESSES = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, "mace_relax", "*_rxn")))
OUT = os.path.join(HERE, "sella")

STARTS = {"mace": "ts_guess_mace.xyz",   # tag: start file, skipped if missing
          "raw": "ts_guess.xyz"}
FMAX = 0.05             # eV/A
STEPS = 300
INTERNAL = False        # Sella internal coordinates; try True if Cartesian stalls
FREE_RADIUS = 4.0       # A, framework atoms this close to the adsorbate also move;
                        # keep equal to FRAMEWORK_RADIUS in 03
VIB_DELTA = 0.01        # A, finite-difference step for the frequency check
IMAG_CUTOFF = 50.0      # cm-1, smaller imaginary modes count as numerical noise

if not os.path.isfile(MODEL):
    raise SystemExit("MACE model file not found: " + MODEL)
if not GUESSES:
    raise SystemExit("no mace_relax/<i>_rxn directories; run 03_mace_fixed.py first")

from mace.calculators import mace_mp   # heavy import, once for every guess
mace = mace_mp(model=MODEL, default_dtype="float64", device="cpu")


def bond_lengths(atoms, bonds):
    return "  ".join("%s %s%d-%s%d %.3f" % (change, atoms[i].symbol, i, atoms[j].symbol, j,
                                           atoms.get_distance(i, j, mic=True))
                     for i, j, change in bonds)


def optimize(folder, tag, start, guess):
    atoms = read(os.path.join(folder, start))
    atoms.pbc = True
    nslab = guess["nslab"]
    bonds = [(b["indices"][0], b["indices"][1], b["change"]) for b in guess["bonds"]]

    adsorbate = np.arange(nslab, len(atoms))
    free = list(adsorbate)
    if FREE_RADIUS > 0:
        for k in range(nslab):
            if atoms.get_distances(k, adsorbate, mic=True).min() < FREE_RADIUS:
                free.append(k)
    free = sorted(free)
    atoms.set_constraint(FixAtoms(indices=[k for k in range(len(atoms)) if k not in free]))
    atoms.calc = mace

    print("  %-4s start  E %.4f eV  %s" % (tag, atoms.get_potential_energy(), bond_lengths(atoms, bonds)))
    opt = Sella(atoms, order=1, internal=INTERNAL,
                trajectory=os.path.join(folder, "sella_%s.traj" % tag),
                logfile=os.path.join(folder, "sella_%s.log" % tag))
    opt.run(fmax=FMAX, steps=STEPS)
    fmax = np.linalg.norm(atoms.get_forces(), axis=1).max()   # fixed atoms read zero
    converged = fmax < FMAX
    print("  %-4s end    E %.4f eV  %s" % (tag, atoms.get_potential_energy(), bond_lengths(atoms, bonds)))
    print("  %-4s %d steps, fmax %.3f eV/A, %s"
          % (tag, opt.get_number_of_steps(), fmax, "converged" if converged else "NOT converged"))
    write(os.path.join(folder, "ts_sella_%s.xyz" % tag), atoms)
    if not converged:
        return

    # -- frequency check on the free atoms ---------------------------------
    vib_dir = os.path.join(folder, "vib_%s" % tag)
    shutil.rmtree(vib_dir, ignore_errors=True)     # stale displacements give wrong modes
    vib = Vibrations(atoms, indices=free, name=vib_dir, delta=VIB_DELTA)
    vib.run()
    freqs = vib.get_frequencies()
    imaginary = [f.imag for f in freqs if f.imag > IMAG_CUTOFF]
    print("  %-4s %d imaginary mode(s) above %.0f cm-1: %s"
          % (tag, len(imaginary), IMAG_CUTOFF, "  ".join("%.0fi" % f for f in imaginary)))
    if imaginary:
        mode = vib.get_mode(0)
        mode /= np.linalg.norm(mode)
        parts = []
        for i, j, change in bonds:
            d = atoms.get_distance(i, j, mic=True, vector=True)
            parts.append("%s %s%d-%s%d %+.2f" % (change, atoms[i].symbol, i, atoms[j].symbol, j,
                                                  np.dot(mode[j] - mode[i], d / np.linalg.norm(d))))
        print("  %-4s first mode stretches: %s" % (tag, "  ".join(parts)))
    vib.write_mode(0)                              # vib_<tag>.0.traj, animate in ase gui


for source in GUESSES:
    folder = os.path.join(OUT, os.path.basename(os.path.normpath(source)))
    os.makedirs(folder, exist_ok=True)
    for name in ["ts_guess.json"] + list(STARTS.values()):
        if os.path.isfile(os.path.join(source, name)):
            shutil.copy2(os.path.join(source, name), folder)

    with open(os.path.join(folder, "ts_guess.json")) as handle:
        guess = json.load(handle)
    print("\n%s  %s" % (os.path.basename(folder), guess["reaction"]))
    for tag, start in STARTS.items():
        if os.path.isfile(os.path.join(folder, start)):
            optimize(folder, tag, start, guess)
        else:
            print("  %-4s %s missing, skipped" % (tag, start))