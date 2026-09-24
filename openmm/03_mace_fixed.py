#!/usr/bin/env python
"""MACE relax of every guess in guesses/<i>_rxn/ in alternating stages,
framework always fixed:

  spectators   reacting atoms fixed, spectator atoms relax
  reacting     spectator atoms fixed, reacting atoms relax under springs
               pulling each reacting pair to (radius_i + radius_j) x MULT

One cycle is spectators then reacting. Cycles repeat until the MACE energy
drops by less than E_TOL over a cycle, or MAX_CYCLES is reached.

Reads  guesses/<i>_rxn/ts_guess.xyz, ts_guess.json   (from 01_make_ts_guesses.py)
       or only the directories named on the command line
Writes ts_guess_mace.xyz, mace_stages.traj into each
"""

import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.mixing import SumCalculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.data import covalent_radii
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS
from pyntaz.geometry import framework_indices, adsorbate_indices

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(REPO, "models", "mace-mpa-0-medium.model")
GUESSES = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, "guesses", "*_rxn")))

STEPS = {"spectators": 20, "reacting": 100}
MAX_CYCLES = 3
E_TOL = 0.01                           # eV, stop when a cycle gains less than this
MULT = {"form": 1.35, "break": 1.35}   # TS length = (radius_i + radius_j) x this
K_RESTRAINT = 30.0                     # eV/A^2
FMAX = 0.05                            # eV/A

if not os.path.isfile(MODEL):
    raise SystemExit("MACE model file not found: " + MODEL)
if not GUESSES:
    raise SystemExit("no guesses/<i>_rxn directories; run 01_make_ts_guesses.py first")


class BondRestraints(Calculator):
    """Harmonic springs between given atom pairs: E = 1/2 k (r - r0)^2."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, pairs, k, **kwargs):
        super().__init__(**kwargs)
        self.pairs = pairs          # [(i, j, r0), ...] in A
        self.k = k

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        energy = 0.0
        forces = np.zeros((len(self.atoms), 3))
        for i, j, r0 in self.pairs:
            d = self.atoms.get_distance(i, j, mic=True, vector=True)
            r = np.linalg.norm(d)
            energy += 0.5 * self.k * (r - r0) ** 2
            f = self.k * (r - r0) * d / r
            forces[i] += f
            forces[j] -= f
        self.results["energy"] = energy
        self.results["forces"] = forces


from mace.calculators import mace_mp   # heavy import, once for every guess
mace = mace_mp(model=MODEL, default_dtype="float64", device="cpu")


def relax_guess(folder):
    # -- the structure -----------------------------------------------------
    atoms = read(os.path.join(folder, "ts_guess.xyz"))
    with open(os.path.join(folder, "ts_guess.json")) as handle:
        guess = json.load(handle)

    framework = framework_indices(atoms)
    adsorbate = adsorbate_indices(atoms, framework)
    if len(framework) != guess["nslab"]:
        raise SystemExit("%s: framework has %d atoms but script 01 wrote %d"
                         % (folder, len(framework), guess["nslab"]))

    # -- reacting atoms and spectators -------------------------------------
    radius = covalent_radii[atoms.get_atomic_numbers()]
    reacting_bonds = [(b["indices"][0], b["indices"][1], b["change"]) for b in guess["bonds"]]
    reacting_atoms = sorted({i for i, j, _ in reacting_bonds} | {j for i, j, _ in reacting_bonds})
    spectators = [i for i in adsorbate if i not in reacting_atoms]
    targets = [(i, j, (radius[i] + radius[j]) * MULT[change]) for i, j, change in reacting_bonds]

    def name(i):
        return "%s%d" % (atoms[i].symbol, i)

    def show(tag):
        print("  " + tag)
        for (i, j, change), (_, _, r0) in zip(reacting_bonds, targets):
            print("    %-5s  %5s-%-5s  %.3f A   (TS %.3f)"
                  % (change, name(i), name(j), atoms.get_distance(i, j, mic=True), r0))

    print("\n%s  %s" % (os.path.basename(folder), guess["reaction"]))
    show("before")

    # -- the stages --------------------------------------------------------
    atoms.pbc = True
    springs = BondRestraints(targets, K_RESTRAINT)
    traj = Trajectory(os.path.join(folder, "mace_stages.traj"), "w")

    def stage(which):
        if which == "spectators":
            atoms.set_constraint(FixAtoms(indices=sorted(set(framework) | set(reacting_atoms))))
            atoms.calc = mace
        else:
            atoms.set_constraint(FixAtoms(indices=sorted(set(framework) | set(spectators))))
            atoms.calc = SumCalculator([mace, springs])
        optimizer = BFGS(atoms, logfile=None, trajectory=traj)
        optimizer.run(fmax=FMAX, steps=STEPS[which])
        atoms.calc = mace
        return optimizer.get_number_of_steps()

    atoms.calc = mace
    energy = atoms.get_potential_energy()
    print("  start                        E = %.4f eV" % energy)
    for cycle in range(1, MAX_CYCLES + 1):
        n1 = stage("spectators")
        n2 = stage("reacting")
        previous, energy = energy, atoms.get_potential_energy()
        print("  cycle %d  %3d + %3d steps   E = %.4f eV   drop %.4f"
              % (cycle, n1, n2, energy, previous - energy))
        if previous - energy < E_TOL:
            break
    traj.close()

    show("after")
    forces = atoms.get_forces()
    fmax_reacting = np.linalg.norm(forces[[i for i in reacting_atoms if i in adsorbate]], axis=1).max()
    print("  MACE force left on the reacting atoms without the springs: %.3f eV/A" % fmax_reacting)

    atoms.set_constraint()
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    write(os.path.join(folder, "ts_guess_mace.xyz"), atoms)


for folder in GUESSES:
    relax_guess(folder)
print("\nwrote ts_guess_mace.xyz and mace_stages.traj into %d guess directories" % len(GUESSES))