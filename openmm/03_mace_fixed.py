#!/usr/bin/env python
"""MACE relax of every guess in guesses/<i>_rxn/ in alternating stages:

  framework    adsorbate fixed, framework atoms within FRAMEWORK_RADIUS of
               the adsorbate relax (springs on, so the seated oxygens stay
               at their TS distances); the rest of the framework stays fixed
  spectators   everything fixed except the spectator adsorbate atoms
  reacting     everything fixed except the reacting adsorbate atoms, which
               relax under springs pulling each reacting pair to
               (radius_i + radius_j) x MULT

One cycle is framework, spectators, reacting. Cycles repeat until the MACE
energy changes by less than E_TOL over a cycle, or MAX_CYCLES is reached;
a last spectator stage then relaxes the spectators around the final
reacting geometry.

Reads  guesses/<i>_rxn/ts_guess.xyz, ts_guess.json   (from 01_make_ts_guess.py)
       or only the guess directories named on the command line
Copies both into mace_relax/<i>_rxn/ and works there, so guesses/ stays as
01 wrote it; writes ts_guess_mace.xyz, mace_stages.traj next to the copies.
A rerun overwrites mace_relax/<i>_rxn/.
"""

import glob
import json
import os
import shutil
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
OUT = os.path.join(HERE, "mace_relax")   # not "mace", to keep clear of the mace package name

STEPS = {"framework": 100, "spectators": 20, "reacting": 100}
FRAMEWORK_RADIUS = 4.0                 # A, framework atoms this close to the adsorbate relax;
                                       # set FREE_RADIUS in 04 to the same value
MAX_CYCLES = 3
E_TOL = 0.01                           # eV, stop when a cycle changes E by less than this
MULT = {"form": 1.35, "break": 1.35}   # TS length = (radius_i + radius_j) x this
K_RESTRAINT = 30.0                     # eV/A^2
FMAX = 0.05                            # eV/A

if not os.path.isfile(MODEL):
    raise SystemExit("MACE model file not found: " + MODEL)
if not GUESSES:
    raise SystemExit("no guesses/<i>_rxn directories; run 01_make_ts_guess.py first")


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


def relax_guess(source):
    # -- copy the guess into its own working directory ---------------------
    folder = os.path.join(OUT, os.path.basename(os.path.normpath(source)))
    os.makedirs(folder, exist_ok=True)
    for name in ("ts_guess.xyz", "ts_guess.json"):
        shutil.copy2(os.path.join(source, name), folder)

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
    local = [k for k in framework
             if atoms.get_distances(k, adsorbate, mic=True).min() < FRAMEWORK_RADIUS]
    movers = {"framework": local,
              "spectators": spectators,
              "reacting": [i for i in reacting_atoms if i in adsorbate]}

    def name(i):
        return "%s%d" % (atoms[i].symbol, i)

    def show(tag):
        print("  " + tag)
        for (i, j, change), (_, _, r0) in zip(reacting_bonds, targets):
            print("    %-5s  %5s-%-5s  %.3f A   (TS %.3f)"
                  % (change, name(i), name(j), atoms.get_distance(i, j, mic=True), r0))

    print("\n%s  %s" % (os.path.basename(folder), guess["reaction"]))
    print("  %d framework atoms within %.1f A of the adsorbate relax" % (len(local), FRAMEWORK_RADIUS))
    show("before")

    # -- the stages --------------------------------------------------------
    atoms.pbc = True
    springs = BondRestraints(targets, K_RESTRAINT)
    traj = Trajectory(os.path.join(folder, "mace_stages.traj"), "w")

    def stage(which):
        free = set(movers[which])
        atoms.set_constraint(FixAtoms(indices=[k for k in range(len(atoms)) if k not in free]))
        atoms.calc = mace if which == "spectators" else SumCalculator([mace, springs])
        optimizer = BFGS(atoms, logfile=None, trajectory=traj)
        optimizer.run(fmax=FMAX, steps=STEPS[which])
        atoms.calc = mace
        return optimizer.get_number_of_steps()

    atoms.calc = mace
    energy = atoms.get_potential_energy()
    print("  start                        E = %.4f eV" % energy)
    for cycle in range(1, MAX_CYCLES + 1):
        n0 = stage("framework")
        n1 = stage("spectators")
        n2 = stage("reacting")
        previous, energy = energy, atoms.get_potential_energy()
        print("  cycle %d  %3d + %3d + %3d steps   E = %.4f eV   change %+.4f"
              % (cycle, n0, n1, n2, energy, energy - previous))
        if abs(previous - energy) < E_TOL:
            break
    n1 = stage("spectators")
    energy = atoms.get_potential_energy()
    print("  final spectators  %3d steps   E = %.4f eV" % (n1, energy))
    traj.close()

    show("after")
    atoms.set_constraint()                 # FixAtoms would zero the forces we want to report
    forces = atoms.get_forces()
    fmax_reacting = np.linalg.norm(forces[reacting_atoms], axis=1).max()
    fmax_local = np.linalg.norm(forces[local], axis=1).max()
    print("  MACE force left without the springs: reacting atoms %.3f, relaxed framework %.3f eV/A"
          % (fmax_reacting, fmax_local))

    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    write(os.path.join(folder, "ts_guess_mace.xyz"), atoms)


for source in GUESSES:
    relax_guess(source)
print("\nwrote %d directories under %s" % (len(GUESSES), OUT))