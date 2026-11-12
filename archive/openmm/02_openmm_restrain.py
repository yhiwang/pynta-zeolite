#!/usr/bin/env python
"""Restrained relax of ts_guess.xyz with OpenMM.

Three kinds of hold, nothing else:
  framework atoms   pinned in space (mass 0)
  adsorbate bonds   springs at the length they were built with
  reactive bonds    springs pulling to STRETCH x the covalent length
plus a clash wall between adsorbate and framework. No force field: the
energies mean nothing, the geometry is the point.

Reads ts_guess.xyz + ts_guess.json (from 01_make_ts_guess.py); writes
ts_guess_openmm.xyz and minimize.traj (every L-BFGS step, for ase gui).
"""

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
import openmm as mm
import openmm.unit as u
from ase.data import covalent_radii
from ase.io import read, write
from ase.io.trajectory import Trajectory
from pyntaz.geometry import (framework_indices, adsorbate_indices, neighbor_list,
                             bonded_neighbors, find_clashes, min_clearance)

HERE = os.path.dirname(os.path.abspath(__file__))
STRETCH = {"form": 1.4, "break": 1.35}   # reactive bond target = covalent length x this
K_RESTRAINT = 500000.0      # kJ/mol/nm^2  pull on the reactive bonds
K_BOND = 200000.0           # kJ/mol/nm^2  hold on the other adsorbate bonds
WALL = 1.2                  # x covalent length: adsorbate-framework contacts closer
K_WALL = 100000.0           # kJ/mol/nm^2  ... than this are pushed apart this hard
CUTOFF = 0.6                # nm, longer contacts are never a clash

A = 0.1 * u.nanometer                                   # one angstrom
K = u.kilojoule_per_mole / u.nanometer**2                # spring constant unit


# -- what to hold ----------------------------------------------------------

def reactive_bonds(atoms, guess):
    """(i, j, target A) for every forming/breaking bond in the json."""
    radii = covalent_radii[atoms.numbers]
    return [(i, j, (radii[i] + radii[j]) * STRETCH[bond["change"]])
            for bond in guess["bonds"] for i, j in [bond["indices"]]]


def held_bonds(atoms, ads, skip):
    """(i, j) for every adsorbate bond not in ``skip``."""
    nl = neighbor_list(atoms, mult=1.2)
    return [(i, j) for i in ads for j in bonded_neighbors(nl, i)
            if j in ads and i < j and (i, j) not in skip]


# -- the OpenMM system -----------------------------------------------------

def build_system(atoms, frame, ads, held, reactive):
    system = mm.System()
    for mass in atoms.get_masses():
        system.addParticle(mass * u.amu)
    for i in frame:
        system.setParticleMass(i, 0)
    system.setDefaultPeriodicBoxVectors(*atoms.cell.array * A)

    springs = mm.HarmonicBondForce()
    for i, j in held:
        springs.addBond(i, j, atoms.get_distance(i, j, mic=True) * A, K_BOND * K)
    system.addForce(springs)

    pulls = mm.CustomBondForce("0.5*k*(r-r0)^2")
    pulls.addPerBondParameter("k")
    pulls.addPerBondParameter("r0")
    for i, j, target in reactive:
        pulls.addBond(i, j, [K_RESTRAINT, target * 0.1])
    system.addForce(pulls)

    wall = mm.CustomNonbondedForce("k*step(r0-r)*(r0-r)^2; r0=wall*(rad1+rad2)")
    wall.addGlobalParameter("k", K_WALL)
    wall.addGlobalParameter("wall", WALL)
    wall.addPerParticleParameter("rad")
    for radius in covalent_radii[atoms.numbers]:
        wall.addParticle([radius * 0.1])
    wall.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    wall.setCutoffDistance(min(CUTOFF, 0.049 * atoms.cell.lengths().min()) * u.nanometer)
    wall.addInteractionGroup(ads, frame)          # framework-framework never computed
    wall.addInteractionGroup(ads, ads)
    wall.createExclusionsFromBonds(held + [(i, j) for i, j, _ in reactive], 2)
    system.addForce(wall)
    return system


class Trace(mm.MinimizationReporter):
    """Keeps energy, gradient norm and positions (A) from every iteration."""

    def __init__(self):
        super().__init__()
        self.energies, self.gradients, self.positions = [], [], []

    def report(self, iteration, x, grad, args):
        self.energies.append(args["system energy"])
        self.gradients.append(float(np.linalg.norm(grad)))
        self.positions.append(np.reshape(x, (-1, 3)) * 10.0)
        return False


def minimize(system, atoms):
    """Relaxed positions (A) and the Trace of how it got there."""
    context = mm.Context(system, mm.VerletIntegrator(0.001 * u.picoseconds),
                         mm.Platform.getPlatformByName("CPU"))
    context.setPositions(atoms.positions * A)
    trace = Trace()
    mm.LocalEnergyMinimizer.minimize(context, tolerance=10.0, maxIterations=1000,
                                     reporter=trace)
    state = context.getState(getPositions=True)
    return state.getPositions(asNumpy=True).value_in_unit(u.angstrom), trace


# -- reporting -------------------------------------------------------------

def report(atoms, guess, frame, ads, held, reactive):
    for (i, j, target), bond in zip(reactive, guess["bonds"]):
        print("  %-5s %-5s %-5s %4d %4d  %.3f A  (target %.3f)"
              % (bond["change"], bond["tags"][0], bond["tags"][1], i, j,
                 atoms.get_distance(i, j, mic=True), target))
    print("  held: " + "  ".join("%d-%d %.2f" % (i, j, atoms.get_distance(i, j, mic=True))
                                 for i, j in held))
    skip = {(i, j) for i, j, _ in reactive} | {(j, i) for i, j, _ in reactive}
    clashes = find_clashes(atoms, ads, frame, ignore_pairs=skip)
    binders = {i for i, _, _ in reactive} | {j for _, j, _ in reactive}
    tail = [i for i in ads if i not in binders]
    clearance = min_clearance(atoms.positions[tail], atoms.positions[frame],
                              atoms.cell, atoms.pbc) if tail else float("nan")
    print("  tail clearance %.3f A, %d clash(es) %s" % (clearance, len(clashes), clashes or ""))


# -- run -------------------------------------------------------------------

atoms = read(os.path.join(HERE, "ts_guess.xyz"))
with open(os.path.join(HERE, "ts_guess.json")) as handle:
    guess = json.load(handle)

frame = list(range(guess["nslab"]))
ads = adsorbate_indices(atoms, frame)
if sorted(framework_indices(atoms)) != frame:
    raise SystemExit("framework by connectivity disagrees with nslab")
if not np.allclose(atoms.cell.array, np.diag(atoms.cell.lengths())):
    raise SystemExit("cell is not orthorhombic; OpenMM wants a reduced box")

reactive = reactive_bonds(atoms, guess)
skip = {(i, j) for i, j, _ in reactive} | {(j, i) for i, j, _ in reactive}
held = held_bonds(atoms, ads, skip)

print("%s on %s %s, %s" % (guess["reaction"], guess["code"],
                          "-".join(guess["t_labels"]), guess["pair"]))
print("%d framework atoms pinned, %d adsorbate atoms (%s): %d bonds held, %d pulled"
      % (len(frame), len(ads), "".join(atoms[i].symbol for i in ads), len(held), len(reactive)))

print("\nbefore")
report(atoms, guess, frame, ads, held, reactive)

system = build_system(atoms, frame, ads, held, reactive)
relaxed, trace = minimize(system, atoms)

print("\n%d iterations, E %.1f -> %.1f kJ/mol, |grad| %.0f -> %.0f"
      % (len(trace.energies), trace.energies[0], trace.energies[-1],
         trace.gradients[0], trace.gradients[-1]))

with Trajectory(os.path.join(HERE, "minimize.traj"), "w") as traj:
    for positions in trace.positions:
        snapshot = atoms.copy()
        snapshot.positions = positions
        traj.write(snapshot)

moved = np.linalg.norm(relaxed - atoms.positions, axis=1)
atoms.positions = relaxed
print("\nafter")
report(atoms, guess, frame, ads, held, reactive)
print("  framework moved %.0e A, adsorbate up to %.3f A (atom %d)"
      % (moved[frame].max(), moved[ads].max(), ads[int(np.argmax(moved[ads]))]))

write(os.path.join(HERE, "ts_guess_openmm.xyz"), atoms)
print("\nwrote ts_guess_openmm.xyz, minimize.traj")