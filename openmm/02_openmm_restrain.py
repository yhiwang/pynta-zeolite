#!/usr/bin/env python
"""Restrained relax of ts_guess.xyz with OpenMM.

The framework is frozen; the forming and breaking bonds from ts_guess.json
are held at STRETCH times their covalent length; the adsorbate's other bonds
are held where they were built; steric repulsion between adsorbate and
framework does the rest. No force field -- the energies mean nothing, the
geometry is the point.

Writes ts_guess_openmm.xyz and minimize.traj (every L-BFGS iteration,
for ase gui) next to this file.
"""

import json
import os
import sys
import time

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
STRETCH = {"form": 1.4, "break": 1.35}   # restraint target = covalent length x this
K_RESTRAINT = 500000.0      # kJ/mol/nm^2
K_BOND = 200000.0           # kJ/mol/nm^2, the adsorbate's own bonds
CUTOFF = 0.8                # nm, steric repulsion (clamped to half the box)
SIGMA = {"Si": 0.34, "Al": 0.36, "O": 0.30, "C": 0.34, "H": 0.24}   # nm

NM, KJ = u.nanometer, u.kilojoule_per_mole

# -- read the guess, check the split two ways ------------------------------
atoms = read(os.path.join(HERE, "ts_guess.xyz"))
with open(os.path.join(HERE, "ts_guess.json")) as handle:
    guess = json.load(handle)
nslab = guess["nslab"]
frame = list(range(nslab))
ads = adsorbate_indices(atoms, frame)
found = framework_indices(atoms)
print("%s on %s %s, %s, clearance %.3f A"
      % (guess["reaction"], guess["code"], "-".join(guess["t_labels"]),
         guess["pair"], guess["clearance"]))
print("%d atoms: %d framework + %d adsorbate (%s); connectivity finds %d framework"
      % (len(atoms), nslab, len(ads), "".join(atoms[i].symbol for i in ads), len(found)))
if sorted(found) != frame:
    raise SystemExit("framework by connectivity disagrees with nslab")

radii = covalent_radii[atoms.get_atomic_numbers()]
restraints = [(i, j, (radii[i] + radii[j]) * STRETCH[b["change"]], b)
              for b in guess["bonds"] for i, j in [b["indices"]]]
reactive = {(i, j) for i, j, _, _ in restraints} | {(j, i) for i, j, _, _ in restraints}


def report(tag, atoms):
    print("\n%s" % tag)
    for i, j, target, bond in restraints:
        print("  %-6s %-6s %-6s  %4d %4d   %.3f A  (target %.3f)"
              % (bond["change"], bond["tags"][0], bond["tags"][1], i, j,
                 atoms.get_distance(i, j, mic=True), target))
    clashes = find_clashes(atoms, ads, frame, ignore_pairs=reactive)
    print("  clearance %.3f A,  %d clash(es)%s"
          % (min_clearance(atoms.positions[ads], atoms.positions[frame],
                           atoms.cell, atoms.pbc),
             len(clashes), ": " + str(clashes) if clashes else ""))


report("before", atoms)

# -- the OpenMM system -----------------------------------------------------
cell = atoms.cell.array
if not np.allclose(cell, np.diag(np.diag(cell))):
    raise SystemExit("cell is not orthorhombic; OpenMM wants a reduced box")

system = mm.System()
for mass in atoms.get_masses():
    system.addParticle(mass * u.amu)
for i in frame:
    system.setParticleMass(i, 0)                     # frozen
system.setDefaultPeriodicBoxVectors(*(cell * 0.1) * NM)

nl = neighbor_list(atoms, mult=1.2)
bonds = mm.HarmonicBondForce()
held = 0
for i in ads:
    for j in bonded_neighbors(nl, i):
        if j in ads and i < j and (i, j) not in reactive:
            bonds.addBond(i, j, atoms.get_distance(i, j, mic=True) * 0.1 * NM,
                          K_BOND * KJ / NM**2)
            held += 1
system.addForce(bonds)

rep = mm.CustomNonbondedForce("eps*((sig/r)^12-2*(sig/r)^6); sig=0.5*(s1+s2); eps=0.4")
rep.addPerParticleParameter("s")
for symbol in atoms.get_chemical_symbols():
    rep.addParticle([SIGMA.get(symbol, 0.32)])
rep.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
cutoff = min(CUTOFF, 0.49 * np.diag(cell).min() * 0.1)   # OpenMM: < half the box
rep.setCutoffDistance(cutoff * NM)
rep.addInteractionGroup(ads, frame)                 # framework-framework skipped
rep.addInteractionGroup(ads, ads)
for i, j, _, _ in restraints:
    rep.addExclusion(i, j)
system.addForce(rep)

restr = mm.CustomBondForce("0.5*k*(r-r0)^2")
restr.addPerBondParameter("k")
restr.addPerBondParameter("r0")
for i, j, target, _ in restraints:
    restr.addBond(i, j, [K_RESTRAINT, target * 0.1])
system.addForce(restr)
print("\n%d adsorbate bonds held, %d restrained, repulsion over %d x %d + %d x %d pairs "
      "within %.2f nm" % (held, len(restraints), len(ads), nslab, len(ads), len(ads), cutoff))

context = mm.Context(system, mm.VerletIntegrator(0.001 * u.picoseconds),
                     mm.Platform.getPlatformByName("CPU"))
context.setPositions((atoms.get_positions() * 0.1) * NM)


# -- minimize, keeping every iteration -------------------------------------
class Trace(mm.MinimizationReporter):
    def __init__(self):
        super().__init__()
        self.frames = []

    def report(self, iteration, x, grad, args):
        self.frames.append((args["system energy"],
                            np.linalg.norm(grad),
                            np.array(x).reshape(-1, 3) * 10.0))
        return False


trace = Trace()
start = time.time()
mm.LocalEnergyMinimizer.minimize(context, tolerance=10.0, maxIterations=1000,
                                 reporter=trace)
print("\n%d L-BFGS iterations in %.2f s" % (len(trace.frames), time.time() - start))
for k in sorted(set(range(0, len(trace.frames), max(1, len(trace.frames) // 8)))
                | {len(trace.frames) - 1}):
    print("  it %3d   E %10.1f kJ/mol   |grad| %9.1f" % (k, trace.frames[k][0],
                                                          trace.frames[k][1]))

with Trajectory(os.path.join(HERE, "minimize.traj"), "w") as traj:
    for _, _, positions in trace.frames:
        snapshot = atoms.copy()
        snapshot.set_positions(positions)
        traj.write(snapshot)

state = context.getState(getPositions=True)
before = atoms.get_positions()
atoms.set_positions(state.getPositions(asNumpy=True).value_in_unit(u.angstrom))
report("after", atoms)
moved = np.linalg.norm(atoms.get_positions() - before, axis=1)
print("  framework moved %.1e A, adsorbate moved up to %.3f A (atom %d)"
      % (moved[frame].max(), moved[ads].max(), ads[int(np.argmax(moved[ads]))]))

write(os.path.join(HERE, "ts_guess_openmm.xyz"), atoms)
print("\nwrote ts_guess_openmm.xyz and minimize.traj (ase gui minimize.traj)")