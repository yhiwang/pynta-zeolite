#!/usr/bin/env python
"""Test bench for the step 3 relax: one raw TS guess, one state.

    python harmonic_test.py <guess folder> <state>

<guess folder> holds <stem>_init.xyz and ts_bonds.json (submit.py copies
them there). Every value of the relax comes from
<guess folder>/<state>/params.json, which submit.py writes from the
constants at its top. Writes into <guess folder>/<state>/:

    <state>.xyz / .traj / .log     MACE energy and forces without springs
    endpoint.json                  states with a "bonded" entry in their mode

Differences from pyntaz/harmonic_relax.py, which is not changed:
  - the atom groups come from every forming and breaking bond, in every state
  - each bond type has its own length factor and spring constant k;
    k = 0 means no spring
  - the spring-free relax has its own force criterion
  - the bond check reads the bonds that must be bonded from "bonded"
  - one line per stage: how far every adsorbate atom is from the raw guess
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import _common  # noqa: F401
import layout
import settings
import numpy as np
from ase.calculators.mixing import SumCalculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.geometry import find_mic
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS
from pyntaz.harmonic_relax import BondRestraints, endpoint_check, load_mace, spring_targets

PARAMS_JSON = "params.json"     # <guess folder>/<state>/: written by submit.py


def atom_names(atoms, indices):
    """"C288 H290 ..." -- symbol and index, the index ase gui shows."""
    return " ".join("%s%d" % (atoms[i].symbol, i) for i in indices)


def drift_row(atoms, reference, n_framework, label, n_steps):
    """One line: how far every adsorbate atom is from ``reference`` (A)."""
    delta = (atoms.get_positions() - reference.get_positions())[n_framework:]
    distances = find_mic(delta, atoms.cell, atoms.pbc)[1]
    return "  %-12s %3d steps   " % (label, n_steps) + " ".join("%6.3f" % d for d in distances)


def springs_of(atoms, bonds, mode):
    """({(i, j): (r0, k)} for the bonds with a spring, [BondRestraints]).
    ``mode[change]`` is (length factor, k); r0 = covalent radii sum x factor.
    One calculator per bond type, because each type has its own k."""
    targets, restraints = {}, []
    for change in ("form", "break"):
        factor, k = mode[change]
        pairs = spring_targets(atoms, [bond for bond in bonds if bond[2] == change], {change: factor})
        if k > 0 and pairs:
            restraints.append(BondRestraints(pairs, k))
            targets.update({(i, j): (r0, k) for i, j, r0 in pairs})
    return targets, restraints


def harmonic_relax(atoms, bonds, restraints, n_framework, calc, reference, framework_radius,
                   steps, max_cycles, e_tol, fmax, logfile, trajectory, log):
    """The staged spring relax, in place. ``bonds`` are all forming and
    breaking bonds as (i, j, change) and set the atom groups; ``restraints``
    are the spring calculators; ``reference`` is the raw guess."""
    adsorbate = np.arange(n_framework, len(atoms))
    reacting = sorted({i for i, j, _ in bonds} | {j for i, j, _ in bonds})
    local = [f for f in range(n_framework)
             if atoms.get_distances(f, adsorbate, mic=True).min() < framework_radius]
    movers = {"framework": local,
              "spectators": [i for i in adsorbate if i not in reacting],
              "reacting": [i for i in reacting if i >= n_framework]}
    log("  spectators: %s" % atom_names(atoms, movers["spectators"]))
    log("  reacting:   %s" % atom_names(atoms, movers["reacting"]))

    def stage(which):
        free = set(movers[which])
        atoms.set_constraint(FixAtoms(indices=[a for a in range(len(atoms)) if a not in free]))
        atoms.calc = calc if which == "spectators" else SumCalculator([calc] + restraints)
        optimizer = BFGS(atoms, logfile=logfile, trajectory=trajectory)
        optimizer.run(fmax=fmax, steps=steps[which])
        atoms.calc = calc
        n_steps = optimizer.get_number_of_steps()
        log(drift_row(atoms, reference, n_framework, which, n_steps))
        return n_steps

    atoms.calc = calc
    energy = atoms.get_potential_energy()
    log("  %d framework atoms within %.1f A of the adsorbate relax" % (len(local), framework_radius))
    log("  start                              E = %.4f eV" % energy)
    cycles, settled = 0, False
    for cycle in range(1, max_cycles + 1):
        counts = [stage(which) for which in ("framework", "spectators", "reacting")]
        previous, energy = energy, atoms.get_potential_energy()
        cycles = cycle
        log("  cycle %d  %3d + %3d + %3d steps   E = %.4f eV   change %+.4f"
            % ((cycle,) + tuple(counts) + (energy, energy - previous)))
        if abs(energy - previous) < e_tol:
            settled = True
            break
    final_steps = stage("spectators")
    log("  final spectators  %3d steps         E = %.4f eV" % (final_steps, atoms.get_potential_energy()))
    atoms.set_constraint()
    return cycles, settled


def state_relax(atoms, bonds, n_framework, calc, mode, sites, framework_radius, steps,
                max_cycles, e_tol, fmax, free_steps, free_fmax, cutoff, logfile, trajectory, log):
    """Relax a raw TS guess toward one state, in place: the spring stages
    (when a bond has k > 0), then the spring-free relax (``mode["free_relax"]``)
    and the bond check."""
    atoms.pbc = True
    start = atoms.copy()
    targets, restraints = springs_of(atoms, bonds, mode)
    logfile = open(logfile, "w")
    trajectory = Trajectory(trajectory, "w")
    log("  %-25s" % "moved from raw guess (A)"
        + " ".join("%6s" % name for name in
                   atom_names(atoms, range(n_framework, len(atoms))).split()))

    cycles = settled = None
    if restraints:
        cycles, settled = harmonic_relax(atoms, bonds, restraints, n_framework, calc, start,
                                         framework_radius, steps, max_cycles, e_tol, fmax,
                                         logfile, trajectory, log)
    else:
        log("  every k is 0: no springs, spring cycle skipped")

    free_steps_done = free_converged = None
    if mode["free_relax"]:
        adsorbate = np.arange(n_framework, len(atoms))
        free = set(adsorbate.tolist())
        free |= {f for f in range(n_framework)
                 if atoms.get_distances(f, adsorbate, mic=True).min() < framework_radius}
        atoms.set_constraint(FixAtoms(indices=[a for a in range(len(atoms)) if a not in free]))
        atoms.calc = calc
        optimizer = BFGS(atoms, logfile=logfile, trajectory=trajectory)
        optimizer.run(fmax=free_fmax, steps=free_steps)
        free_steps_done = optimizer.get_number_of_steps()
        free_converged = bool(np.linalg.norm(atoms.get_forces(), axis=1).max() < free_fmax)
        log(drift_row(atoms, start, n_framework, "spring-free", free_steps_done))
        log("  free relax  %3d steps, converged %s   E = %.4f eV"
            % (free_steps_done, free_converged, atoms.get_potential_energy()))

    logfile.close()
    trajectory.close()

    atoms.set_constraint()
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    atoms.calc = None

    bonded = mode.get("bonded")
    report, other, verdict = endpoint_check(start, atoms, bonds, n_framework, bonded or (),
                                            sites=sites, cutoff=cutoff)
    if bonded is None:
        report = [row[:5] + (None,) for row in report]
        verdict = None
    return {"energy": energy, "forces": forces, "targets": targets,
            "cycles": cycles, "settled": settled,
            "free_steps": free_steps_done, "free_converged": free_converged,
            "bonds": report, "other": other, "verdict": verdict}


# --------------------------------------------------------------------------
# the job: one state of one guess
# --------------------------------------------------------------------------

if len(sys.argv) != 3:
    sys.exit(__doc__)
guess_dir, state = os.path.abspath(sys.argv[1]), sys.argv[2]
out_dir = os.path.join(guess_dir, state)
with open(os.path.join(out_dir, PARAMS_JSON)) as handle:
    params = json.load(handle)
mode = params["mode"]
if not os.path.isfile(settings.MACE_MODEL):
    sys.exit("MACE model file not found: " + settings.MACE_MODEL)

xyz_name = [name for name in sorted(os.listdir(guess_dir)) if name.endswith(layout.INITIAL_GUESS_SUFFIX)][0]
with open(os.path.join(guess_dir, layout.TS_BONDS_JSON)) as handle:
    record = json.load(handle)

atoms = read(os.path.join(guess_dir, xyz_name))
n_framework = record["n_framework"]
bonds = [(b["indices"][0], b["indices"][1], b["change"]) for b in record["bonds"]]
tags = {tuple(b["indices"]): "-".join(b["tags"]) for b in record["bonds"]}
print("%s   %s   state %s   test %s" % (record["guess"], record["reaction"], state, params["test"]))
print("framework %d atoms, adsorbate %d atoms, oxygens %s, model %s"
      % (n_framework, len(atoms) - n_framework, record["oxygens"],
         os.path.basename(settings.MACE_MODEL)))
print("steps %s, max cycles %d, e_tol %s eV, fmax %s eV/A, free steps %d, free fmax %s eV/A"
      % (params["steps"], params["max_cycles"], params["e_tol"], params["fmax"],
         params["free_steps"], params["free_fmax"]))
for change in ("form", "break"):
    print("  %-5s bonds: length factor %s, k %s eV/A^2%s"
          % ((change,) + tuple(mode[change]) + ("" if mode[change][1] > 0 else "  (no spring)",)))

calc = load_mace(settings.MACE_MODEL, head=settings.MACE_HEAD, d3=settings.MACE_D3,
                 d3_damping=settings.D3_DAMPING, d3_xc=settings.D3_XC)
result = state_relax(
    atoms, bonds, n_framework, calc, mode, sites=record["oxygens"],
    framework_radius=params["framework_radius"], steps=params["steps"],
    max_cycles=params["max_cycles"], e_tol=params["e_tol"], fmax=params["fmax"],
    free_steps=params["free_steps"], free_fmax=params["free_fmax"],
    cutoff=params["bond_cutoff"],
    logfile=os.path.join(out_dir, state + ".log"),
    trajectory=os.path.join(out_dir, state + ".traj"), log=print)

print("bonds after (spring target and k, - = no spring):")
for i, j, change, r, bonded, expected in result["bonds"]:
    spring = "%.3f, k %s" % result["targets"][(i, j)] if (i, j) in result["targets"] else "-"
    check = "" if expected is None else ("  ok" if bonded == expected else "  WRONG")
    print("  %-5s %-12s %-10s %.3f A   (%s)  %s%s"
          % (change, tags[(i, j)], "%d-%d" % (i, j), r, spring,
             "bonded" if bonded else "apart", check))
if result["cycles"] is not None:
    print("springs settled %s (%d cycle%s, max %d)"
          % (result["settled"], result["cycles"], "" if result["cycles"] == 1 else "s",
             params["max_cycles"]))
for i, j, what in result["other"]:
    print("  other bond %s: %s%d-%s%d" % (what, atoms[i].symbol, i, atoms[j].symbol, j))
if result["verdict"] is not None:
    print("verdict   %s" % result["verdict"])
print("energy    %.4f eV" % result["energy"])

atoms.calc = SinglePointCalculator(atoms, energy=result["energy"], forces=result["forces"])
write(os.path.join(out_dir, state + ".xyz"), atoms)
if "bonded" in mode:
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