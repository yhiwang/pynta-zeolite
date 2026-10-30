#!/usr/bin/env python
"""DFT Sella saddle search on one TS that step 11 collected. This is the
worker that job.sh runs inside every DFT_TS/.../<stem>/dft_sella/ folder
(step 12 writes the job).

    python dft_sella_one.py start.xyz

start.xyz is the MACE TS (step 11's ts.xyz) and ts_info.json sits next to
it (both copied there by step 12). VASP runs through ASE with SP_VASP from
dft/dft_settings.py plus DFT_SELLA_VASP_CHANGES (lwave=True, so every Sella
step restarts from the previous wavefunction), in the vasp/ subfolder.
Only the band's free atoms move (ts_info.json "free"), the same region the
MACE NEB converged in. Writes next to start.xyz:

    sella.traj      every Sella step with its DFT energy and forces
    sella.log       Sella's log: energy, force, trust radius per step
    progress.json   steps so far and the geometry of the last accepted step,
                    rewritten after every step (what a resume starts from)
    ts_dft.xyz      where Sella stopped, DFT energy + forces, no constraint
    result.json     verdict and numbers; written last, so it also marks
                    the job as finished

sella.traj holds every DFT evaluation, including the extra ones Sella makes
to estimate curvature, so it has more frames than steps; sella.log has one
line per step. All of them are DFT-labelled frames near the saddle.

Resuming: a job stopped by the time limit leaves progress.json without
result.json. Run again (step 12 --resume): the old sella.traj becomes
sella.0.traj (then .1, ...) and the search continues from the last accepted
step, with the WAVECAR still in vasp/. DFT_SELLA_STEPS counts the steps of
every part together. Sella's Hessian estimate starts over on each resume.

result.json holds converged / fmax / n_steps; the MACE energy of start.xyz
and the DFT energy of the very first frame (DFT at the MACE TS, before
Sella moved anything) and of the last one; how far the TS moved from the
MACE one (adsorbate RMSD, largest displacement of any free atom); every
reacting bond at the MACE TS and the DFT TS.

"converged" means every free atom is below DFT_SELLA_FMAX at a point Sella
treats as a first-order saddle. It is not yet a checked TS: step 14
counts its imaginary modes, and step 15's quasi-IRC shows it joins the
intended initial and final states.

The search itself is step 8's pyntaz.ts_sella.saddle_search, only with VASP
as the calculator and the band's free atoms; what this file adds is what a
days-long DFT search needs and a minutes-long MACE one does not: resuming,
and no frequency check (finite differences on ~100 free atoms would be ~600
DFT single points; step 14 runs one on the adsorbate and the site only).

Needs ase, sella and numpy (no MACE); job.sh loads the VASP module and sets
ASE_VASP_COMMAND and VASP_PP_PATH.
"""

import glob
import json
import os
import re
import shutil
import sys

import _common  # noqa: F401
import layout
import numpy as np
import settings
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.io import read, write
from pyntaz.neb import displacements
from pyntaz.ts_sella import saddle_search

sys.path.insert(0, os.path.join(_common.REPO, "dft"))
import dft_settings as ds  # noqa: E402

PART = "sella.%d.traj"          # an earlier part of a resumed search
BULKY = ("WAVECAR", "CHG", "CHGCAR", "vasprun.xml", "vaspout.h5", "DOSCAR",
         "EIGENVAL", "PROCAR", "XDATCAR", "PCDAT")   # removed from vasp/ once converged


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def vasp_calculator(directory):
    """ASE's Vasp with the training-set settings plus DFT_SELLA_VASP_CHANGES."""
    from ase.calculators.vasp import Vasp
    params = dict(ds.SP_VASP, **settings.DFT_SELLA_VASP_CHANGES)
    return Vasp(directory=directory, **params), params


def read_frames(path):
    """Every frame of a trajectory, [] when it is missing, empty or unreadable
    (a job killed while writing can leave a broken last frame)."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return []
    try:
        return read(path, ":")
    except Exception as error:                          # noqa: BLE001
        print("  could not read %s (%s); ignoring it" % (path, error))
        return []


def earlier_parts(out_dir):
    """Paths of sella.<k>.traj in order of k."""
    found = glob.glob(os.path.join(out_dir, "sella.*.traj"))
    number = re.compile(r"sella\.(\d+)\.traj$")
    keyed = [(int(number.search(p).group(1)), p) for p in found if number.search(p)]
    return [p for _, p in sorted(keyed)]


def rotate_trajectory(out_dir):
    """Move a leftover sella.traj to the next sella.<k>.traj; return every
    earlier part, oldest first."""
    current = os.path.join(out_dir, layout.TS_SELLA_TRAJECTORY)
    parts = earlier_parts(out_dir)
    if os.path.isfile(current) and os.path.getsize(current) > 0:
        target = os.path.join(out_dir, PART % len(parts))
        shutil.move(current, target)
        parts.append(target)
    elif os.path.exists(current):
        os.remove(current)
    return parts


def load_progress(path):
    """(steps done, positions of the last accepted step or None)."""
    if not os.path.isfile(path):
        return 0, None
    progress = load_json(path)
    return int(progress["steps"]), np.array(progress["positions"])


def save_progress(path, steps, positions):
    """Write-then-rename, so a job killed mid-write leaves the old file."""
    with open(path + ".tmp", "w") as handle:
        json.dump({"steps": int(steps), "positions": positions.tolist()}, handle)
    os.replace(path + ".tmp", path)


def lengths(atoms, bonds):
    return [round(float(atoms.get_distance(i, j, mic=True)), 3) for i, j in bonds]


def main(start_path):
    out_dir = os.path.dirname(os.path.abspath(start_path))
    info = load_json(os.path.join(out_dir, layout.TS_COLLECT_INFO))
    free = info["atoms"]["free"]
    n_framework = info["atoms"]["n_framework"]
    bonds = [tuple(b["indices"]) for b in info["bonds"]]
    names = [b["name"] for b in info["bonds"]]

    mace_ts = read(start_path)
    try:
        e_mace = float(mace_ts.get_potential_energy())
    except Exception:                                    # noqa: BLE001
        e_mace = None

    parts = rotate_trajectory(out_dir)
    progress_path = os.path.join(out_dir, layout.DFT_SELLA_PROGRESS)
    done_steps, positions = load_progress(progress_path)
    atoms = mace_ts.copy()
    if positions is not None:
        atoms.positions = positions
    print("%s   %s" % (info["id"], info["where"]["reaction"]))
    print("free atoms %d; %s" % (len(free), "resuming after %d steps (%d earlier part(s))"
                                 % (done_steps, len(parts)) if positions is not None
                                 else "new search"))

    calc, params = vasp_calculator(os.path.join(out_dir, layout.DFT_SELLA_VASP))
    remaining = settings.DFT_SELLA_STEPS - done_steps
    if remaining > 0:
        search = saddle_search(
            atoms, calc, free, fmax=settings.DFT_SELLA_FMAX, max_steps=remaining,
            internal=settings.DFT_SELLA_INTERNAL,
            logfile=os.path.join(out_dir, layout.TS_SELLA_LOG),
            trajectory=os.path.join(out_dir, layout.TS_SELLA_TRAJECTORY),
            observer=lambda a, n: save_progress(progress_path, done_steps + n, a.positions))
        new_steps = search["n_steps"]
    else:
        print("DFT_SELLA_STEPS (%d) already used up" % settings.DFT_SELLA_STEPS)
        atoms.pbc = True
        atoms.set_constraint(FixAtoms(indices=[a for a in range(len(atoms))
                                               if a not in set(free)]))
        atoms.calc = calc
        new_steps = 0

    energy = float(atoms.get_potential_energy())
    largest = float(np.linalg.norm(atoms.get_forces(), axis=1).max())  # fixed atoms read 0
    converged = largest < settings.DFT_SELLA_FMAX
    forces = atoms.get_forces(apply_constraint=False)

    final = atoms.copy()
    final.set_constraint()
    final.calc = SinglePointCalculator(final, energy=energy, forces=forces)
    write(os.path.join(out_dir, layout.DFT_SELLA_STRUCTURE), final, format="extxyz")

    first_frames = read_frames(parts[0] if parts else
                               os.path.join(out_dir, layout.TS_SELLA_TRAJECTORY))
    e_dft_start = float(first_frames[0].get_potential_energy()) if first_frames else None
    adsorbate = np.arange(n_framework, len(atoms))
    moved = displacements(mace_ts, final)
    mace_lengths, dft_lengths = lengths(mace_ts, bonds), lengths(final, bonds)

    print("\n%-26s %8s %8s" % ("bond", "MACE TS", "DFT TS"))
    for name, before, after in zip(names, mace_lengths, dft_lengths):
        print("%-26s %8.3f %8.3f" % (name, before, after))
    print("\nsella  %d steps (%d this part), fmax %.3f eV/A, %s"
          % (done_steps + new_steps, new_steps, largest,
             "converged" if converged else "NOT converged"))
    print("moved from the MACE TS: adsorbate RMSD %.3f A, largest %.3f A"
          % (float(np.sqrt(np.mean(moved[adsorbate] ** 2))), float(moved.max())))

    if converged and not settings.DFT_SELLA_KEEP_WAVECAR:
        for name in BULKY:
            path = os.path.join(out_dir, layout.DFT_SELLA_VASP, name)
            if os.path.isfile(path):
                os.remove(path)

    with open(os.path.join(out_dir, layout.TS_SELLA_RESULT), "w") as handle:
        json.dump({"id": info["id"],
                   "reaction_index": info["where"]["reaction_index"],
                   "reaction": info["where"]["reaction"],
                   "verdict": "converged" if converged else "not_converged",
                   "converged": converged, "fmax": largest,
                   "fmax_limit": settings.DFT_SELLA_FMAX,
                   "n_steps": done_steps + new_steps, "n_parts": len(parts) + 1,
                   "energy_mace_ts": e_mace,
                   "energy_dft_at_mace_ts": e_dft_start,
                   "energy": energy,
                   "rmsd_adsorbate": float(np.sqrt(np.mean(moved[adsorbate] ** 2))),
                   "max_displacement": float(moved.max()),
                   "bonds": [{"name": name, "indices": list(bond), "mace_ts": before,
                              "dft_ts": after}
                             for name, bond, before, after
                             in zip(names, bonds, mace_lengths, dft_lengths)],
                   "n_free": len(free), "internal": settings.DFT_SELLA_INTERNAL,
                   "vasp": {k: params.get(k) for k in ("xc", "ivdw", "encut", "ediff",
                                                       "ispin", "kpts")}},
                  handle, indent=2)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])