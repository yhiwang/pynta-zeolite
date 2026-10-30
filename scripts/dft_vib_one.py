#!/usr/bin/env python
"""DFT vibrations at one TS that step 13 collected. This is the worker that
job.sh runs inside every DFT_TS/.../<stem>/quasi_irc/vib/ folder (step 14
writes the job).

    python dft_vib_one.py ../start.xyz

start.xyz is the DFT Sella TS (step 12's ts_dft.xyz) and ts_info.json sits
next to it (both copied there by step 13). Finite differences with ASE's
Vibrations (pyntaz.ts_sella.frequency_check): every vibrating atom moves
+-DFT_VIB_DELTA along x, y and z, one VASP single point each, 6 n + 1 in all.
VASP runs through ASE with SP_VASP from dft/dft_settings.py plus
DFT_VIB_VASP_CHANGES, in the vasp/ subfolder.

Vibrating atoms (every other atom stays where it is):

    the adsorbate    index >= n_framework (ts_info.json)
    the site Al      ts_info.json site.al; it must list exactly one Al
    its four O       the four framework O nearest that Al (minimum image)
    their four Si    the framework Si nearest each of those O

Every single point must reach EDIFF. When VASP stops at NELM first, the job
stops with an error before that displacement goes into the cache: raise nelm
in DFT_VIB_VASP_CHANGES and rerun step 14 with --resume.

Resuming: a job stopped by the time limit (or by the check above) keeps every
finished displacement in displacements/; step 14 --resume continues with the
rest (the empty file of the unfinished one is removed first). The cache is
only valid for this start.xyz: to redo a finished vibration (other
settings), delete the whole vib/ folder.

Writes in vib/:

    displacements/        ASE's cache, one file per displacement
    displacements.0.traj  the first mode as an animation, for ase gui
    mode.xyz              start.xyz with the first mode as the per-atom array
                          "mode" (step 15 pushes along it)
    result.json           verdict, frequencies, vibrating atoms; written last,
                          so it also marks the job as finished

verdict
    first_order    exactly one imaginary mode above DFT_VIB_IMAG_CUTOFF
    minimum        none
    higher_order   two or more

Needs ase and numpy (no MACE, no Sella); job.sh loads the VASP module and sets
ASE_VASP_COMMAND and VASP_PP_PATH.
"""

import json
import os
import sys

import _common  # noqa: F401
import layout
import numpy as np
import settings
from ase.io import read, write
from pyntaz.ts_sella import frequency_check

sys.path.insert(0, os.path.join(_common.REPO, "dft"))
import dft_settings as ds  # noqa: E402

BULKY = ("WAVECAR", "CHG", "CHGCAR", "vasprun.xml", "vaspout.h5", "DOSCAR",
         "EIGENVAL", "PROCAR", "XDATCAR", "PCDAT")   # removed from vasp/ once finished


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def vasp_calculator(directory):
    """ASE's Vasp with SP_VASP plus DFT_VIB_VASP_CHANGES. It raises when a
    single point ends without reaching EDIFF, so that displacement is never
    stored."""
    from ase.calculators.vasp import Vasp

    params = dict(ds.SP_VASP, **settings.DFT_VIB_VASP_CHANGES)

    class CheckedVasp(Vasp):
        def calculate(self, *args, **kwargs):
            super().calculate(*args, **kwargs)
            if not self.converged:
                raise RuntimeError(
                    "SCF stopped before EDIFF (nelm %s), see %s/OUTCAR: raise nelm in "
                    "DFT_VIB_VASP_CHANGES, then rerun step 14 with --resume"
                    % (params.get("nelm"), self.directory))

    return CheckedVasp(directory=directory, **params), params


def nearest(atoms, i, candidates, n):
    """The n atoms of ``candidates`` nearest atom i (minimum image), nearest
    first, and their distances."""
    distances = atoms.get_distances(i, candidates, mic=True)
    order = np.argsort(distances)[:n]
    return [int(candidates[k]) for k in order], [float(distances[k]) for k in order]


def vibrating_atoms(atoms, info):
    """{"adsorbate", "al", "oxygens", "silicons"}; prints the site atoms."""
    n_framework = info["atoms"]["n_framework"]
    al = info["site"]["al"]
    if len(al) != 1:
        sys.exit("ts_info.json site.al lists %d Al (%s); the vibrating set needs "
                 "exactly one" % (len(al), al))
    al = al[0]
    oxygens, d_o = nearest(atoms, al, [i for i in range(n_framework)
                                       if atoms[i].symbol == "O"], 4)
    framework_si = [i for i in range(n_framework) if atoms[i].symbol == "Si"]
    silicons, d_si = [], []
    for o in oxygens:
        si, d = nearest(atoms, o, framework_si, 1)
        silicons += si
        d_si += d
    print("Al %d  ->  %s" % (al, "   ".join("O %d %.3f" % od for od in zip(oxygens, d_o))))
    print("O   ->  %s" % "   ".join("O %d: Si %d %.3f" % osd
                                    for osd in zip(oxygens, silicons, d_si)))
    if len(set(silicons)) != 4:
        sys.exit("two O of Al %d share their nearest Si (%s): check the site"
                 % (al, silicons))
    return {"adsorbate": list(range(n_framework, len(atoms))), "al": al,
            "oxygens": oxygens, "silicons": silicons}


def main(start_path):
    folder = os.path.dirname(os.path.abspath(start_path))          # quasi_irc/
    out_dir = os.path.join(folder, layout.DFT_VIB_DIR)
    info = load_json(os.path.join(folder, layout.TS_COLLECT_INFO))
    bonds = [(b["indices"][0], b["indices"][1], b["change"]) for b in info["bonds"]]
    names = [b["name"] for b in info["bonds"]]

    atoms = read(start_path)
    print("%s   %s" % (info["id"], info["where"]["reaction"]))
    sets = vibrating_atoms(atoms, info)
    indices = sets["adsorbate"] + [sets["al"]] + sets["oxygens"] + sets["silicons"]
    cache = os.path.join(out_dir, layout.DFT_VIB_CACHE)
    print("vibrating: %d adsorbate atoms + Al + 4 O + 4 Si = %d atoms, %d single points; %s"
          % (len(sets["adsorbate"]), len(indices), 6 * len(indices) + 1,
             "resuming" if os.path.isdir(cache) else "new"))

    calc, params = vasp_calculator(os.path.join(out_dir, layout.DFT_VIB_VASP))
    atoms.calc = calc
    check = frequency_check(atoms, indices, bonds, cache, delta=settings.DFT_VIB_DELTA,
                            imag_cutoff=settings.DFT_VIB_IMAG_CUTOFF, resume=True)
    n_imaginary = len(check["imaginary"])
    verdict = ("minimum" if n_imaginary == 0 else
               "first_order" if n_imaginary == 1 else "higher_order")

    print("\nimaginary modes above %.0f cm-1: %s"
          % (settings.DFT_VIB_IMAG_CUTOFF,
             "  ".join("%.0fi" % f for f in check["imaginary"]) or "none"))
    if check["stretches"]:
        print("first mode stretches:  " + "   ".join(
            "%s %+.2f" % (name, s) for name, s in zip(names, check["stretches"])))
    print("verdict: %s" % verdict)

    ts = read(start_path)
    ts.new_array("mode", np.asarray(check["mode"], dtype=float))
    ts.info["mode_frequency"] = check["frequencies"][0]   # cm-1, negative: imaginary
    write(os.path.join(out_dir, layout.DFT_VIB_MODE), ts, format="extxyz")

    if not settings.DFT_VIB_KEEP_WAVECAR:
        for name in BULKY:
            path = os.path.join(out_dir, layout.DFT_VIB_VASP, name)
            if os.path.isfile(path):
                os.remove(path)

    with open(os.path.join(out_dir, layout.TS_SELLA_RESULT), "w") as handle:
        json.dump({"id": info["id"],
                   "reaction_index": info["where"]["reaction_index"],
                   "reaction": info["where"]["reaction"],
                   "verdict": verdict, "n_imaginary": n_imaginary,
                   "imaginary": check["imaginary"],
                   "imag_cutoff": settings.DFT_VIB_IMAG_CUTOFF,
                   "frequencies": check["frequencies"],
                   "mode_bond_stretches": [{"name": name, "indices": [i, j], "stretch": s}
                                           for name, (i, j, _), s
                                           in zip(names, bonds, check["stretches"])]
                   if check["stretches"] else None,
                   "vibrating": sets, "n_vibrating": len(indices),
                   "delta": settings.DFT_VIB_DELTA,
                   "vasp": {k: params.get(k) for k in ("xc", "ivdw", "encut", "prec", "ediff",
                                                       "lreal", "ispin", "kpts")}},
                  handle, indent=2)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])