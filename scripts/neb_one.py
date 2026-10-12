#!/usr/bin/env python
"""Climbing-image NEB with MACE on one band. This is the worker that job.sh
runs inside every NEB/<i>_rxn/pair_<k>/<stem>/ folder (step 10 writes the
job).

    python neb_one.py band.traj

Reads band.json and neb.json next to the band (step 9 wrote both). Writes
next to it:

    neb.traj          every image of every NEB step, both stages, for ase gui
                      (ase gui neb.traj@-N: shows the last band of N images)
    neb.log           endpoint relax (BFGS), then FIRE: rough band, climbing image
    neb_band.traj     the band where the NEB stopped, MACE energy + forces
    neb.png           energy along that band
    neb_result.json   verdict and numbers; written last, so it also marks
                      the job as finished

The end points are relaxed once more first (NEB_RELAX_ENDPOINTS), with the
band's free atoms and this run's model, then held fixed: step 7 relaxed
each with its own smaller free region, and an end point that is not a
minimum in the band's free space makes its neighbour image dip below it.
The image that starts at the Sella TS is also evaluated before the NEB
moves it, so the top of the band can be compared with the Sella TS on the
same model.

Needs only ase, mace and pyntaz.neb, so it runs on a compute node without
maze or RMG; the repo must be on PYTHONPATH (job.sh exports it).
"""

import json
import os
import sys

import _common  # noqa: F401
import layout
import settings
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.io import read, write
from pyntaz.neb import (ImageCalculator, analyze_band, plot_band, relax_endpoint,
                        run_neb, verdict)
from pyntaz.ts_sella import load_mace


def frozen(atoms, free):
    free = set(free)
    return FixAtoms(indices=[i for i in range(len(atoms)) if i not in free])


def fixed_calculator(atoms, model):
    """MACE energy and forces of ``atoms`` now, kept as a single point."""
    energy = float(model.get_property("energy", atoms))
    forces = model.get_property("forces", atoms).copy()
    return SinglePointCalculator(atoms, energy=energy, forces=forces)


def lengths(atoms, bonds):
    return [round(float(atoms.get_distance(i, j, mic=True)), 3) for i, j, _ in bonds]


def main(band_path):
    if not os.path.isfile(settings.MACE_MODEL):
        sys.exit("MACE model file not found: " + settings.MACE_MODEL)
    out_dir = os.path.dirname(os.path.abspath(band_path))
    with open(os.path.join(out_dir, layout.NEB_BAND_INFO)) as handle:
        band = json.load(handle)
    with open(os.path.join(out_dir, layout.NEB_RECORD)) as handle:
        record = json.load(handle)
    n_framework = record["n_framework"]
    bonds = [(b["indices"][0], b["indices"][1], b["change"]) for b in record["bonds"]]
    names = ["%s %s" % (b["change"], "-".join(b["tags"])) for b in record["bonds"]]
    ts_index = band["ts_index"]

    images = read(band_path, ":")
    if len(images) != band["n_images"]:
        sys.exit("%s has %d images, band.json says %d -- rerun step 9"
                 % (band_path, len(images), band["n_images"]))
    for atoms in images:
        atoms.pbc = True
        atoms.set_constraint(frozen(atoms, band["free"]))
    print("%s   %s" % (record["guess"], record["reaction"]))
    print("%d images (%d before the TS, %d after), %d free atoms (%d framework), %s"
          % (len(images), band["n_before"], band["n_after"], band["n_free"],
             band["n_free_framework"], os.path.basename(settings.MACE_MODEL)))

    model = load_mace(settings.MACE_MODEL)
    endpoints = {}
    for k, name in ((0, "initial"), (len(images) - 1, "final")):
        if settings.NEB_RELAX_ENDPOINTS:
            endpoints[name] = relax_endpoint(images[k], model, fmax=settings.NEB_FMAX,
                                             steps=settings.NEB_STEPS_ROUGH,
                                             logfile=os.path.join(out_dir, layout.NEB_LOG))
            e = endpoints[name]
            print("%-7s relaxed in the band's free region: %+.3f eV, %d steps, max move %.3f A%s"
                  % (name, e["energy"] - e["energy_before"], e["n_steps"],
                     e["max_displacement"], "" if e["converged"] else "  NOT converged"))
        images[k].calc = fixed_calculator(images[k], model)
    for atoms in images[1:-1]:
        atoms.calc = ImageCalculator(model)
    sella_ts = images[ts_index].copy()
    ts_energy = float(images[ts_index].get_potential_energy())
    start_energies = [float(a.get_potential_energy()) for a in images]
    print("start  E - E_IS: " + " ".join("%.3f" % (e - start_energies[0]) for e in start_energies))

    search = run_neb(images, spring=settings.NEB_SPRING, method=settings.NEB_METHOD,
                     fmax_rough=settings.NEB_FMAX_ROUGH, steps_rough=settings.NEB_STEPS_ROUGH,
                     climb=settings.NEB_CLIMB, fmax=settings.NEB_FMAX, steps=settings.NEB_STEPS,
                     trajectory=os.path.join(out_dir, layout.NEB_TRAJECTORY),
                     logfile=os.path.join(out_dir, layout.NEB_LOG))
    energies = search["energies"]
    print("end    E - E_IS: " + " ".join("%.3f" % (e - energies[0]) for e in energies))
    for stage in ("rough", "climb"):
        if stage in search:
            s = search[stage]
            print("%-6s %d steps, fmax %.3f eV/A, %s" % (stage, s["n_steps"], s["fmax"],
                  "converged" if s["converged"] else "NOT converged"))

    final_band = []
    for atoms in images:
        copy = atoms.copy()
        copy.calc = SinglePointCalculator(copy, energy=float(atoms.get_potential_energy()),
                                          forces=atoms.get_forces(apply_constraint=False))
        final_band.append(copy)
    write(os.path.join(out_dir, layout.NEB_FINAL_BAND), final_band)

    analysis = analyze_band(images, energies, sella_ts, n_framework, ts_energy=ts_energy,
                            min_depth=settings.NEB_MIN_DEPTH)
    outcome = verdict(search, analysis, settings.NEB_TS_RMSD)
    top = analysis["top_image"]
    print("\nhighest image %d: Ea_fwd %.3f, Ea_rev %.3f, dE %.3f eV"
          % (top, analysis["Ea_fwd"], analysis["Ea_rev"], analysis["dE_rxn"]))
    print("vs Sella TS: adsorbate RMSD %.3f A (max %.3f), E %+.3f eV"
          % (analysis["top_vs_ts_rmsd"], analysis["top_vs_ts_max"],
             analysis["top_minus_ts_energy"]))
    if analysis["minima"]:
        print("intermediate minima at images %s" % analysis["minima"])
    top_lengths = lengths(images[top], bonds)
    ts_lengths = lengths(sella_ts, bonds)
    print("\n%-22s %10s %10s" % ("bond", "Sella TS", "top image"))
    for name, a, b in zip(names, ts_lengths, top_lengths):
        print("%-22s %10.3f %10.3f" % (name, a, b))
    print("verdict  %s" % outcome)

    plot_band(os.path.join(out_dir, layout.NEB_PLOT), energies, analysis, ts_index,
              title="%s   %s   %s" % (record["guess"], record["reaction"], outcome))

    with open(os.path.join(out_dir, layout.NEB_RESULT), "w") as handle:
        json.dump({"guess": record["guess"], "reaction": record["reaction"],
                   "verdict": outcome, "model": os.path.basename(settings.MACE_MODEL),
                   "n_images": len(images), "ts_index": ts_index,
                   "energies": energies, "start_energies": start_energies,
                   "endpoint_relax": endpoints,
                   "sella_ts_energy": ts_energy,
                   "rough": search["rough"], "climb": search.get("climb"),
                   "converged": search["converged"], "fmax": search["fmax"],
                   "top_image": top, "climbing": settings.NEB_CLIMB,
                   "Ea_fwd": analysis["Ea_fwd"], "Ea_rev": analysis["Ea_rev"],
                   "dE_rxn": analysis["dE_rxn"],
                   "top_vs_ts_rmsd": analysis["top_vs_ts_rmsd"],
                   "top_vs_ts_max": analysis["top_vs_ts_max"],
                   "top_minus_ts_energy": analysis["top_minus_ts_energy"],
                   "minima": analysis["minima"], "path_length": analysis["path_length"],
                   "bonds": [{"name": n, "indices": list(b[:2]), "sella_ts": a, "top": t}
                             for n, b, a, t in zip(names, bonds, ts_lengths, top_lengths)],
                   "settings": {"spring": settings.NEB_SPRING, "method": settings.NEB_METHOD,
                                "fmax_rough": settings.NEB_FMAX_ROUGH,
                                "fmax": settings.NEB_FMAX, "ts_rmsd": settings.NEB_TS_RMSD,
                                "min_depth": settings.NEB_MIN_DEPTH}},
                  handle, indent=2)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])