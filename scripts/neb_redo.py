#!/usr/bin/env python
"""Finish NEB bands whose job died after the NEB itself was done (the band
written as neb_band.traj, no neb_result.json yet), without rerunning the
NEB. Redoes everything neb_one.py does after the optimizer: the comparison
with the Sella TS, the verdict, neb.png and neb_result.json, and appends the
analysis block to job.out.

    python scripts/neb_redo.py                         # every such band under <RUN_DIR>/NEB
    python scripts/neb_redo.py runs/MOR_T4/NEB/2_rxn   # only under this folder

Needs the MACE model for a few single points per band (the Sella TS and the
interior images of the starting band), so run it on a compute node.
"""

import json
import os
import re
import sys

import _common  # noqa: F401
import layout
import settings
from ase.io import read
from neb_one import lengths
from pyntaz.neb import analyze_band, plot_band, verdict
from pyntaz.ts_sella import load_mace

JOB_OUT = "job.out"            # what 10_submit_neb.py names the job's stdout

SEARCH_LINE = re.compile(r"^(rough|climb)\s+(\d+) steps, fmax ([\d.]+) eV/A, (converged|NOT converged)")
ENDPOINT_LINE = re.compile(r"^(initial|final)\s+relaxed in the band's free region: "
                           r"([-+\d.]+) eV, (\d+) steps, max move ([\d.]+) A(.*)$")


def parse_job_out(path):
    """The search summary and endpoint relax lines neb_one.py printed before it died."""
    search, endpoints = {}, {}
    with open(path) as handle:
        for line in handle:
            m = SEARCH_LINE.match(line)
            if m:
                search[m.group(1)] = {"n_steps": int(m.group(2)), "fmax": float(m.group(3)),
                                      "converged": m.group(4) == "converged"}
            m = ENDPOINT_LINE.match(line)
            if m:
                endpoints[m.group(1)] = {"energy_change": float(m.group(2)), "n_steps": int(m.group(3)),
                                         "max_displacement": float(m.group(4)),
                                         "converged": "NOT converged" not in m.group(5)}
    if "rough" not in search:
        return None, endpoints
    last = search.get("climb", search["rough"])
    search.update(converged=last["converged"], fmax=last["fmax"])
    return search, endpoints


def redo(out_dir, model):
    with open(os.path.join(out_dir, layout.NEB_BAND_INFO)) as handle:
        band = json.load(handle)
    with open(os.path.join(out_dir, layout.NEB_RECORD)) as handle:
        record = json.load(handle)
    n_framework = record["n_framework"]
    bonds = [(b["indices"][0], b["indices"][1], b["change"]) for b in record["bonds"]]
    names = ["%s %s" % (b["change"], "-".join(b["tags"])) for b in record["bonds"]]
    ts_index = band["ts_index"]

    search, endpoints = parse_job_out(os.path.join(out_dir, JOB_OUT))
    if search is None:
        print("  no search summary in job.out, skipped")
        return
    images = read(os.path.join(out_dir, layout.NEB_FINAL_BAND), ":")
    start = read(os.path.join(out_dir, layout.NEB_BAND), ":")
    for atoms in images + start:
        atoms.pbc = True
    energies = [float(a.get_potential_energy()) for a in images]
    sella_ts = start[ts_index]
    ts_energy = float(model.get_property("energy", sella_ts))
    # start energies: the re-relaxed endpoints (they do not move in the NEB) and the
    # starting interior images, as neb_one.py printed them
    start_energies = [energies[0]] + [float(model.get_property("energy", a)) for a in start[1:-1]] \
        + [energies[-1]]

    analysis = analyze_band(images, energies, sella_ts, n_framework, ts_energy=ts_energy,
                            min_depth=settings.NEB_MIN_DEPTH, bonds=bonds)
    outcome = verdict(search, analysis, settings.NEB_TS_RMSD, settings.NEB_TS_BOND)
    top = analysis["top_image"]
    top_lengths = lengths(images[top], bonds)
    ts_lengths = lengths(sella_ts, bonds)

    lines = ["", "(analysis redone from neb_band.traj by neb_redo.py)",
             "highest image %d: Ea_fwd %.3f, Ea_rev %.3f, dE %.3f eV"
             % (top, analysis["Ea_fwd"], analysis["Ea_rev"], analysis["dE_rxn"]),
             "vs Sella TS: adsorbate RMSD %.3f A (max %.3f), E %+.3f eV"
             % (analysis["top_vs_ts_rmsd"], analysis["top_vs_ts_max"], analysis["top_minus_ts_energy"])]
    if analysis["minima"]:
        lines.append("intermediate minima at images %s" % analysis["minima"])
    lines += ["", "%-22s %10s %10s" % ("bond", "Sella TS", "top image")]
    lines += ["%-22s %10.3f %10.3f" % (n, a, b) for n, a, b in zip(names, ts_lengths, top_lengths)]
    lines += ["largest reacting-bond change %.3f A (limit %.2f)"
              % (analysis["top_vs_ts_bond_max"], settings.NEB_TS_BOND), "verdict  %s" % outcome]
    with open(os.path.join(out_dir, JOB_OUT), "a") as handle:
        handle.write("\n".join(lines) + "\n")
    print("  %-14s Ea_fwd %.3f  Ea_rev %.3f  dE %+.3f  bond max %.3f A"
          % (outcome, analysis["Ea_fwd"], analysis["Ea_rev"], analysis["dE_rxn"],
             analysis["top_vs_ts_bond_max"]))

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
                   "top_vs_ts_bond_max": analysis["top_vs_ts_bond_max"],
                   "top_minus_ts_energy": analysis["top_minus_ts_energy"],
                   "minima": analysis["minima"], "path_length": analysis["path_length"],
                   "bonds": [{"name": n, "indices": list(b[:2]), "sella_ts": a, "top": t}
                             for n, b, a, t in zip(names, bonds, ts_lengths, top_lengths)],
                   "settings": {"spring": settings.NEB_SPRING, "method": settings.NEB_METHOD,
                                "fmax_rough": settings.NEB_FMAX_ROUGH,
                                "fmax": settings.NEB_FMAX, "ts_rmsd": settings.NEB_TS_RMSD,
                                "ts_bond": settings.NEB_TS_BOND,
                                "min_depth": settings.NEB_MIN_DEPTH},
                   "redone_from_files": True},
                  handle, indent=2)


if __name__ == "__main__":
    top_dir = sys.argv[1] if len(sys.argv) > 1 else layout.RunLayout(settings.RUN_DIR).neb
    todo = [root for root, dirs, files in sorted(os.walk(top_dir))
            if layout.NEB_FINAL_BAND in files and layout.NEB_RESULT not in files]
    if not todo:
        sys.exit("nothing to redo under %s" % top_dir)
    print("%d bands to finish under %s" % (len(todo), top_dir))
    model = load_mace(settings.MACE_MODEL)
    for folder in todo:
        print(os.path.relpath(folder, top_dir))
        redo(folder, model)