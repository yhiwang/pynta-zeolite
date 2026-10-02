#!/usr/bin/env python
"""DFT step 1: pick the structures that get a DFT single point, from the
Sella trajectories of step 8.

    python dft/01_collect_configs.py          # dry run first (COLLECT_WRITE = False)

For every finished run (``result.json`` present) under
``<RUN_DIR>/TS_sella/<i>_rxn/pair_<k>/<stem>/<start>/sella.traj``:

1. keep the end points and the frames in between that moved at least
   ``COLLECT_MIN_DISP`` since the previous kept one, at most
   ``COLLECT_MAX_FRAMES`` per trajectory (see ``sampling.select_frames``)
2. drop frames with two atoms closer than ``COLLECT_MIN_DIST``
3. write each kept frame to its own folder, mirroring the TS_sella tree::

    <RUN_DIR>/DFT_sp/<i>_rxn/pair_<k>/<stem>/<start>/f<frame>/config.extxyz
                                                              source.json
    <RUN_DIR>/DFT_sp/manifest.csv          one row per config
    <RUN_DIR>/DFT_sp/all_configs.extxyz    every config, for ase gui

``config.extxyz`` holds no calculator: the MACE energy and forces are kept
as ``mace_energy`` (info) and ``mace_forces`` (arrays), so they can never be
mistaken for the DFT labels later. The FixAtoms constraint is dropped too --
a single point has no constraints, and the DFT forces on the frozen
framework atoms are real training labels.

Folders that already exist are never overwritten (a DFT job may be running
in them); rerunning after more step 8 jobs finish only adds the new ones.
Runs on the login node and submits nothing.
"""

import csv
import json
import os
import sys
from collections import Counter

import _common  # noqa: F401
import dft_settings as ds
import layout
import settings
from ase.io import read, write
from sampling import min_distance, select_frames

MANIFEST_FIELDS = ["config", "reaction", "pair", "stem", "start", "verdict",
                   "frame", "n_frames", "position", "mace_energy", "min_dist",
                   "n_atoms", "n_framework"]


def sella_runs(ts_sella):
    """[(relative dir, dir)] for every ``<i>_rxn/pair_<k>/<stem>/<start>/``
    holding a Sella trajectory (os.walk, not glob: see layout.py)."""
    runs = []
    for dirpath, _, files in os.walk(ts_sella):
        if layout.TS_SELLA_TRAJECTORY not in files:
            continue
        relative = os.path.relpath(dirpath, ts_sella)
        if len(relative.split(os.sep)) == 4:
            runs.append((relative, dirpath))
    return sorted(runs)


def reaction_index(relative):
    return int(relative.split(os.sep)[0].split("_")[0])


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def labelled(atoms, record):
    """A clean copy for DFT: no calculator, no constraint, MACE labels
    renamed, provenance in ``info``."""
    config = atoms.copy()
    config.set_constraint()
    config.calc = None
    try:
        config.arrays["mace_forces"] = atoms.get_forces(apply_constraint=False)
    except (RuntimeError, AttributeError):
        pass
    config.info.clear()
    config.info.update({key: record[key] for key in MANIFEST_FIELDS
                        if record[key] is not None})
    return config


def collect_run(relative, run_dir):
    """(records, configs, n_clash) for one Sella run."""
    result = load_json(os.path.join(run_dir, layout.TS_SELLA_RESULT))
    bonds_path = os.path.join(run_dir, layout.TS_BONDS_JSON)
    bonds = load_json(bonds_path) if os.path.isfile(bonds_path) else {}
    rxn_dir, pair_dir, stem, start = relative.split(os.sep)

    frames = read(os.path.join(run_dir, layout.TS_SELLA_TRAJECTORY), ":")
    picks = select_frames(frames, stride=ds.COLLECT_STRIDE,
                          min_disp=ds.COLLECT_MIN_DISP,
                          max_frames=ds.COLLECT_MAX_FRAMES)
    records, configs, n_clash = [], [], 0
    for index in picks:
        atoms = frames[index]
        dmin = min_distance(atoms)
        if dmin < ds.COLLECT_MIN_DIST:
            n_clash += 1
            continue
        try:
            energy = float(atoms.get_potential_energy())
        except (RuntimeError, AttributeError):
            energy = None
        position = ("first" if index == 0 else
                    "last" if index == len(frames) - 1 else "interior")
        record = {
            "config": os.path.join(relative, "f%04d" % index),
            "reaction": reaction_index(relative),
            "pair": pair_dir,
            "stem": stem,
            "start": start,
            "verdict": result.get("verdict"),
            "frame": index,
            "n_frames": len(frames),
            "position": position,
            "mace_energy": energy,
            "min_dist": round(dmin, 3),
            "n_atoms": len(atoms),
            "n_framework": bonds.get("n_framework"),
            "status": None,
        }
        records.append(record)
        configs.append(labelled(atoms, record))
    return records, configs, n_clash


def write_config(out_root, record, config):
    """Write one config folder; returns "exists", "new" (dry run) or "written"."""
    folder = os.path.join(out_root, record["config"])
    if os.path.isdir(folder):
        return "exists"
    if not ds.COLLECT_WRITE:
        return "new"
    os.makedirs(folder)
    write(os.path.join(folder, layout.DFT_CONFIG), config, format="extxyz")
    source = {key: value for key, value in record.items() if key != "status"}
    source["trajectory"] = os.path.join(          # relative to the run dir
        "TS_sella", os.path.dirname(record["config"]), layout.TS_SELLA_TRAJECTORY)
    with open(os.path.join(folder, layout.DFT_SOURCE), "w") as handle:
        json.dump(source, handle, indent=2)
    return "written"


def main():
    run = layout.RunLayout(settings.RUN_DIR)
    if not os.path.isdir(run.ts_sella):
        sys.exit("%s not found -- run step 8 first" % run.ts_sella)

    runs = sella_runs(run.ts_sella)
    if ds.COLLECT_REACTIONS is not None:
        runs = [r for r in runs if reaction_index(r[0]) in ds.COLLECT_REACTIONS]
    runs = [r for r in runs if r[0].split(os.sep)[3] in ds.COLLECT_STARTS]
    if not runs:
        sys.exit("no Sella trajectories under %s for these settings" % run.ts_sella)

    all_records, all_configs = [], []
    table = Counter()          # (reaction, start, verdict, what) -> count
    n_unfinished = 0
    for relative, run_dir in runs:
        if not os.path.isfile(os.path.join(run_dir, layout.TS_SELLA_RESULT)):
            n_unfinished += 1
            continue
        verdict = load_json(os.path.join(run_dir, layout.TS_SELLA_RESULT)).get("verdict")
        if ds.COLLECT_VERDICTS is not None and verdict not in ds.COLLECT_VERDICTS:
            continue
        records, configs, n_clash = collect_run(relative, run_dir)
        key = (reaction_index(relative), relative.split(os.sep)[3], verdict)
        table[key + ("runs",)] += 1
        table[key + ("frames",)] += len(records)
        table[key + ("clash",)] += n_clash
        for record, config in zip(records, configs):
            record["status"] = write_config(run.dft_sp, record, config)
            table[key + (record["status"],)] += 1
        all_records += records
        all_configs += configs

    # -- report --------------------------------------------------------------
    print("%-9s %-9s %-14s %5s %7s %6s %6s"
          % ("reaction", "start", "verdict", "runs", "frames", "clash", "new"))
    for key in sorted({k[:3] for k in table}):
        new = table[key + ("new",)] + table[key + ("written",)]
        print("%-9s %-9s %-14s %5d %7d %6d %6d"
              % ("R%d" % key[0], key[1], key[2], table[key + ("runs",)],
                 table[key + ("frames",)], table[key + ("clash",)], new))
    n_new = sum(r["status"] in ("new", "written") for r in all_records)
    n_exist = sum(r["status"] == "exists" for r in all_records)
    by_position = Counter(r["position"] for r in all_records)
    print("\n%d configs (%d new, %d already collected)   first %d, interior %d, last %d"
          % (len(all_records), n_new, n_exist, by_position["first"],
             by_position["interior"], by_position["last"]))
    if n_unfinished:
        print("%d Sella runs have no result.json yet -- rerun later to add them"
              % n_unfinished)

    if not ds.COLLECT_WRITE:
        print("\ndry run: nothing written. Set COLLECT_WRITE = True in "
              "dft/dft_settings.py to write %s" % run.dft_sp)
        return

    # manifest and the combined file always describe everything collected so far
    os.makedirs(run.dft_sp, exist_ok=True)
    manifest = os.path.join(run.dft_sp, layout.DFT_MANIFEST)
    with open(manifest, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)
    combined = os.path.join(run.dft_sp, layout.DFT_ALL_CONFIGS)
    write(combined, all_configs, format="extxyz")
    print("\nwrote %s\n      %s" % (manifest, combined))


if __name__ == "__main__":
    main()