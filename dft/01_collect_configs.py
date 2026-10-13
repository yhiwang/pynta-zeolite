#!/usr/bin/env python
"""DFT step 1: pick the structures that get a DFT single point.

    python dft/01_collect_configs.py          # dry run first (COLLECT_WRITE = False)

Three sources, chosen with COLLECT_SOURCES in dft/dft_settings.py:

  sella      TS_sella/<i>_rxn/pair_<k>/<stem>/<start>/sella.traj  (step 8)
             finished runs only (result.json); the end points plus frames in
             between that moved COLLECT_MIN_DISP, at most COLLECT_MAX_FRAMES
  harmonic   TS_harmonic/<i>_rxn/pair_<k>/<stem>/<state>/  (step 7, initial/final)
             finished states only (endpoint.json); the relaxed structure, plus
             frames of <state>.traj when COLLECT_HARMONIC_MAX_FRAMES > 1. Frame 0
             (the raw step 5 guess, already in sella/raw) is never taken. Only
             the relaxed structure keeps MACE labels: the trajectory frames of
             the spring stages carry MACE + spring energies.
  neb        NEB/<i>_rxn/pair_<k>/<stem>/  (steps 9-10)
             start_band = band.traj (any band step 9 built, no MACE labels),
             final_band = neb_band.traj (finished NEBs only, neb_result.json).
             Images within COLLECT_MIN_DISP of the Sella TS (ts.xyz) are
             skipped -- they are already in sella. The end points of the start
             band are the harmonic end points and are never taken; those of
             the final band (re-relaxed) only with COLLECT_NEB_ENDPOINTS.

Every kept frame goes to its own folder under the round, one tree per source:

    <RUN_DIR>/DFT_sp/<round>/sella/<i>_rxn/pair_<k>/<stem>/<start>/f<frame>/
    <RUN_DIR>/DFT_sp/<round>/harmonic/<i>_rxn/pair_<k>/<stem>/<state>/f<frame>/
    <RUN_DIR>/DFT_sp/<round>/neb/<i>_rxn/pair_<k>/<stem>/<band>/img<NN>/
        config.extxyz   no calculator, no constraint; MACE labels kept as
                        mace_energy (info) / mace_forces (arrays)
        source.json     where it came from
    <RUN_DIR>/DFT_sp/<round>/<source>/manifest.csv, all_configs.extxyz

Frames with two atoms closer than COLLECT_MIN_DIST are dropped. Folders that
already exist are never overwritten (a DFT job may be running in them), so a
rerun only adds new ones. Runs on the login node and submits nothing.
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
from ase.io.trajectory import Trajectory
from sampling import max_displacement, min_distance, select_frames

SOURCES = ("sella", "harmonic", "neb")
NEB_BANDS = {"start_band": layout.NEB_BAND, "final_band": layout.NEB_FINAL_BAND}

MANIFEST_FIELDS = ["config", "source", "reaction", "pair", "stem", "kind", "verdict",
                   "frame", "n_frames", "position", "mace_energy", "min_dist",
                   "n_atoms", "n_framework"]


def load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def find(tree, filename, depth):
    """[(parts, folder)] for every folder ``depth`` levels below ``tree``
    holding ``filename`` (os.walk, not glob: see layout.py)."""
    found = []
    for dirpath, _, files in os.walk(tree):
        if filename not in files:
            continue
        parts = os.path.relpath(dirpath, tree).split(os.sep)
        if len(parts) == depth:
            found.append((parts, dirpath))
    return sorted(found)


def reaction_index(rxn_dir):
    return int(rxn_dir.split("_")[0])


def wanted_reaction(rxn_dir):
    return ds.COLLECT_REACTIONS is None or reaction_index(rxn_dir) in ds.COLLECT_REACTIONS


def wanted_verdict(verdict, allowed):
    return allowed is None or verdict in allowed


# --------------------------------------------------------------------------
# one function per source: returns (units, n_waiting). A unit is one
# trajectory / state / band: where it goes, its verdict and the frames to
# consider as (index, atoms, keep MACE labels).
# --------------------------------------------------------------------------

def sella_units(run):
    units, n_waiting = [], 0
    for parts, folder in find(run.ts_sella, layout.TS_SELLA_TRAJECTORY, 4):
        rxn, pair, stem, start = parts
        if not wanted_reaction(rxn) or start not in ds.COLLECT_STARTS:
            continue
        result = load_json(os.path.join(folder, layout.TS_SELLA_RESULT))
        if result is None:
            n_waiting += 1
            continue
        if not wanted_verdict(result.get("verdict"), ds.COLLECT_VERDICTS):
            continue
        bonds = load_json(os.path.join(folder, layout.TS_BONDS_JSON)) or {}
        frames = read(os.path.join(folder, layout.TS_SELLA_TRAJECTORY), ":")
        picks = select_frames(frames, stride=ds.COLLECT_STRIDE,
                              min_disp=ds.COLLECT_MIN_DISP,
                              max_frames=ds.COLLECT_MAX_FRAMES)
        units.append({"parts": parts, "verdict": result.get("verdict"),
                      "n_framework": bonds.get("n_framework"), "n_frames": len(frames),
                      "frames": [(i, frames[i], True) for i in picks],
                      "origin": os.path.join("TS_sella", *parts, layout.TS_SELLA_TRAJECTORY)})
    return units, n_waiting


def harmonic_units(run):
    units = []
    for parts, folder in find(run.ts_harmonic, layout.ENDPOINT_RESULT, 4):
        rxn, pair, stem, state = parts
        if not wanted_reaction(rxn) or state not in ds.COLLECT_HARMONIC_STATES:
            continue
        result = load_json(os.path.join(folder, layout.ENDPOINT_RESULT))
        if not wanted_verdict(result.get("verdict"), ds.COLLECT_HARMONIC_VERDICTS):
            continue
        record = load_json(os.path.join(os.path.dirname(folder), layout.TS_BONDS_JSON)) or {}
        structure, trajectory, _ = layout.harmonic_files(state)
        relaxed = read(os.path.join(folder, structure))       # MACE, spring-free
        traj_path = os.path.join(folder, trajectory)
        frames = []
        n_frames = 1
        if os.path.isfile(traj_path):
            with Trajectory(traj_path) as traj:
                n_frames = len(traj)
                if (ds.COLLECT_HARMONIC_MAX_FRAMES or 2) > 1 and n_frames > 2:
                    picks = select_frames(traj, stride=ds.COLLECT_STRIDE,
                                          min_disp=ds.COLLECT_MIN_DISP,
                                          max_frames=ds.COLLECT_HARMONIC_MAX_FRAMES)
                    frames = [(i, traj[i], False) for i in picks[:-1] if i != 0]
        frames.append((n_frames - 1, relaxed, True))
        units.append({"parts": parts, "verdict": result.get("verdict"),
                      "n_framework": record.get("n_framework"), "n_frames": n_frames,
                      "frames": frames,
                      "origin": os.path.join("TS_harmonic", *parts, structure)})
    return units, 0


def neb_units(run):
    units, n_waiting = [], 0
    for parts, folder in find(run.neb, layout.NEB_RECORD, 3):
        rxn, pair, stem = parts
        if not wanted_reaction(rxn):
            continue
        record = load_json(os.path.join(folder, layout.NEB_RECORD))
        result = load_json(os.path.join(folder, layout.NEB_RESULT))
        verdict = result.get("verdict") if result else None
        sella_ts = read(os.path.join(folder, layout.NEB_STATES["ts"]))
        for band in ds.COLLECT_NEB_BANDS:
            band_path = os.path.join(folder, NEB_BANDS[band])
            if band == "final_band" and result is None:
                n_waiting += 1
                continue
            if not os.path.isfile(band_path):
                continue
            if result is not None and not wanted_verdict(verdict, ds.COLLECT_NEB_VERDICTS):
                continue
            images = read(band_path, ":")
            ends = band == "final_band" and ds.COLLECT_NEB_ENDPOINTS
            picks = range(len(images)) if ends else range(1, len(images) - 1)
            frames = [(i, images[i], band == "final_band") for i in picks
                      if max_displacement(sella_ts, images[i]) >= ds.COLLECT_MIN_DISP]
            units.append({"parts": parts + [band], "verdict": verdict,
                          "n_framework": record.get("n_framework"), "n_frames": len(images),
                          "frames": frames, "n_near_ts": len(picks) - len(frames),
                          "origin": os.path.join("NEB", *parts, NEB_BANDS[band])})
    return units, n_waiting


UNIT_FUNCTIONS = {"sella": sella_units, "harmonic": harmonic_units, "neb": neb_units}


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------

def labelled(atoms, record, mace):
    """A clean copy for DFT: no calculator, no constraint, MACE labels
    renamed (only when ``mace``), provenance in ``info``."""
    config = atoms.copy()
    config.set_constraint()
    config.calc = None
    if mace:
        try:
            config.arrays["mace_forces"] = atoms.get_forces(apply_constraint=False)
        except (RuntimeError, AttributeError):
            pass
    config.info.clear()
    config.info.update({key: record[key] for key in MANIFEST_FIELDS
                        if record[key] is not None})
    return config


def frame_name(source, index):
    return "img%02d" % index if source == "neb" else "f%04d" % index


def write_config(source_root, record, config, origin):
    """Write one config folder; returns "exists", "new" (dry run) or "written"."""
    folder = os.path.join(source_root, record["config"])
    if os.path.isdir(folder):
        return "exists"
    if not ds.COLLECT_WRITE:
        return "new"
    os.makedirs(folder)
    write(os.path.join(folder, layout.DFT_CONFIG), config, format="extxyz")
    source = {key: value for key, value in record.items() if key != "status"}
    source["trajectory"] = origin                      # relative to the run dir
    with open(os.path.join(folder, layout.DFT_SOURCE), "w") as handle:
        json.dump(source, handle, indent=2)
    return "written"


def collect_source(run, source, table):
    """Collect one source; returns (records, configs, n_waiting)."""
    source_root = os.path.join(run.dft_sp, ds.COLLECT_ROUND, source)
    units, n_waiting = UNIT_FUNCTIONS[source](run)
    records, configs = [], []
    for unit in units:
        rxn, pair, stem, kind = unit["parts"]
        key = (source, reaction_index(rxn), kind, unit["verdict"])
        table[key + ("runs",)] += 1
        table[key + ("near_ts",)] += unit.get("n_near_ts", 0)
        for index, atoms, mace in unit["frames"]:
            dmin = min_distance(atoms)
            if dmin < ds.COLLECT_MIN_DIST:
                table[key + ("clash",)] += 1
                continue
            energy = None
            if mace:
                try:
                    energy = float(atoms.get_potential_energy())
                except (RuntimeError, AttributeError):
                    pass
            last = unit["n_frames"] - 1
            record = {
                "config": os.path.join(rxn, pair, stem, kind, frame_name(source, index)),
                "source": source,
                "reaction": reaction_index(rxn),
                "pair": pair,
                "stem": stem,
                "kind": kind,
                "verdict": unit["verdict"],
                "frame": index,
                "n_frames": unit["n_frames"],
                "position": "first" if index == 0 else "last" if index == last else "interior",
                "mace_energy": energy,
                "min_dist": round(dmin, 3),
                "n_atoms": len(atoms),
                "n_framework": unit["n_framework"],
                "status": None,
            }
            config = labelled(atoms, record, mace)
            record["status"] = write_config(source_root, record, config, unit["origin"])
            table[key + ("frames",)] += 1
            table[key + (record["status"],)] += 1
            records.append(record)
            configs.append(config)
    return records, configs, n_waiting


def write_manifest(source_root, records, configs):
    os.makedirs(source_root, exist_ok=True)
    with open(os.path.join(source_root, layout.DFT_MANIFEST), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    write(os.path.join(source_root, layout.DFT_ALL_CONFIGS), configs, format="extxyz")


def main():
    run = layout.RunLayout(settings.RUN_DIR)
    unknown = [s for s in ds.COLLECT_SOURCES if s not in SOURCES]
    if unknown:
        sys.exit("unknown COLLECT_SOURCES %s, expected some of %s" % (unknown, SOURCES))
    bad_bands = [b for b in ds.COLLECT_NEB_BANDS if b not in NEB_BANDS]
    if bad_bands:
        sys.exit("unknown COLLECT_NEB_BANDS %s, expected some of %s" % (bad_bands, list(NEB_BANDS)))
    print("%s -> %s   sources: %s"
          % (settings.RUN_DIR, os.path.join(run.dft_sp, ds.COLLECT_ROUND),
             ", ".join(ds.COLLECT_SOURCES)))

    table = Counter()
    collected = {}
    for source in ds.COLLECT_SOURCES:
        records, configs, n_waiting = collect_source(run, source, table)
        collected[source] = (records, configs)
        if n_waiting:
            print("%s: %d runs not finished yet -- rerun later to add them" % (source, n_waiting))

    # -- report --------------------------------------------------------------
    print("\n%-9s %-9s %-11s %-14s %5s %7s %6s %8s %6s"
          % ("source", "reaction", "kind", "verdict", "runs", "frames", "clash",
             "near_ts", "new"))
    for key in sorted({k[:4] for k in table}, key=lambda k: tuple(str(x) for x in k)):
        new = table[key + ("new",)] + table[key + ("written",)]
        print("%-9s %-9s %-11s %-14s %5d %7d %6d %8d %6d"
              % (key[0], "R%d" % key[1], key[2], key[3], table[key + ("runs",)],
                 table[key + ("frames",)], table[key + ("clash",)],
                 table[key + ("near_ts",)], new))
    print()
    for source, (records, _) in collected.items():
        status = Counter(r["status"] for r in records)
        print("%-9s %4d configs (%d new, %d already collected)"
              % (source, len(records), status["new"] + status["written"], status["exists"]))

    if not ds.COLLECT_WRITE:
        print("\ndry run: nothing written. Set COLLECT_WRITE = True in dft/dft_settings.py")
        return
    for source, (records, configs) in collected.items():
        if records:
            write_manifest(os.path.join(run.dft_sp, ds.COLLECT_ROUND, source), records, configs)
    print("\nwrote manifest.csv + all_configs.extxyz in each source folder")


if __name__ == "__main__":
    main()