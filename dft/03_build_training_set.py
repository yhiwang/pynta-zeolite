#!/usr/bin/env python
"""DFT step 3: turn the finished single points into MACE training files.

    python dft/03_build_training_set.py

For every ``<RUN_DIR>/DFT_sp/<round>/<source>/.../<config>/`` with a finished
OUTCAR, the config is written with its DFT labels under the names MACE
trains on:

    info["REF_energy"]    DFT energy (sigma -> 0), eV
    arrays["REF_forces"]  DFT forces in the config's atom order, eV/A

The MACE labels (mace_energy, mace_forces) and the provenance stay alongside
under their own names; ``round`` and ``source`` are added from the path.
``config_type`` is "R<reaction>_<source>" (R2_neb, R2_harmonic, R2_sella),
so MACE reports errors per reaction and per source.

TRAIN_ROUNDS / TRAIN_SOURCES pick which rounds and sources go in (None: all).

Dropped, and listed in the report:
  - SCF that stopped at NELM without reaching EDIFF
  - any DFT force above TRAIN_MAX_FORCE (a broken geometry, not data)
  - OUTCAR geometry that does not match config.extxyz

Split: each config (TRAIN_SPLIT_BY = "config") or each run / band / state
folder ("run": all frames of one Sella run or one NEB band land in the same
set, so no test frame has a near-twin in train) is assigned to train, valid
or test by a hash of its path and TRAIN_SEED, in the shares
TRAIN_VALID_FRACTION and TRAIN_TEST_FRACTION. A hash, not a shuffle, so
adding configs in a later round never moves the earlier ones.
TRAIN_KEEP_SPLITS reuses an earlier split.csv for the configs it covers, so
a test set stays a test set across rounds; TRAIN_HOLDOUT_REACTIONS sends
whole reactions to test. Pass valid to MACE with --valid_file.

    <RUN_DIR>/DFT_sp/training/train.extxyz
    <RUN_DIR>/DFT_sp/training/valid.extxyz
    <RUN_DIR>/DFT_sp/training/test.extxyz
    <RUN_DIR>/DFT_sp/training/split.csv     which config went where, and why

Read-only on the DFT folders, safe to rerun as more jobs finish (the four
files are rewritten each time). Runs on the login node.
"""

import csv
import hashlib
import os
import sys
from collections import Counter, defaultdict

import _common  # noqa: F401
import dft_settings as ds
import layout
import numpy as np
import settings
from ase.io import read, write
from vasp_io import finished, read_dft, scf_converged

TRAINING_DIR = "training"           # DFT_sp/ (folders starting with this are skipped)
ATOMS_DIR = "DFT_atoms"             # <RUN_DIR>/DFT_atoms/<element>/, dft step 4
TRAIN_FILE = "train.extxyz"
VALID_FILE = "valid.extxyz"
TEST_FILE = "test.extxyz"
SPLIT_FILE = "split.csv"
SETS = ("train", "valid", "test")


def config_folders(root):
    """[(relative, folder)] for every config.extxyz under DFT_sp, filtered
    by TRAIN_ROUNDS / TRAIN_SOURCES (the first two path parts)."""
    found = []
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(TRAINING_DIR)]
        if layout.DFT_CONFIG not in files:
            continue
        relative = os.path.relpath(dirpath, root)
        parts = relative.split(os.sep)
        if len(parts) < 3:
            print("skipped %s: expected DFT_sp/<round>/<source>/..." % relative)
            continue
        if ds.TRAIN_ROUNDS is not None and parts[0] not in ds.TRAIN_ROUNDS:
            continue
        if ds.TRAIN_SOURCES is not None and parts[1] not in ds.TRAIN_SOURCES:
            continue
        found.append((relative, dirpath))
    return sorted(found)


def labelled(relative, folder):
    """(atoms with REF_* labels, None) or (None, reason it was dropped)."""
    if not finished(folder):
        return None, "not finished"
    if not scf_converged(folder):
        return None, "SCF not converged"
    config = read(os.path.join(folder, layout.DFT_CONFIG))
    try:
        energy, forces = read_dft(folder, config)
    except Exception as error:                     # noqa: BLE001
        return None, "unreadable (%s)" % error
    if np.linalg.norm(forces, axis=1).max() > ds.TRAIN_MAX_FORCE:
        return None, "force above %.0f eV/A" % ds.TRAIN_MAX_FORCE
    round_name, source, rxn = relative.split(os.sep)[:3]
    config.info.setdefault("reaction", int(rxn.split("_")[0]))
    config.info["REF_energy"] = energy
    config.arrays["REF_forces"] = forces
    config.info["round"] = round_name
    config.info["source"] = source
    config.info["config_type"] = "R%s_%s" % (config.info["reaction"], source)
    return config, None


def isolated_atoms(run_dir):
    """{element: atoms with REF_energy, config_type=IsolatedAtom} for every
    finished, converged dft step 4 job."""
    found = {}
    root = os.path.join(run_dir, ATOMS_DIR)
    if not os.path.isdir(root):
        return found
    for element in sorted(os.listdir(root)):
        folder = os.path.join(root, element)
        if not (finished(folder) and scf_converged(folder)):
            continue
        atom = read(os.path.join(folder, layout.DFT_CONFIG))
        energy, forces = read_dft(folder, atom)
        atom.info.clear()
        atom.info.update(REF_energy=energy, config_type="IsolatedAtom")
        atom.arrays["REF_forces"] = forces * 0.0
        atom.set_initial_magnetic_moments(None)
        found[element] = atom
    return found


def earlier_splits(dft_sp):
    """{config relative path: set} from the split.csv files in
    TRAIN_KEEP_SPLITS, with the prefix put back on."""
    kept = {}
    for prefix, split_path in ds.TRAIN_KEEP_SPLITS.items():
        path = os.path.join(dft_sp, split_path)
        if not os.path.isfile(path):
            print("warning: %s not found -- TRAIN_KEEP_SPLITS entry for %s ignored"
                  % (path, prefix))
            continue
        with open(path) as handle:
            for row in csv.DictReader(handle):
                if row["set"] in SETS:
                    kept[os.path.join(prefix, row["config"])] = row["set"]
    return kept


def hashed_set(unit):
    """train / valid / test from a hash of the unit's path (stable across reruns)."""
    digest = hashlib.sha1(("%s:%s" % (ds.TRAIN_SEED, unit)).encode()).hexdigest()
    u = int(digest[:12], 16) / 16 ** 12               # uniform in [0, 1)
    if u < ds.TRAIN_TEST_FRACTION:
        return "test"
    if u < ds.TRAIN_TEST_FRACTION + ds.TRAIN_VALID_FRACTION:
        return "valid"
    return "train"


def split(kept, earlier):
    """{config: (set, reason)} for the kept (relative, atoms)."""
    sets = {}
    for relative, atoms in kept:
        if int(atoms.info["reaction"]) in ds.TRAIN_HOLDOUT_REACTIONS:
            sets[relative] = ("test", "held-out reaction")
        elif relative in earlier:
            sets[relative] = (earlier[relative], "earlier split")
        else:
            unit = os.path.dirname(relative) if ds.TRAIN_SPLIT_BY == "run" else relative
            sets[relative] = (hashed_set(unit), "")
    return sets


def main():
    run = layout.RunLayout(settings.RUN_DIR)
    if not os.path.isdir(run.dft_sp):
        sys.exit("%s not found -- run the dft steps first" % run.dft_sp)
    if ds.TRAIN_SPLIT_BY not in ("config", "run"):
        sys.exit('TRAIN_SPLIT_BY must be "config" or "run"')

    kept, dropped = [], Counter()
    split_rows = []
    for relative, folder in config_folders(run.dft_sp):
        atoms, reason = labelled(relative, folder)
        if atoms is None:
            dropped[reason.split(" (")[0]] += 1
            split_rows.append({"config": relative, "set": "dropped", "reason": reason})
            if reason.startswith("unreadable"):
                print("dropped %s: %s" % (relative, reason))
            continue
        kept.append((relative, atoms))
    if not kept:
        sys.exit("no finished single points yet")

    sets = split(kept, earlier_splits(run.dft_sp))
    data = {which: [] for which in SETS}
    for relative, atoms in kept:
        which, reason = sets[relative]
        data[which].append(atoms)
        split_rows.append({"config": relative, "set": which, "reason": reason})

    # isolated atoms (dft step 4) go to train only: MACE takes its E0s from
    # every config_type=IsolatedAtom in the training file
    atoms_e0 = isolated_atoms(settings.RUN_DIR)
    data["train"] += list(atoms_e0.values())
    needed = sorted({s for group in data.values() for a in group
                     for s in a.get_chemical_symbols()})
    missing = [e for e in needed if e not in atoms_e0]

    out = os.path.join(run.dft_sp, TRAINING_DIR)
    os.makedirs(out, exist_ok=True)
    for which, name in zip(SETS, (TRAIN_FILE, VALID_FILE, TEST_FILE)):
        write(os.path.join(out, name), data[which], format="extxyz")
    with open(os.path.join(out, SPLIT_FILE), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["config", "set", "reason"])
        writer.writeheader()
        writer.writerows(sorted(split_rows, key=lambda r: r["config"]))

    # -- report --------------------------------------------------------------
    per_type = defaultdict(Counter)
    per_round = defaultdict(Counter)
    for which, group in data.items():
        for atoms in group:
            per_type[atoms.info["config_type"]][which] += 1
            if "round" in atoms.info:
                per_round[atoms.info["round"]][which] += 1
    print("%-14s %6s %6s %6s" % ("config_type", "train", "valid", "test"))
    for name in sorted(per_type):
        print("%-14s %6d %6d %6d" % ((name,) + tuple(per_type[name][w] for w in SETS)))
    for name in sorted(per_round):
        print("%-14s %6d %6d %6d" % ((name,) + tuple(per_round[name][w] for w in SETS)))
    n_earlier = sum(1 for r in split_rows if r["reason"] == "earlier split")
    print("\nsplit by %s:  train %d, valid %d, test %d configs (%d kept from an earlier split)"
          % (ds.TRAIN_SPLIT_BY, len(data["train"]), len(data["valid"]), len(data["test"]),
             n_earlier))
    print("isolated atoms in train: %s" % (", ".join(
        "%s %.4f eV" % (e, a.info["REF_energy"]) for e, a in atoms_e0.items()) or "none"))
    if missing:
        print("  missing E0 for %s -- run dft/04_isolated_atoms.py (or train with "
              "--E0s=estimated)" % ", ".join(missing))
    for reason, n in sorted(dropped.items()):
        print("dropped %4d  %s" % (n, reason))
    formulas = Counter(a.get_chemical_formula() for group in data.values() for a in group)
    print("%d different compositions: %s" % (len(formulas), ", ".join(
        "%s x%d" % kv for kv in formulas.most_common())))
    print("\nwrote %s/{%s, %s, %s, %s}" % (out, TRAIN_FILE, VALID_FILE, TEST_FILE, SPLIT_FILE))
    print("MACE:  --train_file %s --valid_file %s --test_file %s"
          % (TRAIN_FILE, VALID_FILE, TEST_FILE))
    print("       --energy_key=REF_energy --forces_key=REF_forces")


if __name__ == "__main__":
    main()