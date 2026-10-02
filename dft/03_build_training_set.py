#!/usr/bin/env python
"""DFT step 3: turn the finished single points into MACE training files.

    python dft/03_build_training_set.py

For every ``<RUN_DIR>/DFT_sp/.../f<frame>/`` with a finished OUTCAR, the
config is written with its DFT labels under the names MACE trains on:

    info["REF_energy"]    DFT energy (sigma -> 0), eV
    arrays["REF_forces"]  DFT forces in the config's atom order, eV/A

The MACE labels (mace_energy, mace_forces) and the provenance (reaction,
pair, start, verdict, frame, ...) stay alongside, under their own names.
``config_type`` is set to "R<reaction>_<start>" so MACE reports errors per
group.

Dropped, and listed in the report:
  - SCF that stopped at NELM without reaching EDIFF
  - any DFT force above TRAIN_MAX_FORCE (a broken geometry, not data)
  - OUTCAR geometry that does not match config.extxyz

Split (TRAIN_SPLIT_BY):
  "config"  every config goes to train / valid / test at random, in the
            shares TRAIN_VALID_FRACTION and TRAIN_TEST_FRACTION
  "run"     whole Sella runs are assigned instead (all frames of a run land
            in the same set), so no test frame has a near-twin in train
Either way the shuffle is done within each reaction, so every reaction
appears in train, valid and test in the chosen shares.
TRAIN_HOLDOUT_REACTIONS sends whole reactions to test in both modes. The
seed TRAIN_SEED makes the split reproducible. Pass valid to MACE with
--valid_file (not --valid_fraction).

    <RUN_DIR>/DFT_sp/training/train.extxyz
    <RUN_DIR>/DFT_sp/training/valid.extxyz
    <RUN_DIR>/DFT_sp/training/test.extxyz
    <RUN_DIR>/DFT_sp/training/split.csv     which config went where, and why

Read-only on the DFT folders, safe to rerun as more jobs finish (the three
files are rewritten each time). Runs on the login node.
"""

import csv
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

TRAINING_DIR = "training"           # DFT_sp/
ATOMS_DIR = "DFT_atoms"             # <RUN_DIR>/DFT_atoms/<element>/, dft step 4
TRAIN_FILE = "train.extxyz"
VALID_FILE = "valid.extxyz"
TEST_FILE = "test.extxyz"
SPLIT_FILE = "split.csv"


def config_folders(root):
    found = []
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != TRAINING_DIR]
        if layout.DFT_CONFIG in files:
            found.append((os.path.relpath(dirpath, root), dirpath))
    return sorted(found)


def labelled(folder):
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
    config.info["REF_energy"] = energy
    config.arrays["REF_forces"] = forces
    config.info["config_type"] = "R%s_%s" % (config.info.get("reaction"),
                                             config.info.get("start"))
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


def split(kept):
    """{config: "train" | "valid" | "test"} for the kept (relative, atoms).
    The shuffle is done separately within each reaction (stratified), so
    every reaction appears in all three sets in the chosen shares."""
    rng = np.random.default_rng(ds.TRAIN_SEED)
    sets = {}
    units = defaultdict(lambda: defaultdict(list))   # reaction -> unit -> configs
    for relative, atoms in kept:
        reaction = int(atoms.info["reaction"])
        if reaction in ds.TRAIN_HOLDOUT_REACTIONS:
            sets[relative] = "test"
            continue
        unit = os.path.dirname(relative) if ds.TRAIN_SPLIT_BY == "run" else relative
        units[reaction][unit].append(relative)
    for reaction in sorted(units):
        names = sorted(units[reaction])
        order = rng.permutation(len(names))
        n_test = int(round(ds.TRAIN_TEST_FRACTION * len(names)))
        n_valid = int(round(ds.TRAIN_VALID_FRACTION * len(names)))
        for rank, k in enumerate(order):
            which = ("test" if rank < n_test else
                     "valid" if rank < n_test + n_valid else "train")
            for relative in units[reaction][names[k]]:
                sets[relative] = which
    return sets


def main():
    run = layout.RunLayout(settings.RUN_DIR)
    if not os.path.isdir(run.dft_sp):
        sys.exit("%s not found -- run the dft steps first" % run.dft_sp)

    kept, dropped = [], Counter()
    split_rows = []
    for relative, folder in config_folders(run.dft_sp):
        atoms, reason = labelled(folder)
        if atoms is None:
            dropped[reason.split(" (")[0]] += 1
            split_rows.append({"config": relative, "set": "dropped", "reason": reason})
            if reason.startswith("unreadable"):
                print("dropped %s: %s" % (relative, reason))
            continue
        kept.append((relative, atoms))
    if not kept:
        sys.exit("no finished single points yet")

    if ds.TRAIN_SPLIT_BY not in ("config", "run"):
        sys.exit('TRAIN_SPLIT_BY must be "config" or "run"')
    sets = split(kept)
    data = {"train": [], "valid": [], "test": []}
    for relative, atoms in kept:
        which = sets[relative]
        data[which].append(atoms)
        split_rows.append({"config": relative, "set": which,
                           "reason": "held-out reaction" if
                           atoms.info["reaction"] in ds.TRAIN_HOLDOUT_REACTIONS else ""})

    # isolated atoms (dft step 4) go to train only: MACE takes its E0s from
    # every config_type=IsolatedAtom in the training file
    atoms_e0 = isolated_atoms(settings.RUN_DIR)
    data["train"] += list(atoms_e0.values())
    needed = sorted({s for group in data.values() for a in group
                     for s in a.get_chemical_symbols()})
    missing = [e for e in needed if e not in atoms_e0]

    out = os.path.join(run.dft_sp, TRAINING_DIR)
    os.makedirs(out, exist_ok=True)
    for which, name in (("train", TRAIN_FILE), ("valid", VALID_FILE), ("test", TEST_FILE)):
        write(os.path.join(out, name), data[which], format="extxyz")
    with open(os.path.join(out, SPLIT_FILE), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["config", "set", "reason"])
        writer.writeheader()
        writer.writerows(sorted(split_rows, key=lambda r: r["config"]))

    # -- report --------------------------------------------------------------
    per_type = defaultdict(Counter)
    for which, group in data.items():
        for atoms in group:
            per_type[atoms.info["config_type"]][which] += 1
    print("%-14s %6s %6s %6s" % ("config_type", "train", "valid", "test"))
    for name in sorted(per_type):
        print("%-14s %6d %6d %6d" % (name, per_type[name]["train"],
                                     per_type[name]["valid"], per_type[name]["test"]))
    print("\nsplit by %s:  train %d, valid %d, test %d configs"
          % (ds.TRAIN_SPLIT_BY, len(data["train"]), len(data["valid"]), len(data["test"])))
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