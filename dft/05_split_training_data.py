#!/usr/bin/env python
"""DFT step 5: split the finished single points into the training files of N models.

    python dft/05_split_training_data.py

It reads every pick under

    runs/<MODEL>/<round>/<FRAMEWORK>/DFT_sp/<i>_rxn/<site>/<pair>/<stem>/<name>/

for settings.ROUND and the earlier rounds of the model (settings.MODELS),
and gives each structure its DFT values under the names MACE trains on:

    info["REF_energy"]     DFT energy (sigma -> 0), eV
    arrays["REF_forces"]   DFT forces in the atom order of the structure, eV/A

The MACE values of the model that sampled the structure stay as mace_energy
and mace_forces. config_type is R<reaction>, so MACE reports its test
errors per reaction.

Left out, counted in the report and listed in split.csv:

    not finished        no finished OUTCAR
    SCF not converged   VASP stopped at NELM
    unreadable          e.g. the OUTCAR geometry is not the one in config.extxyz
    DFT force           a DFT force above TRAIN_MAX_FORCE (None: no limit)

The split. Each structure gets a number between 0 and 1 from a hash of
TRAIN_SEED and its path, so the same structure always lands in the same
set, also after a later round has added structures:

    test    number below TRAIN_TEST_FRACTION. The model is not in the hash:
            the test set is the same for every model
    valid   a second number, with the model name in the hash: each model
            holds out a different TRAIN_VALID_FRACTION of all structures
    train   the rest

The isolated atoms of dft/04 (runs/DFT_atoms/) go into every train file as
config_type=IsolatedAtom: MACE takes its E0s from them. When an element of
the data has no finished atom, the report is printed and nothing is written.

    runs/<MODEL>/<ROUND>/<FRAMEWORK>/DFT_sp/training/test.extxyz
    runs/<MODEL>/<ROUND>/<FRAMEWORK>/DFT_sp/training/model_<k>/train.extxyz, valid.extxyz
    runs/<MODEL>/<ROUND>/<FRAMEWORK>/DFT_sp/training/split.csv

k = 1 .. TRAIN_MODELS. split.csv has one row per structure: its set in each
model, or why it was left out. All files are written again on every run.
Runs on the login node and submits nothing.
"""

import csv
import hashlib
import os
import sys
from collections import Counter

import _common  # noqa: F401
import dft_settings as ds
import layout
import numpy as np
import settings
from ase.io import read, write
from vasp_io import finished, read_dft, scf_converged


def draw(*parts):
    """A number in [0, 1) from a hash of the parts: the same parts always give the same number."""
    digest = hashlib.sha1(":".join(str(part) for part in parts).encode()).hexdigest()
    return int(digest[:12], 16) / 16 ** 12


if ds.TRAIN_MODELS < 1:
    sys.exit("TRAIN_MODELS in dft/dft_settings.py must be 1 or more")
if ds.TRAIN_VALID_FRACTION + ds.TRAIN_TEST_FRACTION >= 1.0:
    sys.exit("TRAIN_VALID_FRACTION + TRAIN_TEST_FRACTION must be below 1")

models = [layout.model_dirname(number) for number in range(1, ds.TRAIN_MODELS + 1)]
rounds = list(settings.MODELS[settings.MODEL])
rounds = rounds[:rounds.index(settings.ROUND) + 1]          # this round and the earlier ones
out = os.path.join(layout.framework_dir(settings.MODEL, settings.ROUND, settings.FRAMEWORK),
                   layout.DFT_SP, layout.DFT_TRAINING)
if os.path.isdir(out):                 # the fine-tune step trains every model folder it finds here
    old = sorted(name for name in os.listdir(out)
                 if name.startswith(layout.DFT_MODEL_PREFIX) and name not in models)
    if old:
        sys.exit("%s holds %s but TRAIN_MODELS = %d -- delete the folder(s) or raise TRAIN_MODELS"
                 % (out, ", ".join(old), ds.TRAIN_MODELS))
print("%s   rounds: %s   models: %d" % (out, ", ".join(rounds), len(models)))

# -- read ----------------------------------------------------------------------
kept = []               # (split.csv row, structure with the DFT values)
rows = []               # every split.csv row, the left-out structures too
left_out = Counter()    # reason -> number of structures
largest = {}            # reaction index -> largest DFT force of a readable structure
for round_name in rounds:
    dft_sp = os.path.join(layout.framework_dir(settings.MODEL, round_name, settings.FRAMEWORK),
                          layout.DFT_SP)
    if not os.path.isdir(dft_sp):
        sys.exit("%s not found -- the single points of %s must be on this machine"
                 % (dft_sp, round_name))
    for reaction in sorted(os.listdir(dft_sp)):
        if not reaction.split("_")[0].isdigit():
            continue                                         # pool/, training/
        index = int(reaction.split("_")[0])
        for folder, _, names in sorted(os.walk(os.path.join(dft_sp, reaction))):
            if layout.DFT_CONFIG not in names:
                continue
            atoms = read(os.path.join(folder, layout.DFT_CONFIG))
            name = "/".join([round_name] + os.path.relpath(folder, dft_sp).split(os.sep))
            row = {"config": name, "round": round_name, "reaction": index,
                   "site": atoms.info.get("site"), "file": atoms.info.get("file")}
            rows.append(row)
            if not finished(folder):
                row["left_out"] = "not finished"
                left_out["not finished"] += 1
                continue
            if not scf_converged(folder):
                row["left_out"] = "SCF not converged"
                left_out["SCF not converged"] += 1
                continue
            try:
                energy, forces = read_dft(folder, atoms)
            except Exception as error:                       # noqa: BLE001
                row["left_out"] = "unreadable (%s)" % error
                left_out["unreadable"] += 1
                print("left out %s: %s" % (name, row["left_out"]))
                continue
            fmax = float(np.linalg.norm(forces, axis=1).max())
            largest[index] = max(largest.get(index, 0.0), fmax)
            if ds.TRAIN_MAX_FORCE is not None and fmax > ds.TRAIN_MAX_FORCE:
                row["left_out"] = "DFT force %.1f eV/A" % fmax
                left_out["DFT force above %g eV/A" % ds.TRAIN_MAX_FORCE] += 1
                print("left out %s: %s" % (name, row["left_out"]))
                continue
            atoms.pbc = True                 # the DFT was periodic: MACE must see the same neighbours
            atoms.info["REF_energy"] = energy
            atoms.arrays["REF_forces"] = forces
            atoms.info["round"] = round_name
            atoms.info["config_type"] = "R%d" % index
            kept.append((row, atoms))
if not kept:
    sys.exit("no finished single point under %s" % ", ".join(rounds))

# -- split ---------------------------------------------------------------------
valid_share = ds.TRAIN_VALID_FRACTION / (1.0 - ds.TRAIN_TEST_FRACTION)    # of what is not test
for row, _ in kept:
    if draw(ds.TRAIN_SEED, "test", row["config"]) < ds.TRAIN_TEST_FRACTION:
        row.update({model: "test" for model in models})
        continue
    for model in models:
        in_valid = draw(ds.TRAIN_SEED, model, row["config"]) < valid_share
        row[model] = "valid" if in_valid else "train"

# -- isolated atoms --------------------------------------------------------------
atoms_dir = os.path.join(layout.RUNS, layout.DFT_ATOMS)
needed = sorted({symbol for _, atoms in kept for symbol in atoms.get_chemical_symbols()})
isolated, missing = [], []
for element in needed:
    folder = os.path.join(atoms_dir, element)
    if not (finished(folder) and scf_converged(folder)):
        missing.append(element)
        continue
    atom = read(os.path.join(folder, layout.DFT_CONFIG))
    energy, forces = read_dft(folder, atom)
    atom.info.clear()
    atom.info.update(REF_energy=energy, config_type="IsolatedAtom")
    atom.arrays["REF_forces"] = forces * 0.0
    atom.set_initial_magnetic_moments(None)
    isolated.append(atom)

# -- report --------------------------------------------------------------------
counts = Counter()      # (reaction index or "all", what) -> number of structures
for row in rows:
    for key in (row["reaction"], "all"):
        counts[key, "picks"] += 1
        if "left_out" in row:
            counts[key, "left out"] += 1
            continue
        counts[key, "kept"] += 1
        counts[key, "test"] += row[models[0]] == "test"
        for model in models:
            counts[key, model, row[model]] += 1
largest["all"] = max(largest.values())
print("\n%-9s %6s %9s %6s %8s %6s" % ("reaction", "picks", "left out", "kept", "max |F|", "test")
      + "".join("  %13s %5s" % (model + " train", "valid") for model in models))
for key in sorted({row["reaction"] for row in rows}) + ["all"]:
    print("%-9s %6d %9d %6d %8s %6d"
          % (key if key == "all" else "R%d" % key, counts[key, "picks"], counts[key, "left out"],
             counts[key, "kept"], "%.2f" % largest[key] if key in largest else "-",
             counts[key, "test"])
          + "".join("  %13d %5d" % (counts[key, model, "train"], counts[key, model, "valid"])
                    for model in models))
print("max |F|: the largest DFT force on an atom, eV/A (the left-out structures too)")
for reason, number in sorted(left_out.items()):
    print("left out %5d  %s" % (number, reason))
if len(models) > 1:
    shared = sum(all(row[model] == "train" for model in models) for row, _ in kept)
    print("in the train set of every model: %d of %d structures" % (shared, len(kept)))
print("isolated atoms: %s"
      % (", ".join("%s %.4f eV" % (atom.get_chemical_symbols()[0], atom.info["REF_energy"])
                   for atom in isolated) or "none"))
if missing:
    sys.exit("\nno finished isolated atom for %s under %s -- run dft/04_isolated_atoms.py first. "
             "Nothing written" % (", ".join(missing), atoms_dir))

# -- write ---------------------------------------------------------------------
os.makedirs(out, exist_ok=True)
write(os.path.join(out, layout.DFT_TEST),
      [atoms for row, atoms in kept if row[models[0]] == "test"], format="extxyz")
for model in models:
    os.makedirs(os.path.join(out, model), exist_ok=True)
    write(os.path.join(out, model, layout.DFT_TRAIN),
          [atoms for row, atoms in kept if row[model] == "train"] + isolated, format="extxyz")
    write(os.path.join(out, model, layout.DFT_VALID),
          [atoms for row, atoms in kept if row[model] == "valid"], format="extxyz")
with open(os.path.join(out, layout.DFT_SPLIT), "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["config", "round", "reaction", "site", "file",
                                                "left_out"] + models)
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda row: row["config"]))
print("\nwrote %s" % os.path.join(out, layout.DFT_TEST))
for model in models:
    print("wrote %s/{%s, %s}" % (os.path.join(out, model), layout.DFT_TRAIN, layout.DFT_VALID))
print("wrote %s" % os.path.join(out, layout.DFT_SPLIT))