#!/usr/bin/env python
"""MACE step 0, OMAT-0 run only: the OMat24 replay file.

    python mace_ft/00_omat24_replay.py      # with the omat24-read env's python

Reads every sub-dataset in FT_OMAT24_DIR (the train part of the OMat24 1M
subsplit), keeps the configs that pass FT_FILTER_TYPE_PT for FT_ATOMIC_NUMBERS
(the rule 01_train.py uses for the MP file) and prints the counts. With
FT_OMAT24_WRITE it writes them to FT_OMAT24_FILE with REF_energy / REF_forces
(the OMat24 DFT labels; no stress, training does not use it) -- all of them,
or a random FT_OMAT24_KEEP of them (seed FT_SEED). Then set
FT_REPLAY = FT_OMAT24_FILE: 01_train.py hands the file to MACE whole.
"""

import glob
import os
import random
import sys

import _common  # noqa: F401
import ft_settings as fs
from ase.io import write
from ase_db_backends.aselmdb import LMDBDatabase


def passes(numbers, allowed, filter_type):
    """01_train.py's filter rule, on sets of atomic numbers."""
    if filter_type == "none":
        return True
    if filter_type == "combinations":       # only listed elements, not all needed
        return numbers <= allowed
    if filter_type == "exclusive":          # exactly the listed elements
        return numbers == allowed
    return allowed <= numbers               # inclusive: all listed, others allowed


allowed = set(fs.FT_ATOMIC_NUMBERS)
subsets = sorted(os.listdir(fs.FT_OMAT24_DIR)) if os.path.isdir(fs.FT_OMAT24_DIR) else []
if not subsets:
    sys.exit("no OMat24 sub-datasets in %s -- check FT_OMAT24_DIR" % fs.FT_OMAT24_DIR)

kept = []                                   # passing configs, only when writing
n_pass_all = n_all = 0
for subset in subsets:
    n_pass = n = 0
    for path in sorted(glob.glob(os.path.join(fs.FT_OMAT24_DIR, subset, "*.aselmdb"))):
        db = LMDBDatabase(path, readonly=True, use_lock_file=False)
        for row in db.select():
            n += 1
            if not passes(set(row.numbers.tolist()), allowed, fs.FT_FILTER_TYPE_PT):
                continue
            n_pass += 1
            if fs.FT_OMAT24_WRITE:
                atoms = row.toatoms()
                energy, forces = atoms.get_potential_energy(), atoms.get_forces()
                atoms.calc = None
                atoms.info = {"REF_energy": energy}
                atoms.arrays["REF_forces"] = forces
                kept.append(atoms)
        db.close()
    print("%-26s %7d of %8d pass" % (subset, n_pass, n))
    n_pass_all += n_pass
    n_all += n
print("%-26s %7d of %8d pass %s %s"
      % ("total", n_pass_all, n_all, fs.FT_FILTER_TYPE_PT, fs.FT_ATOMIC_NUMBERS))

if not fs.FT_OMAT24_WRITE:
    print("\ncount only: choose FT_OMAT24_KEEP, then set FT_OMAT24_WRITE = True in "
          "mace_ft/ft_settings.py")
    sys.exit()
if n_pass_all == 0:
    sys.exit("no config passes -- nothing to write")
if fs.FT_OMAT24_KEEP is not None:
    if fs.FT_OMAT24_KEEP > len(kept):
        sys.exit("FT_OMAT24_KEEP = %d, but only %d configs pass" % (fs.FT_OMAT24_KEEP, len(kept)))
    picked = sorted(random.Random(fs.FT_SEED).sample(range(len(kept)), fs.FT_OMAT24_KEEP))
    kept = [kept[i] for i in picked]
os.makedirs(os.path.dirname(fs.FT_OMAT24_FILE), exist_ok=True)
write(fs.FT_OMAT24_FILE, kept, format="extxyz")
print("\nwrote %d configs to %s" % (len(kept), fs.FT_OMAT24_FILE))