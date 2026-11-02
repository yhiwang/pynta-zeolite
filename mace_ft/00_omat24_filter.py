#!/usr/bin/env python
"""MACE step 0a, OMAT-0 run only: save every OMat24 config that passes the filter.

    python mace_ft/00_omat24_filter.py      # omat24-read env, as a CPU job (~4 h)

Reads every .aselmdb file under FT_OMAT24_DIR (the unpacked full OMat24 train
set, any folder layout), keeps the configs that pass FT_FILTER_TYPE_PT for
FT_ATOMIC_NUMBERS (the rule 01_train.py uses for the MP file), prints the
counts per file and writes them all to FT_OMAT24_PASS with REF_energy /
REF_forces (the OMat24 DFT labels; no stress, training does not use it).
00_omat24_sample.py then picks the replay from that file.
"""

import glob
import os
import sys

import numpy as np

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
paths = sorted(glob.glob(os.path.join(fs.FT_OMAT24_DIR, "**", "*.aselmdb"), recursive=True))
if not paths:
    sys.exit("no .aselmdb files under %s -- check FT_OMAT24_DIR" % fs.FT_OMAT24_DIR)

kept = []
n_pass_all = n_all = 0
for path in paths:
    n_pass = n = 0
    db = LMDBDatabase(path, readonly=True, use_lock_file=False)
    for row in db.select():
        n += 1
        if not passes({int(z) for z in row.numbers}, allowed, fs.FT_FILTER_TYPE_PT):  # list or array
            continue
        n_pass += 1
        atoms = row.toatoms()
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float)
        atoms.calc = None
        atoms.info = {"REF_energy": energy}
        atoms.arrays["REF_forces"] = forces
        kept.append(atoms)
    db.close()
    print("%-60s %7d of %9d pass" % (os.path.relpath(path, fs.FT_OMAT24_DIR), n_pass, n),
          flush=True)
    n_pass_all += n_pass
    n_all += n
print("%-60s %7d of %9d pass %s %s"
      % ("total", n_pass_all, n_all, fs.FT_FILTER_TYPE_PT, fs.FT_ATOMIC_NUMBERS))

if n_pass_all == 0:
    sys.exit("no config passes -- nothing to write")
os.makedirs(os.path.dirname(fs.FT_OMAT24_PASS), exist_ok=True)
write(fs.FT_OMAT24_PASS, kept, format="extxyz")
print("wrote %d configs to %s" % (len(kept), fs.FT_OMAT24_PASS))