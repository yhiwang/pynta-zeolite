#!/usr/bin/env python
"""MACE step 0b, OMAT-0 run only: pick the replay from the saved OMat24 configs.

    python mace_ft/00_omat24_sample.py      # omat24-read env, login node, seconds

Reads FT_OMAT24_PASS (written by 00_omat24_filter.py), picks FT_OMAT24_KEEP
configs at random with seed FT_SEED (None = all of them) and writes them to
FT_OMAT24_FILE, the file FT_REPLAY points to for the OMAT-0 run.
"""

import os
import random
import sys

import _common  # noqa: F401
import ft_settings as fs
from ase.io import read, write

if not os.path.isfile(fs.FT_OMAT24_PASS):
    sys.exit("%s not found -- run mace_ft/00_omat24_filter.py first" % fs.FT_OMAT24_PASS)
configs = read(fs.FT_OMAT24_PASS, ":")
n_saved = len(configs)
if fs.FT_OMAT24_KEEP is not None:
    if fs.FT_OMAT24_KEEP > n_saved:
        sys.exit("FT_OMAT24_KEEP = %d, but only %d configs are saved"
                 % (fs.FT_OMAT24_KEEP, n_saved))
    picked = sorted(random.Random(fs.FT_SEED).sample(range(n_saved), fs.FT_OMAT24_KEEP))
    configs = [configs[i] for i in picked]
write(fs.FT_OMAT24_FILE, configs, format="extxyz")
print("wrote %d of %d saved configs to %s (seed %d)"
      % (len(configs), n_saved, fs.FT_OMAT24_FILE, fs.FT_SEED))