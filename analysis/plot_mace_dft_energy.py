#!/usr/bin/env python
"""MACE (Sella) vs DFT energy for every finished config in DFT_sp.

    python analysis/plot_mace_dft_energy.py [runs/MOR_T4/DFT_sp]

Left:  E_DFT - E_MACE per config. The two use different functionals (and
       MACE has no D3), so this is a big constant offset; only its spread
       matters.
Right: energy relative to the first frame of the same Sella run, DFT vs
       MACE. Points on the diagonal = MACE gets the energy change right.
"""

import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

root = sys.argv[1] if len(sys.argv) > 1 else "runs/MOR_T4/DFT_sp"

# 1. grab both energies from every finished config folder
runs = defaultdict(dict)          # Sella run folder -> {frame: (e_mace, e_dft)}
for dirpath, _, files in os.walk(root):
    if "config.extxyz" not in files or "OUTCAR" not in files:
        continue
    try:
        e_dft = read(os.path.join(dirpath, "OUTCAR")).get_potential_energy()
    except Exception:             # OUTCAR still being written
        continue
    e_mace = read(os.path.join(dirpath, "config.extxyz")).info["mace_energy"]
    frame = int(os.path.basename(dirpath)[1:])        # f0045 -> 45
    runs[os.path.dirname(dirpath)][frame] = (e_mace, e_dft)

# 2. offset per config, and energies relative to each run's first frame, per start
offset = []
rel = {"harmonic": ([], []), "raw": ([], [])}       # start -> (de_dft, de_mace)
for run, frames in runs.items():
    start = os.path.basename(run)                   # .../harmonic or .../raw
    first = frames[min(frames)]
    for frame, (e_mace, e_dft) in frames.items():
        offset.append(e_dft - e_mace)
        if frame != min(frames):
            rel[start][0].append(e_dft - first[1])
            rel[start][1].append(e_mace - first[0])
de_dft = np.array(rel["harmonic"][0] + rel["raw"][0])
de_mace = np.array(rel["harmonic"][1] + rel["raw"][1])

# 3. plot
fig, (left, right) = plt.subplots(1, 2, figsize=(11, 5))
left.hist(offset, bins=40)
left.set_xlabel("E_DFT - E_MACE  [eV]")
left.set_ylabel("configs")
left.set_title("offset per config (n=%d)" % len(offset))

for start, (x, y) in rel.items():
    x, y = np.array(x), np.array(y)
    if len(x):
        right.scatter(x, y, s=12, alpha=0.6,
                      label="%s  (MAE %.3f eV)" % (start, np.abs(y - x).mean()))
right.legend(frameon=False)
lim = [min(de_dft.min(), de_mace.min()), max(de_dft.max(), de_mace.max())]
right.plot(lim, lim, color="0.5", lw=0.8)
mae = np.abs(de_mace - de_dft).mean()
right.set_xlabel("DFT   E - E(first frame)  [eV]")
right.set_ylabel("MACE  E - E(first frame)  [eV]")
right.set_title("relative energy within each Sella run  (MAE %.3f eV)" % mae)

fig.tight_layout()
out = os.path.join(root, "mace_vs_dft_energy.png")
fig.savefig(out, dpi=150)
print("%d configs, %d relative energies, MAE %.3f eV -> %s"
      % (len(offset), len(de_dft), mae, out))