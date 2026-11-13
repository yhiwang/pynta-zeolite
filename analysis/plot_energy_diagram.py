#!/usr/bin/env python
"""Three-level energy diagram (IS, TS, FS), one panel per reaction, and a
copy of the plotted NEB bands.

For each reaction, the panel shows the connected NEB band with the lowest
Ea_fwd over all O pairs and bands. Energies are from neb_band.traj, as in
neb_barriers_table.py: IS = 0, TS = highest interior image - IS,
FS = last image - IS. The chosen O pair is printed, not plotted.

The last section copies each plotted band into OUT_DIR/<i>_rxn/: the NEB
result files, plus IS.xyz, TS.xyz and FS.xyz (first, top and last image of
neb_band.traj, with MACE energy and forces). OUT_DIR is deleted and written
again on every run.

    python analysis/plot_energy_diagram.py
    # writes <NEB_DIR>/energy_diagram.png, <NEB_DIR>/energy_diagram.pdf
    # and <run dir>/mace_neb_results/<i>_rxn/
"""

import json
import os
import shutil

import numpy as np
from ase.io import read, write

NEB_DIR = "runs/MOR_T4/NEB"
OUT_NAME = "energy_diagram"
OUT_DIR = os.path.join(os.path.dirname(NEB_DIR), "mace_neb_results")
RESULT_FILES = ("neb_band.traj", "neb_result.json", "neb.json", "band.json", "neb.png")
NROWS, NCOLS = 2, 4
COLOR = "#1b9e77"            # "connected" color of neb_barriers_table.py

# ---- collect: one entry per connected band ---------------------------------
bands = []
for root, dirs, files in sorted(os.walk(NEB_DIR)):
    if "neb_band.traj" not in files or "neb_result.json" not in files:
        continue
    with open(os.path.join(root, "neb_result.json")) as handle:
        result = json.load(handle)
    if result["verdict"] != "connected":
        continue
    e = np.array([a.get_potential_energy() for a in read(os.path.join(root, "neb_band.traj"), ":")])
    top = np.argmax(e[1:-1]) + 1
    rel = os.path.relpath(root, NEB_DIR)
    with open(os.path.join(root, "neb.json")) as handle:
        record = json.load(handle)
    bands.append({"reaction": int(rel.split(os.sep)[0].split("_")[0]),
                  "pair": " · ".join("O%d" % o for o in record.get("oxygens", [])) or rel.split(os.sep)[1],
                  "stem": rel, "top": top, "Ea_fwd": e[top] - e[0], "dE": e[-1] - e[0]})

# ---- pick the lowest Ea_fwd per reaction ------------------------------------
reactions = sorted({b["reaction"] for b in bands})
best = {r: min((b for b in bands if b["reaction"] == r), key=lambda b: b["Ea_fwd"])
        for r in reactions}

print("%-4s %-12s %8s %8s  %s" % ("", "pair", "Ea_fwd", "dE", "band"))
for r in reactions:
    b = best[r]
    print("R%-3d %-12s %8.2f %8.2f  %s" % (r, b["pair"], b["Ea_fwd"], b["dE"], b["stem"]))

# ---- draw --------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(NROWS, NCOLS, figsize=(3.0 * NCOLS, 3.2 * NROWS), dpi=150,
                         sharey=True, squeeze=False)
X_TS = 2.5                    # peak position; IS bar 0-1, FS bar 4-5
for k, ax in enumerate(axes.flat):
    if k >= len(reactions):
        ax.set_visible(False)
        continue
    r = reactions[k]
    b = best[r]
    ea, de = b["Ea_fwd"], b["dE"]
    ax.plot([0.0, 1.0], [0.0, 0.0], color=COLOR, lw=3, solid_capstyle="butt")
    ax.plot([4.0, 5.0], [de, de], color=COLOR, lw=3, solid_capstyle="butt")
    # two half-parabolas, joined at the TS peak with zero slope
    x_up, x_down = np.linspace(1.0, X_TS, 50), np.linspace(X_TS, 4.0, 50)
    ax.plot(x_up, ea - ea * ((x_up - X_TS) / (X_TS - 1.0)) ** 2, color=COLOR, lw=1.5)
    ax.plot(x_down, ea - (ea - de) * ((x_down - X_TS) / (4.0 - X_TS)) ** 2, color=COLOR, lw=1.5)
    ax.text(X_TS, ea + 0.1, "$E_a$ = %.2f" % ea, ha="center", va="bottom", fontsize=9)
    ax.text(4.5, de - 0.1, "ΔE = %.2f" % de, ha="center", va="top", fontsize=9)
    ax.set_xticks([0.5, 2.5, 4.5])
    ax.set_xticklabels(["IS", "TS", "FS"])
    ax.set_xlim(-0.4, 5.4)
    ax.margins(y=0.15)
    ax.set_title("R%d" % r, fontsize=8, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
for row in axes:
    row[0].set_ylabel("E − E$_{IS}$ (eV, MACE)")
fig.tight_layout()
for ext in ("png", "pdf"):
    out = os.path.join(NEB_DIR, "%s.%s" % (OUT_NAME, ext))
    fig.savefig(out)
    print("wrote %s" % out)

# ---- copy the plotted bands into OUT_DIR -------------------------------------
if os.path.isdir(OUT_DIR):
    shutil.rmtree(OUT_DIR)
for r in reactions:
    b = best[r]
    source = os.path.join(NEB_DIR, b["stem"])
    target = os.path.join(OUT_DIR, b["stem"].split(os.sep)[0])     # same <i>_rxn name as NEB_DIR
    os.makedirs(target)
    for name in RESULT_FILES:
        if os.path.isfile(os.path.join(source, name)):
            shutil.copy2(os.path.join(source, name), target)
        else:
            print("missing %s in %s" % (name, source))
    images = read(os.path.join(source, "neb_band.traj"), ":")
    for name, image in (("IS", images[0]), ("TS", images[b["top"]]), ("FS", images[-1])):
        write(os.path.join(target, "%s.xyz" % name), image)
    print("copied %s -> %s" % (source, target))