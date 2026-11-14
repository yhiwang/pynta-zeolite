#!/usr/bin/env python
"""Three-level energy diagram (IS, TS, FS), one panel per reaction, and a
copy of the plotted NEB bands.

For each reaction, the panel shows the connected NEB band with the lowest
Ea_fwd over all O pairs and bands. Energies are from neb_band.traj, as in
neb_barriers_table.py: IS = 0, TS = highest interior image - IS,
FS = last image - IS. The chosen O pair is printed, not plotted.

The last section copies each plotted band into <run_dir>/mace_neb_results/<i>_rxn/: the NEB
result files, plus IS.xyz, TS.xyz and FS.xyz (first, top and last image of
neb_band.traj, with MACE energy and forces). That folder is deleted and written
again on every run.

    python analysis/plot_energy_diagram.py
    # writes <run_dir>/NEB/energy_diagram.png, energy_diagram.pdf and
    # <run_dir>/mace_neb_results/<i>_rxn/ for every site of FRAMEWORK, with
    # the MODEL and ROUND set in scripts/settings.py
"""

import json
import os
import shutil

import numpy as np
from ase.io import read, write

import _common  # noqa: F401
import layout
import settings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_NAME = "energy_diagram"
OUT_SUBDIR = "mace_neb_results"   # <run_dir>/mace_neb_results/
RESULT_FILES = ("neb_band.traj", "neb_result.json", "neb.json", "band.json", "neb.png")
NROWS, NCOLS = 2, 4
COLOR = "#1b9e77"            # "connected" color of neb_barriers_table.py

for site, run_dir in layout.run_dirs(settings.MODEL, settings.ROUND, settings.FRAMEWORK, settings.SITES):
    print("\n=== %s" % run_dir)
    neb_dir = layout.RunLayout(run_dir).neb
    out_dir = os.path.join(run_dir, OUT_SUBDIR)
    if not os.path.isdir(neb_dir):
        print("%s not found -- run step 6 first -- site skipped" % neb_dir)
        continue

    # ---- collect: one entry per connected band ---------------------------------
    bands = []
    for root, dirs, files in sorted(os.walk(neb_dir)):
        if "neb_band.traj" not in files or "neb_result.json" not in files:
            continue
        with open(os.path.join(root, "neb_result.json")) as handle:
            result = json.load(handle)
        if result["verdict"] != "connected":
            continue
        e = np.array([a.get_potential_energy() for a in read(os.path.join(root, "neb_band.traj"), ":")])
        top = np.argmax(e[1:-1]) + 1
        rel = os.path.relpath(root, neb_dir)
        with open(os.path.join(root, "neb.json")) as handle:
            record = json.load(handle)
        bands.append({"reaction": int(rel.split(os.sep)[0].split("_")[0]),
                      "pair": " · ".join("O%d" % o for o in record.get("oxygens", [])) or rel.split(os.sep)[1],
                      "stem": rel, "top": top, "Ea_fwd": e[top] - e[0], "dE": e[-1] - e[0]})

    if not bands:
        print("no connected NEB bands under %s -- site skipped" % neb_dir)
        continue

    # ---- pick the lowest Ea_fwd per reaction ------------------------------------
    reactions = sorted({b["reaction"] for b in bands})
    best = {r: min((b for b in bands if b["reaction"] == r), key=lambda b: b["Ea_fwd"])
            for r in reactions}

    print("%-4s %-12s %8s %8s  %s" % ("", "pair", "Ea_fwd", "dE", "band"))
    for r in reactions:
        b = best[r]
        print("R%-3d %-12s %8.2f %8.2f  %s" % (r, b["pair"], b["Ea_fwd"], b["dE"], b["stem"]))

    # ---- draw --------------------------------------------------------------------
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
        out = os.path.join(neb_dir, "%s.%s" % (OUT_NAME, ext))
        fig.savefig(out)
        print("wrote %s" % out)
    plt.close(fig)

    # ---- copy the plotted bands into out_dir -------------------------------------
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    for r in reactions:
        b = best[r]
        source = os.path.join(neb_dir, b["stem"])
        target = os.path.join(out_dir, b["stem"].split(os.sep)[0])     # same <i>_rxn name as neb_dir
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