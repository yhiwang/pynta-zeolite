#!/usr/bin/env python
"""Tables of NEB barriers, one panel per site: one column per reaction, one
row per O pair (atom indices, in the order of neb.json), the cell is Ea_fwd
(eV, from neb_band.traj) in a box colored by the verdict in neb_result.json.
When a pair has several bands (flips, torsions) the cell shows the band with
the best verdict, and the lowest Ea_fwd among those; the small number in the
corner is how many bands the cell holds. A cell reads "unfinished" when its
NEB folders have no neb_band.traj yet. The small table at the right of a
panel gives the crystal label of every site oxygen (from framework.json).

    python analysis/neb_barriers_table.py
    # for the MODEL, ROUND and FRAMEWORK set in scripts/settings.py, writes
    #   <run_dir>/NEB/neb_table.png                      one image per site
    #   runs/<MODEL>/<ROUND>/<FRAMEWORK>/neb_table.png   all sites in one figure
"""

import json
import os
import sys

import numpy as np
from ase.io import read

import _common  # noqa: F401
import layout
import settings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT_PNG = "neb_table.png"
CELL = 0.55                 # inches per square; the font sizes below fit this size
PANELS_PER_ROW = 4          # panels side by side in the all-sites figure

RANK = ["connected", "different_ts", "intermediate", "not_converged", "no json"]
COLOR = {"connected": "#1b9e77", "different_ts": "#e6ab02", "intermediate": "#7570b3",
         "not_converged": "#bdbdbd", "no json": "#e8e8e8"}
EDGE = "#cccccc"            # outline of a cell with no number


def load(path):
    with open(path) as handle:
        return json.load(handle)


def pair_label(pair):
    return " · ".join("O%d" % o for o in pair)


# ---- collect: one panel per site -----------------------------------------------
frame_dir = os.path.join(layout.RUNS, settings.MODEL, settings.ROUND, settings.FRAMEWORK)
panels = []
for site, run_dir in layout.run_dirs(settings.MODEL, settings.ROUND, settings.FRAMEWORK, settings.SITES):
    print("\n=== %s" % run_dir)
    neb_dir = layout.RunLayout(run_dir).neb
    panel = {"site": site, "neb_dir": neb_dir, "rows": [], "cells": {}, "labels": [],
             "names": {}, "models": set()}
    panels.append(panel)
    if not os.path.isdir(neb_dir):
        print("%s not found -- run step 5 first -- empty panel" % neb_dir)
        continue

    # one entry per NEB folder; Ea_fwd stays None until the job wrote neb_band.traj
    bands = []
    for root, dirs, files in sorted(os.walk(neb_dir)):
        if layout.NEB_RECORD not in files:
            continue
        record = load(os.path.join(root, layout.NEB_RECORD))
        rel = os.path.relpath(root, neb_dir)
        band = {"reaction": int(rel.split(os.sep)[0].split("_")[0]), "name": record.get("reaction", ""),
                "pair": tuple(record["oxygens"]), "stem": rel, "verdict": None, "Ea_fwd": None}
        if layout.NEB_FINAL_BAND in files:
            e = np.array([a.get_potential_energy()
                          for a in read(os.path.join(root, layout.NEB_FINAL_BAND), ":")])
            top = np.argmax(e[1:-1]) + 1
            band["Ea_fwd"] = e[top] - e[0]
            band["verdict"] = "no json"
            if layout.NEB_RESULT in files:
                result = load(os.path.join(root, layout.NEB_RESULT))
                band["verdict"] = result["verdict"]
                panel["models"].add(result["model"])
        bands.append(band)

    if not bands:
        print("no NEB folders under %s -- empty panel" % neb_dir)
        continue
    site_reactions = sorted({b["reaction"] for b in bands})
    panel["rows"] = sorted({b["pair"] for b in bands})      # by atom number
    panel["names"] = {b["reaction"]: b["name"] for b in bands}
    panel["labels"] = sorted((s["indices"][0], s["label"])
                             for s in load(os.path.join(run_dir, layout.FRAMEWORK_JSON))["mono_sites"])

    # ---- pick one band per cell: (band, bands with a result), or (None, 0) = unfinished
    for r in site_reactions:
        for p in panel["rows"]:
            part = [b for b in bands if b["reaction"] == r and b["pair"] == p]
            done = [b for b in part if b["Ea_fwd"] is not None]
            if done:
                best_rank = min(RANK.index(b["verdict"]) for b in done)
                best = min((b for b in done if RANK.index(b["verdict"]) == best_rank),
                           key=lambda b: b["Ea_fwd"])
                panel["cells"][(r, p)] = (best, len(done))
            elif part:
                panel["cells"][(r, p)] = (None, 0)

    print("%-14s" % "" + "".join("%12s" % ("R%d" % r) for r in site_reactions))
    for p in panel["rows"]:
        line = "%-14s" % pair_label(p)
        for r in site_reactions:
            best, n = panel["cells"].get((r, p), (None, None))
            if best is not None:
                line += "%7.2f %-4s" % (best["Ea_fwd"], best["verdict"][:4])
            else:
                line += "%12s" % ("-" if n is None else "unfinished")
        print(line)
    for r in site_reactions:
        print("R%d  %s" % (r, panel["names"][r]))
    print("%d of %d NEB folders have no %s" % (sum(b["Ea_fwd"] is None for b in bands), len(bands),
                                              layout.NEB_FINAL_BAND))

reactions = sorted({r for panel in panels for r, p in panel["cells"]})   # the same columns in every panel
names = {r: name for panel in panels for r, name in panel["names"].items()}
if not reactions:
    sys.exit("no NEB folders for any site of %s -- nothing to draw" % frame_dir)


# ---- draw ------------------------------------------------------------------------
def draw(panels, where, out):
    """One figure with ``panels`` side by side, PANELS_PER_ROW per line. All
    positions are in cells: x to the right, y down."""
    ncol = len(reactions)
    step = 2.6 + ncol + 2.9                     # row labels + cells + label table
    per_row = min(PANELS_PER_ROW, len(panels))
    tops, y = [], 1.8                           # y of the top of every line of panels
    for k in range(0, len(panels), per_row):
        tops.append(y)
        y += 2.0 + max(max(len(panel["rows"]), len(panel["labels"]), 3)
                       for panel in panels[k:k + per_row]) + 1.0
    width, height = 0.4 + step * per_row, y + 1.0 + 0.5 * ncol
    fig = plt.figure(figsize=(CELL * width, CELL * height), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    models = sorted({model for panel in panels for model in panel["models"]})
    ax.text(0.4, 0.6, "NEB barriers Ea_fwd (eV)   %s" % ", ".join(models), fontsize=10, va="center")
    ax.text(0.4, 1.3, where, fontsize=9, va="center", color="#555555")

    for k, panel in enumerate(panels):
        x0 = 0.4 + (k % per_row) * step + 2.6
        y0 = tops[k // per_row] + 2.0
        ax.text(x0, y0 - 1.4, panel["site"], fontsize=13, fontweight="bold", va="center")
        for j, r in enumerate(reactions):
            ax.text(x0 + j + .5, y0 - .4, "R%d" % r, ha="center", va="center", fontsize=9)
        if not panel["rows"]:
            ax.add_patch(Rectangle((x0, y0), ncol, 3, facecolor="white", edgecolor="#999999",
                                   lw=1.0, ls=(0, (4, 3))))
            ax.text(x0 + ncol / 2, y0 + 1.5, "no NEB data", ha="center", va="center",
                    fontsize=10, color="#666666")
            continue
        for i, p in enumerate(panel["rows"]):
            ax.text(x0 - .25, y0 + i + .5, pair_label(p), ha="right", va="center", fontsize=8.5)
            for j, r in enumerate(reactions):
                best, n = panel["cells"].get((r, p), (None, None))
                if best is None:                # no NEB folder (n is None), or unfinished
                    ax.add_patch(Rectangle((x0 + j, y0 + i), 1, 1, facecolor="white", edgecolor=EDGE))
                    if n is not None:
                        ax.text(x0 + j + .5, y0 + i + .5, "unfinished", ha="center", va="center",
                                fontsize=6, color="#000000")
                    continue
                ink = "white" if best["verdict"] in ("connected", "intermediate") else "#262626"
                ax.add_patch(Rectangle((x0 + j, y0 + i), 1, 1, facecolor=COLOR[best["verdict"]],
                                       edgecolor="white"))
                ax.text(x0 + j + .5, y0 + i + .5, "%.2f" % best["Ea_fwd"], ha="center", va="center",
                        fontsize=8.5, color=ink)
                if n > 1:
                    ax.text(x0 + j + .93, y0 + i + .08, str(n), ha="right", va="top",
                            fontsize=5.5, color=ink)
        xs = x0 + ncol + .35                    # atom index -> crystal label
        ax.text(xs, y0 - .4, "atom", fontsize=8, va="center", color="#555555")
        ax.text(xs + 1.1, y0 - .4, "label", fontsize=8, va="center", color="#555555")
        for i, (atom, label) in enumerate(panel["labels"]):
            ax.text(xs, y0 + .5 + .8 * i, "O%d" % atom, fontsize=8.5, va="center")
            ax.text(xs + 1.1, y0 + .5 + .8 * i, label, fontsize=8.5, va="center")

    x = 0.4
    for v in RANK:                              # legend: colored box + word
        ax.add_patch(Rectangle((x, y - .17), .5, .34, facecolor=COLOR[v], edgecolor=EDGE, lw=.5))
        ax.text(x + .65, y, v, fontsize=8, va="center")
        x += 1.1 + 0.135 * len(v)
    for n, r in enumerate(reactions):
        ax.text(0.4, y + .9 + .5 * n, "R%d  %s" % (r, names[r]), fontsize=7.5, va="center",
                color="#555555")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote %s" % out)


print()
for panel in panels:                            # one image per site
    if panel["rows"]:
        draw([panel], panel["neb_dir"], os.path.join(panel["neb_dir"], OUT_PNG))
draw(panels, frame_dir, os.path.join(frame_dir, OUT_PNG))   # all sites in one figure