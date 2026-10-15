#!/usr/bin/env python
"""Table of NEB barriers: one column per reaction, one row per O pair, the
cell is Ea_fwd (eV, from neb_band.traj) colored by the verdict in
neb_result.json. When a pair has several bands (flips, torsions) the cell
shows the band with the best verdict, and the lowest Ea_fwd among those; the
small number in the corner is how many bands the cell holds.

    python analysis/neb_table.py        # writes <NEB_DIR>/neb_table.png
"""

import json
import os

import numpy as np
from ase.io import read

NEB_DIR = "runs/MOR_T4/NEB"
OUT_PNG = "neb_table.png"

RANK = ["connected", "different_ts", "intermediate", "not_converged", "no json"]
COLOR = {"connected": "#1b9e77", "different_ts": "#e6ab02", "intermediate": "#7570b3",
         "not_converged": "#bdbdbd", "no json": "#e8e8e8"}

# ---- collect: one entry per band -------------------------------------------
bands = []
for root, dirs, files in sorted(os.walk(NEB_DIR)):
    if "neb_band.traj" not in files:
        continue
    e = np.array([a.get_potential_energy() for a in read(os.path.join(root, "neb_band.traj"), ":")])
    top = np.argmax(e[1:-1]) + 1
    rel = os.path.relpath(root, NEB_DIR)
    reaction = int(rel.split(os.sep)[0].split("_")[0])
    with open(os.path.join(root, "neb.json")) as handle:
        record = json.load(handle)
    pair = " · ".join("O%d" % o for o in record.get("oxygens", [])) or rel.split(os.sep)[1]
    result = os.path.join(root, "neb_result.json")
    verdict = json.load(open(result))["verdict"] if os.path.isfile(result) else "no json"
    bands.append({"reaction": reaction, "name": record.get("reaction", ""), "pair": pair,
                  "stem": rel, "verdict": verdict, "Ea_fwd": e[top] - e[0]})

reactions = sorted({b["reaction"] for b in bands})
pairs = sorted({b["pair"] for b in bands})
names = {b["reaction"]: b["name"] for b in bands}
models = {json.load(open(os.path.join(NEB_DIR, b["stem"], "neb_result.json")))["model"]
          for b in bands if b["verdict"] != "no json"}

# ---- pick one band per cell ------------------------------------------------
cells = {}
for r in reactions:
    for p in pairs:
        part = [b for b in bands if b["reaction"] == r and b["pair"] == p]
        if not part:
            continue
        best_rank = min(RANK.index(b["verdict"]) for b in part)
        best = min((b for b in part if RANK.index(b["verdict"]) == best_rank),
                   key=lambda b: b["Ea_fwd"])
        cells[(r, p)] = (best, len(part))

print("%-14s" % "" + "".join("%12s" % ("R%d" % r) for r in reactions))
for p in pairs:
    line = "%-14s" % p
    for r in reactions:
        if (r, p) in cells:
            best, n = cells[(r, p)]
            line += "%7.2f %-4s" % (best["Ea_fwd"], best["verdict"][:4])
        else:
            line += "%12s" % "-"
    print(line)
for r in reactions:
    print("R%d  %s" % (r, names[r]))

# ---- draw --------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

fig, ax = plt.subplots(figsize=(1.6 * len(reactions) + 2.2, 0.7 * len(pairs) + 2.6), dpi=150)
for j, r in enumerate(reactions):
    for i, p in enumerate(pairs):
        if (r, p) not in cells:
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor="white", edgecolor="#cccccc"))
            continue
        best, n = cells[(r, p)]
        ax.add_patch(Rectangle((j, i), 1, 1, facecolor=COLOR[best["verdict"]], edgecolor="white"))
        ax.text(j + .5, i + .5, "%.2f" % best["Ea_fwd"], ha="center", va="center", fontsize=10,
                color="white" if best["verdict"] in ("connected", "intermediate") else "#262626")
        if n > 1:
            ax.text(j + .95, i + .08, str(n), ha="right", va="top", fontsize=7, color="#555555")
ax.set_xlim(0, len(reactions))
ax.set_ylim(len(pairs), 0)
ax.set_xticks([j + .5 for j in range(len(reactions))])
ax.set_xticklabels(["R%d" % r for r in reactions])
ax.set_yticks([i + .5 for i in range(len(pairs))])
ax.set_yticklabels(pairs)
ax.xaxis.tick_top()
ax.tick_params(length=0)
for side in ax.spines.values():
    side.set_visible(False)
ax.set_title("NEB barriers Ea_fwd (eV)   %s   %s" % (NEB_DIR, ", ".join(sorted(models))),
             fontsize=9, loc="left", pad=22)
ax.legend(handles=[Patch(color=COLOR[v], label=v) for v in RANK],
          loc="upper left", bbox_to_anchor=(0, -0.02), ncol=5, frameon=False, fontsize=8)
fig.text(0.02, 0.01, "\n".join("R%d  %s" % (r, names[r]) for r in reactions), fontsize=7,
         va="bottom", color="#555555")
fig.subplots_adjust(left=0.18, right=0.98, top=0.86, bottom=0.07 + 0.02 * len(reactions))
out = os.path.join(NEB_DIR, OUT_PNG)
fig.savefig(out)
print("wrote %s" % out)