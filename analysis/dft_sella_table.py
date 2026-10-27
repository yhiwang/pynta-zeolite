#!/usr/bin/env python
"""Table of the DFT Sella runs (step 12): one column per reaction, one row per
O pair (ordered oxygens, as in neb_barriers_table). The cell is

    dE = E(end) - E(start)   (eV)

colored by the Sella verdict in result.json, with "same" / "not same" under
it:

    start   frame 0 of the first trajectory part (sella.0.traj after a resume,
            else sella.traj): DFT at the MACE NEB TS, before Sella moved
    end     ts_dft.xyz: where Sella stopped
    same    no adsorbate atom moved more than DFT_SELLA_MAX_MOVE from start
            to end (minimum image, pyntaz.neb.displacements)

One TS per cell: TS graph check "ok" first (dft_sella_graph_check.py), then
converged, then the lowest end energy; "no start" and "no result" come last.
The small number in the corner is how many TSs the cell holds. Every cell
with more than one TS also gets a one-line table in OUT_MULTI_PNG, numbered
in the same order; the terminal lists the stem of each number.

The terminal shows, for every TS, dE, the largest adsorbate move and its
atom, the graph verdict with every graph problem and the limit it broke
(dft_sella_graph_check.problem_lines), and the forming/breaking bond lengths
at the start and the end.

    python analysis/dft_sella_table.py      # writes the two PNGs in DFT_TS_DIR
"""

import json
import os
import sys

import numpy as np
from ase.io import read

import _common  # noqa: F401
import layout
from dft_sella_graph_check import (atom_name, check_ts, dft_sella_dirs, load_info,
                                   problem_lines, read_start)
from pyntaz.neb import displacements

# ---- settings -------------------------------------------------------------------
DFT_TS_DIR = "runs/MOR_T4/DFT_TS"
OUT_PNG = "dft_sella_table.png"            # in DFT_TS_DIR
OUT_MULTI_PNG = "dft_sella_multi.png"      # in DFT_TS_DIR

DFT_SELLA_MAX_MOVE = 1.0     # A, same: no adsorbate atom moves more than this, start to end

COLOR = {"converged": ("#85B7EB", "#042C53"),        # (fill, text)
         "not_converged": ("#FAC775", "#412402"),
         "no start": ("#D3D1C7", "#2C2C2A"),
         "no result": ("#D3D1C7", "#2C2C2A")}
AVAILABLE = {"same": 0, "not same": 0, "no start": 1, "no result": 2}


def distance(atoms, i, j):
    return float(atoms.get_distance(i, j, mic=True))


# ---- collect: one entry per TS with a dft_sella/ folder -------------------------
entries = []
for folder in dft_sella_dirs(DFT_TS_DIR):
    info = load_info(folder)
    n_framework = info["atoms"]["n_framework"]
    check = check_ts(folder, DFT_TS_DIR)
    entry = {"reaction": info["where"]["reaction_index"], "name": info["where"]["reaction"],
             "id": info["id"], "pair": " · ".join("O%d" % o for o in info["site"]["oxygens"]),
             "check": check, "graph": check["status"], "status": "no result",
             "verdict": None, "dE": None, "e_end": None, "move": None, "move_atom": None,
             "bonds": []}
    entries.append(entry)
    if not os.path.isfile(os.path.join(folder, layout.TS_SELLA_RESULT)):
        continue
    with open(os.path.join(folder, layout.TS_SELLA_RESULT)) as handle:
        entry["verdict"] = json.load(handle)["verdict"]
    end = read(os.path.join(folder, layout.DFT_SELLA_STRUCTURE))
    entry["e_end"] = float(end.get_potential_energy())
    start, e_start = read_start(folder)
    if start is None:
        entry["status"] = "no start"
        entry["bonds"] = [(b["name"], None, distance(end, *b["indices"])) for b in info["bonds"]]
        continue

    entry["dE"] = entry["e_end"] - e_start
    adsorbate = np.arange(n_framework, len(end))
    moved = displacements(start, end, adsorbate)
    k = int(np.argmax(moved))
    entry["move"], entry["move_atom"] = float(moved[k]), int(adsorbate[k])
    entry["status"] = "same" if entry["move"] <= DFT_SELLA_MAX_MOVE else "not same"
    entry["bonds"] = [(b["name"], distance(start, *b["indices"]), distance(end, *b["indices"]))
                      for b in info["bonds"]]

if not entries:
    sys.exit("no dft_sella/ folder under %s" % DFT_TS_DIR)


def rank(e):
    """Lower is better: graph check, then converged, then end energy."""
    return (AVAILABLE[e["status"]], 0 if e["graph"] == "ok" else 1,
            0 if e["verdict"] == "converged" else 1,
            e["e_end"] if e["e_end"] is not None else np.inf)


reactions = sorted({e["reaction"] for e in entries})
pairs = sorted({e["pair"] for e in entries})
names = {e["reaction"]: e["name"] for e in entries}
cells = {}
for e in entries:
    cells.setdefault((e["reaction"], e["pair"]), []).append(e)
for part in cells.values():
    part.sort(key=rank)
multi = [(r, p) for r in reactions for p in pairs if len(cells.get((r, p), [])) > 1]

# ---- terminal: the table -----------------------------------------------------------
def short(e):
    if e["status"] in ("no result", "no start"):
        return e["status"]
    return "%.2f %s %s" % (e["dE"], "conv" if e["verdict"] == "converged" else "nc",
                           "same" if e["status"] == "same" else "not")


print("%-14s" % "" + "".join("%20s" % ("R%d" % r) for r in reactions))
for p in pairs:
    line = "%-14s" % p
    for r in reactions:
        part = cells.get((r, p))
        if not part:
            line += "%20s" % "-"
        else:
            line += "%20s" % (short(part[0]) + (" [%d]" % len(part) if len(part) > 1 else ""))
    print(line)
print("dE = E(end) - E(start) in eV; conv / nc = converged / not converged; "
      "same / not = largest adsorbate move <= / > %.2f A; [n] = TSs in the cell"
      % DFT_SELLA_MAX_MOVE)
for r in reactions:
    print("R%d  %s" % (r, names[r]))

# ---- terminal: every TS ---------------------------------------------------------
print("\n---- every TS, by cell (# = column in %s) ----" % OUT_MULTI_PNG)
for r in reactions:
    for p in pairs:
        part = cells.get((r, p), [])
        for n, e in enumerate(part):
            print("\nR%d  %s  %s%s" % (r, p, "#%d " % (n + 1) if len(part) > 1 else "", e["id"]))
            line = "    %s   graph %s" % (e["verdict"] or "-", e["graph"])
            if e["dE"] is not None:
                line += "   dE %.3f eV   %s: largest adsorbate move %.3f A, %s" % (
                    e["dE"], e["status"], e["move"], atom_name(e["check"], e["move_atom"]))
            elif e["status"] != e["graph"]:
                line += "   %s" % e["status"]
            print(line)
            for problem in problem_lines(e["check"]):
                print("    " + problem)
            if e["bonds"]:
                print("    %-30s %7s %7s %7s" % ("forming/breaking bond", "start", "end", "change"))
                for name, r0, r1 in e["bonds"]:
                    print("    %-30s %7s %7.3f %7s" % (name, "-" if r0 is None else "%.3f" % r0, r1,
                                                      "-" if r0 is None else "%+.3f" % (r1 - r0)))

# ---- draw ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


def draw_cell(ax, x, y, e, corner=None):
    """One V2 cell at (x, y): fill by Sella verdict, dE, and same / not same."""
    fill, ink = COLOR[e["status"] if e["status"] in ("no result", "no start") else e["verdict"]]
    ax.add_patch(Rectangle((x, y), 1, 1, facecolor=fill, edgecolor="white", linewidth=2))
    if e["status"] in ("no result", "no start"):
        ax.text(x + .5, y + .5, e["status"], ha="center", va="center", fontsize=8, color=ink)
    else:
        ax.text(x + .5, y + .42, "%.2f" % e["dE"], ha="center", va="center", fontsize=10, color=ink)
        ax.text(x + .5, y + .72, e["status"], ha="center", va="center", fontsize=7, color=ink)
    if corner:
        ax.text(x + .95, y + .08, str(corner), ha="right", va="top", fontsize=7, color=ink)


legend = [Patch(color=COLOR["converged"][0], label="converged"),
          Patch(color=COLOR["not_converged"][0], label="not converged"),
          Patch(color=COLOR["no result"][0], label="no result / no start")]
footer = ["same: no adsorbate atom moves more than %.2f A from start to end" % DFT_SELLA_MAX_MOVE,
          "cell: TS graph check ok first (dft_sella_graph_check.py), then converged, "
          "then lowest end energy"] + ["R%d  %s" % (r, names[r]) for r in reactions]

# main table
fig_w = max(1.6 * len(reactions) + 2.6, 7.0)          # wide enough for title and legend
fig_h = 0.7 * len(pairs) + 1.6 + 0.12 * len(footer)
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
for j, r in enumerate(reactions):
    for i, p in enumerate(pairs):
        part = cells.get((r, p))
        if not part:
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor="white", edgecolor="#cccccc"))
            continue
        draw_cell(ax, j, i, part[0], len(part) if len(part) > 1 else None)
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
ax.set_title("DFT Sella   dE = E(end) - E(start) (eV)   %s" % DFT_TS_DIR,
             fontsize=9, loc="left", pad=22)
ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(0, -0.02), ncol=3,
          frameon=False, fontsize=8)
fig.text(0.02, 0.1 / fig_h, "\n".join(footer), fontsize=6, va="bottom", color="#555555")
fig.subplots_adjust(left=1.3 / fig_w, right=1 - 0.15 / fig_w, top=1 - 0.75 / fig_h,
                    bottom=(0.5 + 0.12 * len(footer)) / fig_h)
out = os.path.join(DFT_TS_DIR, OUT_PNG)
fig.savefig(out)
print("\nwrote %s" % out)

# one-line tables: every cell with more than one TS
if not multi:
    print("no cell has more than one TS; %s not written" % OUT_MULTI_PNG)
    sys.exit(0)
width = max(len(cells[k]) for k in multi)
step = 1.6                                               # data units per one-line table
fig_w, fig_h = max(1.1 * width + 2.0, 7.0), 1.2 * len(multi) + 1.6
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
for k, (r, p) in enumerate(multi):
    y = step * k
    ax.text(-0.1, y + 0.9, "R%d  %s" % (r, p), ha="right", va="center", fontsize=9)
    for n, e in enumerate(cells[(r, p)]):
        ax.text(n + .5, y + .2, str(n + 1), ha="center", va="center", fontsize=8, color="#555555")
        draw_cell(ax, n, y + .4, e)
ax.set_xlim(0, width)
ax.set_ylim(step * len(multi), 0)
ax.axis("off")
ax.set_title("DFT Sella: cells with more than one TS, best first (stems in the terminal)\n%s"
             % DFT_TS_DIR, fontsize=9, loc="left")
ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(0, 0), ncol=3,
          frameon=False, fontsize=8)
fig.subplots_adjust(left=1.6 / fig_w, right=0.98, top=1 - 0.6 / fig_h, bottom=0.6 / fig_h)
out = os.path.join(DFT_TS_DIR, OUT_MULTI_PNG)
fig.savefig(out)
print("wrote %s" % out)