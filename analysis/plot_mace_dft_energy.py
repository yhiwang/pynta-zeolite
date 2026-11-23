#!/usr/bin/env python
"""Parity plots, MACE vs DFT, for the finished single points of one run:
one figure per reaction and one figure with all reactions.

    python analysis/plot_mace_dft_energy.py        # MODEL, ROUND, FRAMEWORK below

It reads every pick under runs/<MODEL>/<ROUND>/<FRAMEWORK>/DFT_sp/<i>_rxn/
(config.extxyz holds the MACE values of the model that sampled it, the
OUTCAR next to it the DFT values) and writes

    DFT_sp/<i>_rxn/mace_vs_dft_parity.png    energy | forces, one reaction
    DFT_sp/mace_vs_dft_parity.png            energy row, force row, one column per reaction

Energy: the whole structure, in eV (the units of a barrier), relative to
the mean of the reaction on each axis:

    x = E_DFT - mean(E_DFT),   y = E_MACE - mean(E_MACE)

All structures of a reaction have the same atoms, so they compare directly;
the one constant this removes is the offset between the two methods (for a
foundation model: its PBE reference against the RPBE-D3 single points). The
table prints that offset. What is left is the scatter that ends up in a
barrier. If a reaction ever holds more than one composition, each gets its
own mean.

Forces: components in eV/A, no reference needed. Atoms whose stored MACE
force is exactly zero are left out: they were fixed when the file was
written and their MACE force was stored with the constraint applied
(ts_sella, the inner neb_start images), while DFT gives their real force.

Color is the pipeline step (step 3 harmonic, step 4 Sella, step 6 NEB) and
the marker is the file, so six files need only three colors.

Left out and counted: picks whose OUTCAR is not finished, whose SCF did not
converge, or that hold no MACE energy. Prints one table. Reading 1000
OUTCARs takes a few minutes.
"""

import json
import os
import sys
from collections import Counter, defaultdict

import _common  # noqa: F401
import layout
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ase.io import read  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, os.path.join(_common.REPO, "dft"))
from vasp_io import finished, read_dft, scf_converged  # noqa: E402

MODEL = "mpa0"                  # the run to plot: runs/<MODEL>/<ROUND>/<FRAMEWORK>/DFT_sp
ROUND = "round0"
FRAMEWORK = "MOR"

FIGURE = "mace_vs_dft_parity.png"
MAX_POINTS = 40000              # force components drawn per panel (all are counted)
# the figure with all reactions, in inches
PANEL = 2.95                    # side of one square panel
COLUMN_GAP = 0.65               # space between two columns
ROW_GAP = 0.70                  # space between the energy row and the force row
LEFT, RIGHT, BOTTOM, TOP = 0.75, 0.20, 0.60, 1.10      # space around the panels
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
STYLE = {                       # file: (color = pipeline step, marker = file)
    "ts_harmonic": (BLUE, "o"),
    "initial": (BLUE, "v"),
    "final": (BLUE, "^"),
    "ts_sella": (ORANGE, "D"),
    "neb_start": (AQUA, "s"),
    "neb_band": (AQUA, "P"),
}
OTHER = ("0.45", "x")           # a file that is not in STYLE
INK, INK2, LINE = "#0b0b0b", "#52514e", "#b5b4ae"


def rmse(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else float("nan")


def mae(x):
    return float(np.mean(np.abs(x))) if len(x) else float("nan")


def force_components(members):
    """(DFT, MACE, color) of every component of every atom with a nonzero
    stored MACE force, and the number of atoms left out as fixed."""
    dft, mace, colors, fixed = [], [], [], 0
    for r in members:
        if r["f_mace"] is None:
            continue
        keep = np.linalg.norm(r["f_mace"], axis=1) > 0
        fixed += int((~keep).sum())
        dft.append(r["f_dft"][keep].ravel())
        mace.append(r["f_mace"][keep].ravel())
        colors += [STYLE.get(r["file"], OTHER)[0]] * (3 * int(keep.sum()))
    if not dft:
        return np.array([]), np.array([]), np.array([]), fixed
    return np.concatenate(dft), np.concatenate(mace), np.array(colors), fixed


def square(ax, unit):
    """Same limits on both axes, the y = x line, labels."""
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], color=LINE, lw=1.0, zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("DFT  (%s)" % unit, color=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LINE)
    ax.tick_params(colors=INK2, labelsize=9)


def note(ax, text):
    ax.text(0.97, 0.04, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color=INK2, linespacing=1.4)


def energy_panel(ax, members):
    for name in files:
        chosen = [r for r in members if r["file"] == name]
        if chosen:
            color, marker = STYLE.get(name, OTHER)
            ax.scatter([r["x"] for r in chosen], [r["y"] for r in chosen], s=30, color=color,
                       marker=marker, alpha=0.85, edgecolors="white", linewidths=0.5)
    error = np.array([r["y"] - r["x"] for r in members])
    square(ax, "eV")
    note(ax, "n = %d\nRMSE %.3f eV\nMAE %.3f eV" % (len(members), rmse(error), mae(error)))


def force_panel(ax, members):
    f_dft, f_mace, colors, _ = force_components(members)
    if len(f_dft):
        order = rng.permutation(len(f_dft))[:MAX_POINTS]    # mixed, so no file hides the others
        ax.scatter(f_dft[order], f_mace[order], s=3, c=list(colors[order]), alpha=0.35,
                   linewidths=0, rasterized=True)
    square(ax, "eV/Å")
    note(ax, "RMSE %.3f eV/Å\nMAE %.3f eV/Å" % (rmse(f_mace - f_dft), mae(f_mace - f_dft)))


def legend_handles(members=None):
    handles = []
    for name in files:
        count = "" if members is None else "  (%d)" % sum(r["file"] == name for r in members)
        color, marker = STYLE.get(name, OTHER)
        handles.append(Line2D([], [], linestyle="", marker=marker, color=color, markersize=7,
                              markeredgecolor="white", markeredgewidth=0.5, label=name + count))
    return handles


dft_sp = os.path.join(layout.framework_dir(MODEL, ROUND, FRAMEWORK), layout.DFT_SP)
if not os.path.isdir(dft_sp):
    sys.exit("%s not found" % dft_sp)
run_name = "%s / %s / %s" % (MODEL, ROUND, FRAMEWORK)

# 1. read every finished pick that holds a MACE energy
rows = []
left_out = Counter()
for reaction in sorted(os.listdir(dft_sp)):
    if not reaction.split("_")[0].isdigit():
        continue                                 # pool/ and other folders
    index = int(reaction.split("_")[0])
    n_before = len(rows)
    for folder, _, names in sorted(os.walk(os.path.join(dft_sp, reaction))):
        if layout.DFT_CONFIG not in names:
            continue
        if not finished(folder):
            left_out["OUTCAR not finished"] += 1
            continue
        if not scf_converged(folder):
            left_out["SCF not converged"] += 1
            continue
        config = read(os.path.join(folder, layout.DFT_CONFIG))
        if "mace_energy" not in config.info:
            left_out["no MACE energy"] += 1
            continue
        try:
            e_dft, f_dft = read_dft(folder, config)
        except Exception as error:                          # noqa: BLE001
            print("skipped %s: %s" % (folder, error))
            left_out["OUTCAR not readable"] += 1
            continue
        with open(os.path.join(folder, layout.DFT_SOURCE)) as handle:
            source = json.load(handle)
        f_mace = config.arrays.get("mace_forces")
        rows.append({"reaction": index, "file": source["file"], "site": source["site"],
                     "formula": config.get_chemical_formula(),
                     "e_dft": float(e_dft), "e_mace": float(config.info["mace_energy"]),
                     "f_dft": np.asarray(f_dft, dtype=float).reshape(-1, 3),
                     "f_mace": None if f_mace is None
                     else np.asarray(f_mace, dtype=float).reshape(-1, 3)})
    print("read R%d: %d single points" % (index, len(rows) - n_before), flush=True)
if not rows:
    sys.exit("no finished single point with a MACE energy under %s  (left out: %s)"
             % (dft_sp, dict(left_out)))

# 2. energies relative to the mean of the reaction (per composition, normally one)
by_reaction = defaultdict(list)
for r in rows:
    by_reaction[r["reaction"]].append(r)
offsets = {}
for index, members in by_reaction.items():
    formulas = sorted({r["formula"] for r in members})
    if len(formulas) > 1:
        print("R%d holds %d compositions: each one has its own mean" % (index, len(formulas)))
    for formula in formulas:
        same = [r for r in members if r["formula"] == formula]
        mean_dft = np.mean([r["e_dft"] for r in same])
        mean_mace = np.mean([r["e_mace"] for r in same])
        offsets.setdefault(index, []).append(mean_mace - mean_dft)
        for r in same:
            r["x"] = r["e_dft"] - mean_dft
            r["y"] = r["e_mace"] - mean_mace
reactions = sorted(by_reaction)
files = [name for name in STYLE if any(r["file"] == name for r in rows)]
files += sorted({r["file"] for r in rows} - set(files))

# 3. table
print("\n%-10s %-12s %5s %10s %10s %10s %10s %12s"
      % ("reaction", "file", "n", "E RMSE", "E MAE", "F RMSE", "F MAE", "E_MACE-E_DFT"))
print("%-10s %-12s %5s %10s %10s %10s %10s %12s"
      % ("", "", "", "eV", "eV", "eV/A", "eV/A", "mean, eV"))


def table_line(label, name, members, offset=""):
    error = np.array([r["y"] - r["x"] for r in members])
    f_dft, f_mace, _, _ = force_components(members)
    print("%-10s %-12s %5d %10.3f %10.3f %10.3f %10.3f %12s"
          % (label, name, len(members), rmse(error), mae(error),
             rmse(f_mace - f_dft), mae(f_mace - f_dft), offset))


for index in reactions:
    members = by_reaction[index]
    for name in files:
        chosen = [r for r in members if r["file"] == name]
        if chosen:
            table_line("R%d" % index, name, chosen)
    table_line("R%d" % index, "all", members,
               ", ".join("%.2f" % offset for offset in offsets[index]))
    print()
table_line("all", "all", rows)
print("(E: relative to the mean of the reaction on each axis, whole structure. "
      "F: atoms with a nonzero stored MACE force)")
for reason, count in sorted(left_out.items()):
    print("(left out: %d, %s)" % (count, reason))

# 4. one figure per reaction: energy | forces
rng = np.random.default_rng(0)
for index in reactions:
    members = by_reaction[index]
    fig, (energy_ax, force_ax) = plt.subplots(1, 2, figsize=(11.5, 5.6))
    energy_panel(energy_ax, members)
    force_panel(force_ax, members)
    energy_ax.set_ylabel("MACE  (eV)", color=INK2)
    force_ax.set_ylabel("MACE  (eV/Å)", color=INK2)
    energy_ax.set_title("Energy, relative to the mean of the reaction", fontsize=10, color=INK)
    force_ax.set_title("Force components, free atoms", fontsize=10, color=INK)
    energy_ax.legend(handles=legend_handles(members), frameon=False, fontsize=9,
                     loc="upper left", labelcolor=INK2, handletextpad=0.3)
    fig.suptitle("%s    R%d    MACE vs DFT" % (run_name, index), fontsize=12, color=INK)
    fig.tight_layout()
    out = os.path.join(dft_sp, layout.reaction_dirname(index), FIGURE)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote %s" % out)

# 5. all reactions: energy row, force row, one column per reaction. The sizes are set
# in inches and every slot is a square, so no label can end up under another panel
n = len(reactions)
width = LEFT + n * PANEL + (n - 1) * COLUMN_GAP + RIGHT
height = BOTTOM + 2 * PANEL + ROW_GAP + TOP
fig, axes = plt.subplots(2, n, figsize=(width, height), squeeze=False)
fig.subplots_adjust(left=LEFT / width, right=1 - RIGHT / width, bottom=BOTTOM / height,
                    top=1 - TOP / height, wspace=COLUMN_GAP / PANEL, hspace=ROW_GAP / PANEL)
for column, index in enumerate(reactions):
    members = by_reaction[index]
    energy_panel(axes[0, column], members)
    force_panel(axes[1, column], members)
    axes[0, column].set_title("R%d" % index, fontsize=11, color=INK)
axes[0, 0].set_ylabel("MACE energy, relative  (eV)", color=INK2)
axes[1, 0].set_ylabel("MACE force components  (eV/Å)", color=INK2)
fig.legend(handles=legend_handles(), frameon=False, fontsize=10, ncol=len(files),
           loc="upper center", bbox_to_anchor=(0.5, 1 - 0.40 / height), labelcolor=INK2,
           handletextpad=0.3)
fig.suptitle("%s    MACE vs DFT, %d single points" % (run_name, len(rows)), fontsize=13, color=INK,
             y=1 - 0.10 / height)
out = os.path.join(dft_sp, FIGURE)
fig.savefig(out, dpi=150)
print("wrote %s" % out)