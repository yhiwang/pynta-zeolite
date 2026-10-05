"""One table and one figure for the initial and final states of step 7
(used by analysis/summarize_endpoints.py).

    collect        walk TS_harmonic/ and return one row per guess
    write_csv      the rows as a CSV, fixed column order
    plot_summary   the figure:
                     A  survivors per O pair and reaction, "survived / guesses"
                     B  outcome shares per reaction: initial, final, both
                     C  where each endpoint ended: normalised breaking vs
                        forming bond length
                     D  energies of the survivors within each reaction,
                        relative to that reaction's lowest initial state

Each endpoint (one state of one guess) gets a status:

    ok              endpoint.json verdict "ok" and the free relax converged
    wrong_bonds     a reacting bond ended on the wrong side
    other_bonds     some other adsorbate bond changed
    not_converged   verdict "ok" but the free relax hit its step cap
    failed          no endpoint.json and the job's stderr has a traceback
    pending         no endpoint.json yet

A guess *survived* when both its initial and final state are ok. Colors: by
default green for ok / survived and red for anything else (light grey while
a job has no result yet); ``reasons=True`` (summarize_endpoints.py --reasons)
colors by status instead. The CSV always holds the full status.

Bond lengths are normalised by the sum of the ASE covalent radii,
r / (r_i + r_j): 1 is bonded, above the step 7 bond cutoff (1.25) apart.
"""

import csv
import json
import os
from collections import Counter

import numpy as np
from ase.data import covalent_radii
from ase.io import read

from ts_summary import (EMPTY, GREEN, GREY, GRID, INK, RED, SOFT, _family_label,
                        _job_state, _reaction_index, _style, _subdirs)

STATES = ("initial", "final")
STATUSES = ("ok", "wrong_bonds", "other_bonds", "not_converged", "failed", "pending")
LABELS = {"ok": "ok", "wrong_bonds": "wrong bonds", "other_bonds": "other bonds",
          "not_converged": "not converged", "failed": "job failed", "pending": "no result yet"}
COLORS = {"ok": GREEN, "wrong_bonds": "#d95f02", "other_bonds": "#e6ab02",
          "not_converged": "#bdbdbd", "failed": "#636363", "pending": GREY}
RANK = {status: rank for rank, status in enumerate(STATUSES)}   # lower is better
SIMPLE = {"ok": ("survived", GREEN), "pending": ("no result yet", GREY)}
SIMPLE_OTHER = ("lost", RED)

COLUMNS = ["reaction", "reaction_name", "reaction_family", "pair", "sites", "O_a", "O_b",
           "flip", "stem", "survived", "initial", "final", "E_initial", "E_final", "dE_rxn",
           "initial_steps", "final_steps", "initial_break_norm", "initial_form_norm",
           "final_break_norm", "final_form_norm", "initial_other", "final_other"]

# file names inside TS_harmonic/; summarize_endpoints.py passes scripts/layout.py's
FILES = {"info": "info.json", "bonds": "ts_bonds.json", "result": "endpoint.json",
         "err": "job.err", "guess_suffix": "_init.xyz"}


# --------------------------------------------------------------------------
# collecting
# --------------------------------------------------------------------------

def status_of(result):
    """Status of one endpoint from its endpoint.json dict."""
    if result["verdict"] == "ok" and not result.get("free_converged", True):
        return "not_converged"
    return result["verdict"]


def guess_status(row):
    """Status of a whole guess: the worse of its two endpoints."""
    return max((row[state] for state in STATES), key=lambda status: RANK[status])


def _mean_norm(bonds, change, radius):
    values = [b["r"] / (radius[b["indices"][0]] + radius[b["indices"][1]])
              for b in bonds if b["change"] == change]
    return float(np.mean(values)) if values else None


def collect(ts_harmonic_dir, files=None):
    """[row dict] for every TS_harmonic/<i>_rxn/pair_<k>/<stem>/ guess with a
    ts_bonds.json, sorted by reaction, pair, stem. Keys are :data:`COLUMNS`
    plus ``reaction_label``, ``reaction_short`` and ``pair_label`` for plotting."""
    files = dict(FILES, **(files or {}))
    rows = []
    for reaction_name in _subdirs(ts_harmonic_dir):
        index = _reaction_index(reaction_name)
        if index is None:
            continue
        reaction_dir = os.path.join(ts_harmonic_dir, reaction_name)
        info = {}
        info_path = os.path.join(reaction_dir, files["info"])
        if os.path.isfile(info_path):
            with open(info_path) as handle:
                info = json.load(handle)
        reaction = info.get("reaction", reaction_name)
        family = info.get("reaction_family", "")
        for pair in _subdirs(reaction_dir):
            pair_record = info.get("pairs", {}).get(pair, {})
            oxygens = sorted(pair_record.get("oxygens", []))
            sites = pair_record.get("sites", [])
            for stem in _subdirs(os.path.join(reaction_dir, pair)):
                guess_dir = os.path.join(reaction_dir, pair, stem)
                if not os.path.isfile(os.path.join(guess_dir, files["bonds"])):
                    continue
                row = {column: None for column in COLUMNS}
                row.update({"reaction": index, "reaction_name": reaction,
                            "reaction_family": family, "pair": pair,
                            "sites": " / ".join(sites),
                            "O_a": oxygens[0] if oxygens else None,
                            "O_b": oxygens[1] if len(oxygens) > 1 else None,
                            "flip": int(stem[4]) if stem.startswith("flip") else None,
                            "stem": stem, "reaction_label": "R%d" % index,
                            "reaction_short": _family_label(family, reaction),
                            "pair_label": " · ".join("O%d" % o for o in oxygens) or pair})
                radius = None
                for state in STATES:
                    state_dir = os.path.join(guess_dir, state)
                    result_path = os.path.join(state_dir, files["result"])
                    if not os.path.isfile(result_path):
                        row[state] = _job_state(state_dir, files)
                        continue
                    with open(result_path) as handle:
                        result = json.load(handle)
                    if radius is None:
                        atoms = read(os.path.join(guess_dir, stem + files["guess_suffix"]))
                        radius = covalent_radii[atoms.get_atomic_numbers()]
                    row.update({state: status_of(result),
                                "E_" + state: result["energy"],
                                state + "_steps": result.get("free_steps"),
                                state + "_break_norm": _mean_norm(result["bonds"], "break", radius),
                                state + "_form_norm": _mean_norm(result["bonds"], "form", radius),
                                state + "_other": " | ".join(
                                    "%s %d-%d" % (o["change"], *o["indices"])
                                    for o in result.get("other", []))})
                row["survived"] = row["initial"] == "ok" and row["final"] == "ok"
                if row["survived"]:
                    row["dE_rxn"] = row["E_final"] - row["E_initial"]
                rows.append(row)
    return rows


def write_csv(rows, path):
    """``rows`` as a CSV with :data:`COLUMNS`; floats rounded for reading."""
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (round(value, 4) if isinstance(value, float) else value)
                             for key, value in row.items()})


def survivor_counts(rows):
    """{reaction: Counter} with ``guesses``, ``initial`` and ``final`` (ok
    endpoints) and ``survived``, for printing."""
    counts = {}
    for row in rows:
        count = counts.setdefault(row["reaction"], Counter())
        count["guesses"] += 1
        count["initial"] += row["initial"] == "ok"
        count["final"] += row["final"] == "ok"
        count["survived"] += row["survived"]
    return counts


# --------------------------------------------------------------------------
# the figure
# --------------------------------------------------------------------------

def paint(status, reasons):
    """(label, color) of ``status``: by status with ``reasons``, else the
    two-color scheme of :data:`SIMPLE`."""
    if reasons:
        return LABELS[status], COLORS[status]
    return SIMPLE.get(status, SIMPLE_OTHER)


def legend_entries(statuses, reasons):
    """[(label, color)] in key order for the statuses present, merged in the
    two-color scheme."""
    entries = []
    for status in STATUSES:
        if status in statuses and paint(status, reasons) not in entries:
            entries.append(paint(status, reasons))
    return entries


def _panel_map(ax, rows, reactions, pairs, column_labels, row_labels, reasons):
    """A: pair x reaction, survived / guesses, colored by the best guess."""
    from matplotlib.patches import Rectangle

    for column, reaction in enumerate(reactions):
        for line, pair_label in enumerate(pairs):
            cell = [r for r in rows if r["reaction"] == reaction and r["pair_label"] == pair_label]
            if not cell:
                ax.add_patch(Rectangle((column + .02, line + .05), .96, .9, fc=EMPTY, ec="none"))
                ax.text(column + .5, line + .5, "no guess", ha="center", va="center",
                        fontsize=7.5, color="#b5b5b5")
                continue
            best = min((guess_status(r) for r in cell), key=lambda s: RANK[s])
            ax.add_patch(Rectangle((column + .02, line + .05), .96, .9,
                                   fc=paint(best, reasons)[1], ec="none"))
            ax.text(column + .5, line + .5, "%d/%d" % (sum(r["survived"] for r in cell), len(cell)),
                    ha="center", va="center", fontsize=8.5, fontweight="bold",
                    color=INK if best == "pending" or (reasons and best in ("not_converged", "other_bonds"))
                    else "white")
    ax.set_xlim(0, len(reactions))
    ax.set_ylim(len(pairs), 0)
    ax.set_xticks(np.arange(len(reactions)) + .5)
    ax.set_xticklabels([column_labels[r] for r in reactions], fontsize=8.8, color=INK)
    ax.set_yticks(np.arange(len(pairs)) + .5)
    ax.set_yticklabels([row_labels[p] for p in pairs], fontsize=8.8, color=INK)
    ax.xaxis.tick_top()
    _style(ax, keep=())


def _panel_bars(ax, rows, reactions, reasons):
    """B: outcome shares, one bar each for initial, final and both."""
    ticks, labels, y = [], [], 0.0
    for reaction in reactions:
        part = [r for r in rows if r["reaction"] == reaction]
        for k, which in enumerate(STATES + ("both",)):
            statuses = [guess_status(r) if which == "both" else r[which] for r in part]
            left = 0.0
            for label, color in legend_entries(set(statuses), reasons):
                n = sum(paint(s, reasons)[0] == label for s in statuses)
                ax.barh(y, n / len(part), left=left, color=color, height=.78,
                        edgecolor="white", linewidth=.8)
                left += n / len(part)
            ok = sum(s == "ok" for s in statuses)
            ax.text(1.02, y, "%d/%d" % (ok, len(part)), va="center", fontsize=7.5, color=SOFT)
            ticks.append(y)
            labels.append(("R%d  " % reaction if k == 0 else "") + which)
            y += 1
        y += .6
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8.3, color=INK)
    ax.set_ylim(y - .6 - .5, -1.3)
    ax.text(1.02, -.95, "ok", fontsize=7.5, color=SOFT)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .5, 1])
    ax.set_xticklabels(["0", "50%", "100%"])
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    _style(ax, keep=("bottom",))


def _panel_bond_map(ax, rows, reasons, cutoff):
    """C: normalised breaking vs forming bond length of every endpoint;
    filled initial, open final."""
    from matplotlib.lines import Line2D

    points = [(r[s + "_break_norm"], r[s + "_form_norm"], r[s], s == "initial")
              for r in rows for s in STATES
              if r[s + "_break_norm"] is not None and r[s + "_form_norm"] is not None]
    hi = max([3.0] + [min(4.0, max(x, y) + .15) for x, y, _, _ in points])
    ax.axvspan(0.6, cutoff, color="#f2f2f2", zorder=0)
    ax.axhspan(0.6, cutoff, color="#f2f2f2", zorder=0)
    ax.axvline(cutoff, color=SOFT, lw=.8, ls="--", zorder=1)
    ax.axhline(cutoff, color=SOFT, lw=.8, ls="--", zorder=1)
    ax.text(cutoff + .03, hi - .05, "bond cutoff", fontsize=7.5, color=SOFT, rotation=90, va="top")
    ax.text(0.66, (cutoff + hi) / 2, "initial state expected here", fontsize=7.5, color=SOFT,
            rotation=90, ha="left", va="center")
    ax.text((cutoff + hi) / 2, 0.66, "final state expected here", fontsize=7.5, color=SOFT,
            ha="center", va="bottom")
    for x, y, status, filled in points:
        color = paint(status, reasons)[1]
        ax.scatter(min(x, hi - .02), min(y, hi - .02), s=26, zorder=3, linewidth=1.1,
                   facecolor=color if filled else "white", edgecolor=color, alpha=.9)
    ax.set_xlim(0.6, hi)
    ax.set_ylim(0.6, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("breaking bonds   r / (r$_i$ + r$_j$)")
    ax.set_ylabel("forming bonds   r / (r$_i$ + r$_j$)")
    ax.grid(color=GRID)
    ax.set_axisbelow(True)
    _style(ax)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mec=SOFT, mfc=SOFT if s == "initial"
                              else "white", label=s) for s in STATES],
              loc="upper right", frameon=False, fontsize=8)
    if not points:
        ax.text(.5, .5, "no finished endpoints yet", transform=ax.transAxes, ha="center",
                color=SOFT)


def _panel_energies(ax, rows, reactions):
    """D: survivors' initial and final energies relative to each reaction's
    lowest initial state; a line joins the two states of one guess."""
    from matplotlib.lines import Line2D

    rng = np.random.default_rng(0)            # fixed jitter, same figure every run
    any_survivor = False
    for column, reaction in enumerate(reactions):
        kept = [r for r in rows if r["reaction"] == reaction and r["survived"]]
        if not kept:
            ax.text(column, 0, "none\nsurvived", ha="center", va="bottom", fontsize=7.5,
                    color="#b5b5b5")
            continue
        any_survivor = True
        lowest = min(r["E_initial"] for r in kept)
        for r in kept:
            jitter = rng.uniform(-.04, .04)
            x = (column - .15 + jitter, column + .15 + jitter)
            y = (r["E_initial"] - lowest, r["E_final"] - lowest)
            ax.plot(x, y, color=GREEN, lw=.6, alpha=.45, zorder=2)
            ax.scatter(x[0], y[0], s=24, zorder=3, color=GREEN, linewidth=1.1)
            ax.scatter(x[1], y[1], s=24, zorder=3, facecolor="white", edgecolor=GREEN,
                       linewidth=1.1)
    ax.set_xticks(range(len(reactions)))
    ax.set_xticklabels(["R%d" % r for r in reactions], color=INK)
    ax.set_xlim(-.6, len(reactions) - .4)
    ax.set_ylabel("E − lowest initial state of that reaction   (eV)")
    ax.axhline(0, color=SOFT, lw=.6)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    _style(ax)
    if not any_survivor:
        ax.set_ylim(-.1, 1)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mec=GREEN,
                              mfc=GREEN if s == "initial" else "white", label=s) for s in STATES],
              loc="upper right", frameon=False, fontsize=8)


def plot_summary(rows, out_path, title, reasons=False, cutoff=1.25):
    """The summary figure (module docstring) for ``rows`` from :func:`collect`.
    ``title`` is the subtitle's run name; ``reasons`` colors by status instead
    of survived / lost; ``cutoff`` is the step 7 bond cutoff, drawn in C."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch

    if not rows:
        raise ValueError("nothing to plot: no step 7 guesses found")
    reactions = sorted({r["reaction"] for r in rows})
    pairs = sorted({r["pair_label"] for r in rows},
                   key=lambda p: [int(t[1:]) if t[1:].isdigit() else t for t in p.split(" · ")])
    column_labels = {r["reaction"]: "%s\n%s" % (r["reaction_label"], r["reaction_short"])
                     for r in rows}
    sites = {r["pair_label"]: r["sites"] for r in rows}
    row_labels = {p: ("%s\n%s" % (p, sites[p]) if sites[p] else p) for p in pairs}

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
                         "axes.edgecolor": "#cfcfcf", "axes.labelcolor": INK,
                         "xtick.color": SOFT, "ytick.color": SOFT,
                         "axes.titleweight": "bold", "axes.titlesize": 11,
                         "axes.titlelocation": "left", "axes.titlepad": 12})
    map_width = max(7.5, 1.45 * len(reactions))
    map_height = max(3.2, 0.62 * len(pairs) + 1.2)
    bars_height = 0.32 * len(reactions) * 3 + 1.5
    top_height = max(map_height, bars_height)
    header, gap = 2.55, 1.5
    width = map_width + 5.2
    height = header + top_height + gap + 5.6 + .75
    fig = plt.figure(figsize=(width, height), dpi=150, facecolor="white")
    gs = GridSpec(2, 2, figure=fig, width_ratios=[map_width, 5.2],
                  height_ratios=[top_height, 5.6], hspace=gap / ((top_height + 5.6) / 2),
                  wspace=.22, left=1.6 / width, right=1 - .5 / width,
                  top=1 - header / height, bottom=.75 / height)

    finished = sum(r[s] not in ("pending", "failed") for r in rows for s in STATES)
    fig.text(1.6 / width, 1 - .42 / height, "Initial / final state summary", fontsize=18,
             fontweight="bold", color=INK)
    fig.text(1.6 / width, 1 - .78 / height,
             "%s   ·   %d reactions   ·   %d O pairs   ·   %d guesses   ·   %d of %d endpoints "
             "finished   ·   %d guesses survived"
             % (title, len(reactions), len(pairs), len(rows), finished, 2 * len(rows),
                sum(r["survived"] for r in rows)), fontsize=10.5, color=SOFT)
    present = legend_entries({r[s] for r in rows for s in STATES}, reasons)
    fig.legend(handles=[Patch(fc=color, label=label) for label, color in present],
               loc="upper left", bbox_to_anchor=(1.6 / width - .004, 1 - 1.02 / height),
               ncol=len(present), frameon=False, fontsize=9.5, handlelength=1.2,
               columnspacing=1.6)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("A   Survivors per O pair and reaction", pad=52)
    _panel_map(ax, rows, reactions, pairs, column_labels, row_labels, reasons)
    ax.text(0, len(pairs) + .25,
            "Each cell: one O pair for one reaction.  Number: guesses with both endpoints ok / "
            "guesses.\nColor: %s"
            % ("the best outcome of any guess there (the worse of its two endpoints)."
               if reasons else "green if any guess there survived, red if none did."),
            ha="left", va="top", fontsize=8, color=SOFT)

    _panel_bars(fig.add_subplot(gs[0, 1]), rows, reactions, reasons)
    fig.axes[-1].set_title("B   Outcomes by reaction", pad=12)

    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("C   Where each endpoint ended")
    _panel_bond_map(ax, rows, reasons, cutoff)

    ax = fig.add_subplot(gs[1, 1])
    ax.set_title("D   Survivor energies within each reaction")
    _panel_energies(ax, rows, reactions)

    fig.savefig(out_path, facecolor="white")
    plt.close(fig)