"""One table and one figure for every Sella run of a run directory
(used by analysis/summarize_ts.py).

    collect        walk TS_sella/ and return one row per (guess, start)
    write_csv      the rows as a CSV, fixed column order
    plot_summary   the figure:
                     A  best outcome per O pair and reaction, one half-cell
                        per start, "TS found / guesses searched"
                     B  verdict shares per reaction and start
                     C  where each search ended: normalised breaking vs
                        forming bond length
                     D  TS energies within each reaction, relative to that
                        reaction's lowest TS

Everything is read from the files steps 5-8 leave behind (info.json,
ts_bonds.json, start.xyz, ts_sella.xyz, result.json), so nothing depends on
the reaction, framework or site rule. A start whose job has not written
result.json yet is a row with verdict "pending" ("failed" when its job.err
holds a Python traceback).

Bond lengths are normalised by the sum of the ASE covalent radii,
r / (r_i + r_j): 1 is bonded, ~1.35 the step 7 spring target, >2 apart.
"""

import csv
import json
import os
import textwrap
from collections import Counter, defaultdict

import numpy as np
from ase.data import covalent_radii
from ase.io import read

from pyntaz.geometry import positional_rmsd

VERDICTS = ("ts", "wrong_mode", "higher_order", "minimum", "not_converged", "failed", "pending")
LABELS = {"ts": "TS", "wrong_mode": "wrong mode", "higher_order": "higher order",
          "minimum": "minimum", "not_converged": "not converged", "failed": "job failed",
          "pending": "no result yet"}
COLORS = {"ts": "#1b9e77", "wrong_mode": "#e6ab02", "higher_order": "#d95f02",
          "minimum": "#7570b3", "not_converged": "#bdbdbd", "failed": "#636363",
          "pending": "#e3e3e3"}
RANK = {verdict: rank for rank, verdict in enumerate(VERDICTS)}   # lower is better
START_ORDER = ("harmonic", "raw")

COLUMNS = ["reaction", "reaction_name", "reaction_family", "pair", "sites", "O_a", "O_b",
           "flip", "stem", "start", "verdict", "E_start", "E_end", "dE", "steps", "fmax",
           "n_imag", "imag1_cm", "mode_max_stretch", "break_norm_start", "break_norm_end",
           "form_norm_start", "form_norm_end", "rmsd_to_other_start", "bonds"]

# file names inside TS_sella/; summarize_ts.py passes scripts/layout.py's
FILES = {"info": "info.json", "bonds": "ts_bonds.json", "start": "start.xyz",
         "end": "ts_sella.xyz", "result": "result.json", "err": "job.err"}

INK, SOFT, GRID, EMPTY = "#262626", "#6b6b6b", "#e9e9e9", "#f6f6f6"


# --------------------------------------------------------------------------
# collecting
# --------------------------------------------------------------------------

def _subdirs(path):
    return sorted(name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)))


def _reaction_index(name):
    head = name.split("_")[0]
    return int(head) if head.isdigit() else None


def _family_label(family, reaction):
    """Short column label: the family without 'Surface_', else the reaction."""
    text = family.replace("Surface_", "").replace("_", " ") if family else reaction
    if not family and len(text) > 36:
        text = text[:35] + "..."
    return "\n".join(textwrap.wrap(text, 13, break_long_words=False)[:3])


def _job_state(run_dir, files):
    """'failed' if the job's stderr holds a Python traceback, else 'pending'."""
    err = os.path.join(run_dir, files["err"])
    if os.path.isfile(err):
        with open(err, errors="replace") as handle:
            if "Traceback" in handle.read():
                return "failed"
    return "pending"


def _mean_norm(bonds, key, change, radius):
    values = [bond[key] / (radius[bond["indices"][0]] + radius[bond["indices"][1]])
              for bond in bonds
              if bond["name"].split()[0] == change and bond.get(key) is not None]
    return float(np.mean(values)) if values else None


def collect(ts_sella_dir, files=None):
    """[row dict] for every TS_sella/<i>_rxn/pair_<k>/<stem>/<start>/ folder,
    sorted by reaction, pair, stem, start. Keys are :data:`COLUMNS` plus
    ``reaction_label`` ("R0") and ``pair_label`` ("O71 · O73") for plotting."""
    files = dict(FILES, **(files or {}))
    rows = []
    ends = {}                         # (reaction, pair, stem) -> {start: (atoms, n_framework)}
    for reaction_name in _subdirs(ts_sella_dir):
        index = _reaction_index(reaction_name)
        if index is None:
            continue
        reaction_dir = os.path.join(ts_sella_dir, reaction_name)
        info_path = os.path.join(reaction_dir, files["info"])
        info = {}
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
                stem_dir = os.path.join(reaction_dir, pair, stem)
                for start in _subdirs(stem_dir):
                    run_dir = os.path.join(stem_dir, start)
                    if not os.path.isfile(os.path.join(run_dir, files["bonds"])):
                        continue
                    row = {column: None for column in COLUMNS}
                    row.update({"reaction": index, "reaction_name": reaction,
                                "reaction_family": family, "pair": pair,
                                "sites": " / ".join(sites),
                                "O_a": oxygens[0] if oxygens else None,
                                "O_b": oxygens[1] if len(oxygens) > 1 else None,
                                "flip": int(stem[4]) if stem.startswith("flip") else None,
                                "stem": stem, "start": start,
                                "reaction_label": "R%d" % index,
                                "reaction_short": _family_label(family, reaction),
                                "pair_label": " · ".join("O%d" % o for o in oxygens) or pair})
                    result_path = os.path.join(run_dir, files["result"])
                    if not os.path.isfile(result_path):
                        row["verdict"] = _job_state(run_dir, files)
                        rows.append(row)
                        continue

                    with open(result_path) as handle:
                        result = json.load(handle)
                    start_atoms = read(os.path.join(run_dir, files["start"]))
                    radius = covalent_radii[start_atoms.get_atomic_numbers()]
                    bonds = result["bonds"]
                    imaginary = result.get("imaginary") or []
                    stretches = [abs(b["stretch"]) for b in bonds if b.get("stretch") is not None]
                    row.update({
                        "verdict": result["verdict"],
                        "E_start": result["energy_start"], "E_end": result["energy"],
                        "dE": result["energy"] - result["energy_start"],
                        "steps": result["n_steps"], "fmax": result["fmax"],
                        "n_imag": len(imaginary) if result["converged"] else None,
                        "imag1_cm": imaginary[0] if imaginary else None,
                        "mode_max_stretch": max(stretches) if stretches else None,
                        "break_norm_start": _mean_norm(bonds, "start", "break", radius),
                        "break_norm_end": _mean_norm(bonds, "end", "break", radius),
                        "form_norm_start": _mean_norm(bonds, "start", "form", radius),
                        "form_norm_end": _mean_norm(bonds, "end", "form", radius),
                        "bonds": " | ".join("%s %.3f>%.3f" % (b["name"], b["start"], b["end"])
                                            for b in bonds)})
                    end_path = os.path.join(run_dir, files["end"])
                    if os.path.isfile(end_path):
                        with open(os.path.join(run_dir, files["bonds"])) as handle:
                            n_framework = json.load(handle)["n_framework"]
                        ends.setdefault((index, pair, stem), {})[start] = (read(end_path), n_framework)
                    rows.append(row)

    for row in rows:                  # adsorbate RMSD between the two starts' saddles
        found = ends.get((row["reaction"], row["pair"], row["stem"]), {})
        others = [s for s in found if s != row["start"]]
        if row["start"] in found and others:
            atoms, n_framework = found[row["start"]]
            other, _ = found[others[0]]
            if len(other) == len(atoms):
                row["rmsd_to_other_start"] = positional_rmsd(
                    atoms, other, list(range(n_framework, len(atoms))))
    return rows


def write_csv(rows, path):
    """``rows`` as a CSV with :data:`COLUMNS`; floats rounded for reading."""
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (round(value, 4) if isinstance(value, float) else value)
                             for key, value in row.items()})


def verdict_counts(rows):
    """{(reaction, start): Counter(verdict)} for printing."""
    counts = defaultdict(Counter)
    for row in rows:
        counts[(row["reaction"], row["start"])][row["verdict"]] += 1
    return counts


# --------------------------------------------------------------------------
# the figure
# --------------------------------------------------------------------------

def _style(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
    ax.tick_params(length=0)


def _panel_map(ax, rows, reactions, pairs, starts, column_labels, row_labels):
    """A: pair x reaction, one sub-cell per start, best verdict + TS / guesses.
    ``column_labels`` {reaction: text}, ``row_labels`` {pair label: text}."""
    from matplotlib.patches import Rectangle

    width = 1.0 / len(starts)
    for column, reaction in enumerate(reactions):
        for line, pair_label in enumerate(pairs):
            cell = [r for r in rows if r["reaction"] == reaction and r["pair_label"] == pair_label]
            if not cell:
                ax.add_patch(Rectangle((column + .02, line + .05), .96, .9, fc=EMPTY, ec="none"))
                ax.text(column + .5, line + .5, "no guess", ha="center", va="center",
                        fontsize=7.5, color="#b5b5b5")
                continue
            for k, start in enumerate(starts):
                part = [r for r in cell if r["start"] == start]
                x0 = column + k * width
                if not part:
                    ax.add_patch(Rectangle((x0 + .02, line + .05), width - .04, .9, fc=EMPTY, ec="none"))
                    continue
                best = min(part, key=lambda r: RANK[r["verdict"]])["verdict"]
                ax.add_patch(Rectangle((x0 + .02, line + .05), width - .04, .9,
                                       fc=COLORS[best], ec="none"))
                n_ts = sum(r["verdict"] == "ts" for r in part)
                dark = best in ("ts", "higher_order", "minimum", "failed")
                ax.text(x0 + width / 2, line + .5, "%d/%d" % (n_ts, len(part)), ha="center",
                        va="center", fontsize=8.5, fontweight="bold",
                        color="white" if dark else INK)
    ax.set_xlim(0, len(reactions))
    ax.set_ylim(len(pairs), 0)
    ax.set_xticks(np.arange(len(reactions)) + .5)
    ax.set_xticklabels([column_labels[r] for r in reactions], fontsize=8.8, color=INK)
    ax.set_yticks(np.arange(len(pairs)) + .5)
    ax.set_yticklabels([row_labels[p] for p in pairs], fontsize=8.8, color=INK)
    ax.xaxis.tick_top()
    _style(ax, keep=())


def _panel_bars(ax, rows, reactions, starts):
    """B: verdict shares, one bar per (reaction, start)."""
    ticks, labels, y = [], [], 0.0
    for reaction in reactions:
        for k, start in enumerate(starts):
            part = [r for r in rows if r["reaction"] == reaction and r["start"] == start]
            left = 0.0
            for verdict in VERDICTS:
                n = sum(r["verdict"] == verdict for r in part)
                if n:
                    ax.barh(y, n / len(part), left=left, color=COLORS[verdict], height=.78,
                            edgecolor="white", linewidth=.8)
                    left += n / len(part)
            ax.text(1.02, y, "%d" % len(part), va="center", fontsize=7.5, color=SOFT)
            ticks.append(y)
            labels.append(("R%d  " % reaction if k == 0 else "") + start)
            y += 1
        y += .6
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8.3, color=INK)
    ax.set_ylim(y - .6 - .5, -1.3)
    ax.text(1.02, -.95, "runs", fontsize=7.5, color=SOFT)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .5, 1])
    ax.set_xticklabels(["0", "50%", "100%"])
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    _style(ax, keep=("bottom",))


def _panel_bond_map(ax, rows, starts):
    """C: normalised breaking vs forming bond length where each search ended."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    done = [r for r in rows if r["break_norm_end"] is not None and r["form_norm_end"] is not None]
    hi = max([3.0] + [min(4.0, max(r["break_norm_end"], r["form_norm_end"]) + .15) for r in done])
    ax.axvspan(0.75, 1.15, color="#f2f2f2", zorder=0)
    ax.axhspan(0.75, 1.15, color="#f2f2f2", zorder=0)
    ax.add_patch(Rectangle((1.2, 1.2), .35, .35, fc="none", ec=COLORS["ts"], ls="--", lw=1.2, zorder=1))
    ax.text(1.57, 1.2, " TS region", ha="left", va="bottom", fontsize=8, color=COLORS["ts"])
    ax.text(0.78, hi - .05, "breaking bonds intact", fontsize=7.5, color=SOFT, rotation=90, va="top")
    ax.text(hi - .05, 0.78, "forming bonds made", fontsize=7.5, color=SOFT, ha="right", va="bottom")
    for row in done:
        filled = row["start"] == starts[0]
        ax.scatter(min(row["break_norm_end"], hi - .02), min(row["form_norm_end"], hi - .02),
                   s=26, zorder=3, facecolor=COLORS[row["verdict"]] if filled else "white",
                   edgecolor=COLORS[row["verdict"]], linewidth=1.1, alpha=.9)
    ax.set_xlim(0.75, hi)
    ax.set_ylim(0.75, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("breaking bonds   r / (r$_i$ + r$_j$)")
    ax.set_ylabel("forming bonds   r / (r$_i$ + r$_j$)")
    ax.grid(color=GRID)
    ax.set_axisbelow(True)
    _style(ax)
    handles = [Line2D([], [], marker="o", ls="", mfc=SOFT if k == 0 else "white", mec=SOFT,
                      label="%s start" % start) for k, start in enumerate(starts)]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8)
    if not done:
        ax.text(.5, .5, "no finished runs yet", transform=ax.transAxes, ha="center",
                color=SOFT)


def _panel_energies(ax, rows, reactions, starts):
    """D: TS energies relative to each reaction's lowest TS."""
    from matplotlib.lines import Line2D

    rng = np.random.default_rng(0)            # fixed jitter, same figure every run
    offsets = np.linspace(-.14, .14, len(starts)) if len(starts) > 1 else [0.0]
    any_ts = False
    for column, reaction in enumerate(reactions):
        ts = [r for r in rows if r["reaction"] == reaction and r["verdict"] == "ts"]
        if not ts:
            ax.text(column, 0, "no TS", ha="center", va="bottom", fontsize=7.5, color="#b5b5b5")
            continue
        any_ts = True
        lowest = min(r["E_end"] for r in ts)
        for k, start in enumerate(starts):
            energies = [r["E_end"] - lowest for r in ts if r["start"] == start]
            x = column + offsets[k] + rng.uniform(-.05, .05, len(energies))
            ax.scatter(x, energies, s=24, zorder=3, linewidth=1.1, edgecolor=COLORS["ts"],
                       facecolor=COLORS["ts"] if k == 0 else "white")
    ax.set_xticks(range(len(reactions)))
    ax.set_xticklabels(["R%d" % r for r in reactions], color=INK)
    ax.set_xlim(-.6, len(reactions) - .4)
    ax.set_ylabel("E − lowest TS of that reaction   (eV)")
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    _style(ax)
    if not any_ts:
        ax.set_ylim(-.1, 1)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mec=COLORS["ts"],
                              mfc=COLORS["ts"] if k == 0 else "white", label="%s start" % s)
                       for k, s in enumerate(starts)],
              loc="upper right", frameon=False, fontsize=8)


def plot_summary(rows, out_path, title):
    """The summary figure (module docstring) for ``rows`` from :func:`collect`.
    ``title`` is the subtitle's run name. Sized to the number of reactions
    and O pairs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch

    if not rows:
        raise ValueError("nothing to plot: no TS_sella runs found")
    reactions = sorted({r["reaction"] for r in rows})
    starts = [s for s in START_ORDER if any(r["start"] == s for r in rows)]
    starts += sorted({r["start"] for r in rows} - set(starts))
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
    bars_height = 0.32 * len(reactions) * len(starts) + 1.5
    top_height = max(map_height, bars_height)
    header = 2.55                     # in: title, subtitle, key, room for panel A's labels
    gap = 1.5                         # in: panel A's note + panel titles of the bottom row
    width = map_width + 5.2
    height = header + top_height + gap + 5.6 + .75
    fig = plt.figure(figsize=(width, height), dpi=150, facecolor="white")
    gs = GridSpec(2, 2, figure=fig, width_ratios=[map_width, 5.2],
                  height_ratios=[top_height, 5.6], hspace=gap / ((top_height + 5.6) / 2),
                  wspace=.22, left=1.6 / width, right=1 - .5 / width,
                  top=1 - header / height, bottom=.75 / height)

    finished = [r for r in rows if r["verdict"] not in ("pending", "failed")]
    guesses = {(r["reaction"], r["pair"], r["stem"]) for r in rows}
    fig.text(1.6 / width, 1 - .42 / height, "TS search summary", fontsize=18,
             fontweight="bold", color=INK)
    fig.text(1.6 / width, 1 - .78 / height,
             "%s   ·   %d reactions   ·   %d O pairs   ·   %d guesses   ·   %d of %d Sella runs "
             "finished   ·   %d TS found"
             % (title, len(reactions), len(pairs), len(guesses), len(finished), len(rows),
                sum(r["verdict"] == "ts" for r in rows)), fontsize=10.5, color=SOFT)
    present = [v for v in VERDICTS if any(r["verdict"] == v for r in rows)]
    fig.legend(handles=[Patch(fc=COLORS[v], label=LABELS[v]) for v in present],
               loc="upper left", bbox_to_anchor=(1.6 / width - .004, 1 - 1.02 / height),
               ncol=len(present), frameon=False, fontsize=9.5, handlelength=1.2,
               columnspacing=1.6)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("A   Best outcome per O pair and reaction", pad=52)
    _panel_map(ax, rows, reactions, pairs, starts, column_labels, row_labels)
    ax.text(0, len(pairs) + .25,
            "Each cell: one O pair for one reaction, split by start (%s).  Number: guesses that "
            "ended as a TS / guesses searched.\nColor: the best outcome any of those guesses "
            "reached (ranked as in the key, TS best)."
            % " | ".join(starts), ha="left", va="top", fontsize=8, color=SOFT)

    _panel_bars(fig.add_subplot(gs[0, 1]), rows, reactions, starts)
    fig.axes[-1].set_title("B   Verdicts by reaction and start", pad=12)

    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("C   Where each search ended")
    _panel_bond_map(ax, rows, starts)

    ax = fig.add_subplot(gs[1, 1])
    ax.set_title("D   TS energies within each reaction")
    _panel_energies(ax, rows, reactions, starts)

    fig.savefig(out_path, facecolor="white")
    plt.close(fig)