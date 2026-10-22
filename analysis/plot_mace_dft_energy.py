#!/usr/bin/env python
"""Parity plots, MACE vs DFT, for every finished single point in DFT_sp:
relative energies and force components, one row per source (sella,
harmonic, neb), one color per round.

    python analysis/plot_mace_dft_energy.py runs/MOR_T4/DFT_sp/round3
    python analysis/plot_mace_dft_energy.py runs/MOR_T4/DFT_sp            # every round

The MACE labels stored in each config are those of the model that sampled
it (round 1: the foundation model, round 2: the round-1 fine-tuned model,
round 3: the round-2 one), so this shows how well that model does on data
it had not seen.

Energy (left column): relative, in eV per configuration -- the units of a
barrier. Energies only compare between structures with the same atoms and
under one model, so configs are grouped by round and composition, and in
each group the config with the lowest DFT energy is the reference:

    x = E_DFT - E_DFT(ref),   y = E_MACE - E_MACE(ref)     (the same ref config)

Per-element references (VASP's PAW atoms, MACE's E0s) cancel, so no shift
is needed, not even for the foundation model. The reference config's own
MACE error moves its whole group off the diagonal by a constant; the table's
"E spread" does not depend on that choice: the RMSE of E_MACE - E_DFT after
removing each group's median (median, so a few broken frames do not move
it), i.e. the scatter that ends up in a barrier.
Groups with a single config have nothing to compare with and are left out.
"abs/atom" is the plain per-atom RMSE, comparable with MACE's training log
(meaningful only for the fine-tuned rounds).

Forces (right column): components in eV/A; forces need no reference (a
constant energy offset has zero gradient). Atoms whose stored MACE force is
exactly zero are left out: they were frozen when the frame was written and
their MACE force was stored with the constraint applied, while the DFT
single point gives their real force. Training is not affected: it uses the
DFT forces of every atom.

neb rows: frames from not_converged bands are drawn hollow and get their own
table line -- if the large errors sit there, they are broken geometries from
bands that blew up; if converged bands are off too, it is the model's gap
in the middle of the path.

Prints one table and writes <folder>/mace_vs_dft_parity.png. Configs without
MACE labels (start-band images, harmonic trajectory frames) are counted but
cannot be plotted. Reading 1000+ OUTCARs takes a few minutes.
"""

import os
import sys
from collections import defaultdict

import _common  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ase.io import read  # noqa: E402

sys.path.insert(0, os.path.join(_common.REPO, "dft"))
from vasp_io import finished, read_dft, scf_converged  # noqa: E402

MAX_POINTS = 40000              # force components drawn per round and source (all are counted)
SOURCES = ("sella", "harmonic", "neb")   # row order; any other source comes after
BROKEN = ("not_converged",)     # neb band verdicts drawn hollow

flags = [a for a in sys.argv[1:] if a.startswith("--")]
if flags:
    print("ignoring %s: energies are relative now, no shift needed" % " ".join(flags))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
root = os.path.normpath(args[0] if args else "runs/MOR_T4/DFT_sp")
dft_sp = root
while os.path.basename(dft_sp) and os.path.basename(dft_sp) != "DFT_sp":
    dft_sp = os.path.dirname(dft_sp)
if not os.path.basename(dft_sp):
    sys.exit("%s is not inside a DFT_sp folder" % root)


def rmse(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else float("nan")


def mae(x):
    return float(np.mean(np.abs(x))) if len(x) else float("nan")


# 1. read every finished, MACE-labelled config
rows = []
n_unfinished = n_unlabelled = 0
for dirpath, dirnames, files in os.walk(root):
    dirnames[:] = sorted(d for d in dirnames if not d.startswith("training"))
    if "config.extxyz" not in files:
        continue
    if not (finished(dirpath) and scf_converged(dirpath)):
        n_unfinished += 1
        continue
    config = read(os.path.join(dirpath, "config.extxyz"))
    if "mace_energy" not in config.info:
        n_unlabelled += 1
        continue
    try:
        e_dft, f_dft = read_dft(dirpath, config)
    except Exception as error:                          # noqa: BLE001
        print("skipped %s: %s" % (dirpath, error))
        n_unfinished += 1
        continue
    parts = os.path.relpath(dirpath, dft_sp).split(os.sep)
    f_mace = config.arrays.get("mace_forces")
    rows.append({"round": parts[0], "source": parts[1],
                 "verdict": config.info.get("verdict"),
                 "formula": config.get_chemical_formula(), "n": len(config),
                 "e_dft": float(e_dft), "e_mace": float(config.info["mace_energy"]),
                 "f_dft": np.asarray(f_dft, dtype=float).reshape(-1, 3),
                 "f_mace": None if f_mace is None
                 else np.asarray(f_mace, dtype=float).reshape(-1, 3)})
if not rows:
    sys.exit("no finished, MACE-labelled configs under %s" % root)

# 2. relative energies within each (round, composition): one model, same atoms
by_model = defaultdict(list)
for r in rows:
    by_model[(r["round"], r["formula"])].append(r)
for members in by_model.values():
    ref = min(members, key=lambda r: r["e_dft"])
    typical = np.median([r["e_mace"] - r["e_dft"] for r in members])
    for r in members:
        r["x"] = r["e_dft"] - ref["e_dft"]
        r["y"] = r["e_mace"] - ref["e_mace"]
        r["spread"] = (r["e_mace"] - r["e_dft"]) - typical
        r["comparable"] = len(members) > 1


def force_components(members):
    """(DFT, MACE) components of every atom with a nonzero stored MACE force,
    and how many atoms were left out as frozen."""
    dft, mace, frozen = [], [], 0
    for r in members:
        if r["f_mace"] is None:
            continue
        keep = np.linalg.norm(r["f_mace"], axis=1) > 0
        frozen += int((~keep).sum())
        dft.append(r["f_dft"][keep])
        mace.append(r["f_mace"][keep])
    if not dft:
        return np.array([]), np.array([]), frozen
    return np.concatenate(dft).ravel(), np.concatenate(mace).ravel(), frozen


groups = defaultdict(list)
for r in rows:
    groups[(r["round"], r["source"])].append(r)
rounds = sorted({r["round"] for r in rows})
sources = [s for s in SOURCES if any(r["source"] == s for r in rows)]
sources += sorted({r["source"] for r in rows} - set(sources))


# 3. table
def table_line(label, members):
    comparable = [r for r in members if r["comparable"]]
    spread = 1000 * np.array([r["spread"] for r in comparable])
    per_atom = 1000 * np.array([(r["e_mace"] - r["e_dft"]) / r["n"] for r in members])
    f_dft, f_mace, frozen = force_components(members)
    df = f_mace - f_dft
    print("%-34s %6d %10.0f %10.0f %10.2f %9.3f %9.3f %8d"
          % (label, len(members), rmse(spread), mae(spread), rmse(per_atom),
             rmse(df), mae(df), frozen))


print("%-34s %6s %10s %10s %10s %9s %9s %8s"
      % ("group", "n", "E spread", "E spread", "abs/atom", "F RMSE", "F MAE", "frozen"))
print("%-34s %6s %10s %10s %10s %9s %9s %8s"
      % ("", "", "RMSE meV", "MAE meV", "RMSE meV", "eV/A", "eV/A", "atoms"))
for rnd in rounds:
    for source in sources:
        members = groups.get((rnd, source))
        if not members:
            continue
        table_line("%s/%s" % (rnd, source), members)
        if source == "neb":
            broken = [r for r in members if r["verdict"] in BROKEN]
            rest = [r for r in members if r["verdict"] not in BROKEN]
            if broken and rest:
                table_line("  converged bands", rest)
                table_line("  not_converged bands", broken)
print("(E spread: per round and composition, median error removed -- what enters a barrier)")
if n_unfinished:
    print("(%d configs not finished / not converged, left out)" % n_unfinished)
if n_unlabelled:
    print("(%d configs without MACE labels, left out)" % n_unlabelled)

# 4. figure: one row per source, relative energy | force components
colors = {rnd: plt.cm.tab10(k % 10) for k, rnd in enumerate(rounds)}
fig, axes = plt.subplots(len(sources), 2, figsize=(11.5, 5.0 * len(sources)), squeeze=False)
rng = np.random.default_rng(0)
for (energy_ax, force_ax), source in zip(axes, sources):
    for rnd in rounds:
        members = groups.get((rnd, source), [])
        if not members:
            continue
        color = colors[rnd]
        comparable = [r for r in members if r["comparable"]]
        if comparable:
            x = np.array([r["x"] for r in comparable])
            y = np.array([r["y"] for r in comparable])
            hollow = np.array([source == "neb" and r["verdict"] in BROKEN for r in comparable])
            spread = 1000 * np.array([r["spread"] for r in comparable])
            energy_ax.scatter(x[~hollow], y[~hollow], s=14, alpha=0.65, color=color,
                              label="%s  spread RMSE %.0f / MAE %.0f meV"
                              % (rnd, rmse(spread), mae(spread)))
            if hollow.any():
                energy_ax.scatter(x[hollow], y[hollow], s=18, facecolors="none",
                                  edgecolors=color, linewidths=0.9,
                                  label="%s  not_converged bands (%d)" % (rnd, hollow.sum()))
        f_dft, f_mace, frozen = force_components(members)
        if len(f_dft):
            df = f_mace - f_dft
            pick = (rng.choice(len(f_dft), MAX_POINTS, replace=False)
                    if len(f_dft) > MAX_POINTS else slice(None))
            force_ax.scatter(f_dft[pick], f_mace[pick], s=2, alpha=0.3, color=color,
                             rasterized=True,
                             label="%s  RMSE %.3f / MAE %.3f eV/A" % (rnd, rmse(df), mae(df)))
    for ax, title, unit in ((energy_ax, "%s: relative energy" % source, "eV"),
                            (force_ax, "%s: force components, free atoms" % source, "eV/A")):
        if not ax.collections:
            ax.set_visible(False)
            continue
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("DFT  [%s]" % unit)
        ax.set_ylabel("MACE  [%s]" % unit)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8, markerscale=2, loc="upper left")
fig.tight_layout()
out = os.path.join(root, "mace_vs_dft_parity.png")
fig.savefig(out, dpi=150)
print("\nwrote %s" % out)