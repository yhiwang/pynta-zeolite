#!/usr/bin/env python
"""Parity plots, MACE vs DFT, for every finished single point in DFT_sp.

    python analysis/plot_mace_dft_energy.py runs/MOR_T4/DFT_sp/round2
    python analysis/plot_mace_dft_energy.py runs/MOR_T4/DFT_sp            # every round
    python analysis/plot_mace_dft_energy.py runs/MOR_T4/DFT_sp/round1 --shift

The MACE labels stored in each config are those of the model that made it
(round 1: the foundation model, round 2: the round-1 fine-tuned model), so
this shows how well that model fits data it had not seen. One color per
<round>/<source> group (round2/sella, round2/harmonic, round2/neb, ...).

Left:  energy per atom, DFT (x) vs MACE (y). Per-atom is MACE's own
       convention, so the RMSE / MAE in meV/atom compare with its training log.
Right: force components, DFT (x) vs MACE (y), RMSE / MAE in eV/A.

--shift  removes, per group and per composition, the median of E_DFT - E_MACE
         before plotting. Needed for the foundation model (round 1), whose
         energies sit a constant away from your DFT (different functional,
         no D3); not needed for a model fine-tuned on these DFT labels.

Prints one table and writes <folder>/mace_vs_dft_parity.png. Configs
without MACE labels (start-band images, harmonic trajectory frames) are
counted but cannot be plotted. Reading 1000+ OUTCARs takes a few minutes.
"""

import os
import sys
from collections import defaultdict

import _common  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

sys.path.insert(0, os.path.join(_common.REPO, "dft"))
from vasp_io import finished, read_dft, scf_converged  # noqa: E402

MAX_POINTS = 40000          # force components drawn per group (all are counted)

args = [a for a in sys.argv[1:] if not a.startswith("--")]
shift = "--shift" in sys.argv
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


# 1. read every finished config
groups = defaultdict(list)
n_unfinished = n_unlabelled = 0
for dirpath, dirnames, files in os.walk(root):
    dirnames[:] = [d for d in dirnames if not d.startswith("training")]
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
    groups["%s/%s" % (parts[0], parts[1])].append({
        "formula": config.get_chemical_formula(),
        "n": len(config),
        "e_dft": e_dft,
        "e_mace": float(config.info["mace_energy"]),
        "f_dft": f_dft.ravel(),
        "f_mace": config.arrays["mace_forces"].ravel()
        if "mace_forces" in config.arrays else None,
    })
if not groups:
    sys.exit("no finished, MACE-labelled configs under %s" % root)

# 2. per group: energies per atom (shifted if asked) and force components
results = {}
for name, rows in sorted(groups.items()):
    offsets = defaultdict(list)
    for r in rows:
        offsets[r["formula"]].append(r["e_dft"] - r["e_mace"])
    median = {f: np.median(v) for f, v in offsets.items()}
    e_dft = np.array([r["e_dft"] / r["n"] for r in rows])
    e_mace = np.array([(r["e_mace"] + (median[r["formula"]] if shift else 0)) / r["n"]
                       for r in rows])
    with_f = [r for r in rows if r["f_mace"] is not None]
    f_dft = np.concatenate([r["f_dft"] for r in with_f]) if with_f else np.array([])
    f_mace = np.concatenate([r["f_mace"] for r in with_f]) if with_f else np.array([])
    results[name] = dict(n=len(rows), e_dft=e_dft, e_mace=e_mace, f_dft=f_dft, f_mace=f_mace,
                         offset=np.median([r["e_dft"] - r["e_mace"] for r in rows]))

# 3. table
print("%-18s %6s %9s %10s %10s %10s %10s"
      % ("group", "n", "offset", "E RMSE", "E MAE", "F RMSE", "F MAE"))
print("%-18s %6s %9s %10s %10s %10s %10s"
      % ("", "", "eV", "meV/atom", "meV/atom", "eV/A", "eV/A"))
for name, r in results.items():
    de = 1000 * (r["e_mace"] - r["e_dft"])
    df = r["f_mace"] - r["f_dft"]
    print("%-18s %6d %9.2f %10.2f %10.2f %10.3f %10.3f"
          % (name, r["n"], r["offset"], rmse(de), mae(de), rmse(df), mae(df)))
if shift:
    print("(energies shifted by the per-composition median offset)")
if n_unfinished:
    print("(%d configs not finished / not converged, left out)" % n_unfinished)
if n_unlabelled:
    print("(%d configs without MACE labels, left out)" % n_unlabelled)

# 4. figure
fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.5))
rng = np.random.default_rng(0)
for name, r in results.items():
    de = 1000 * (r["e_mace"] - r["e_dft"])
    left.scatter(r["e_dft"], r["e_mace"], s=10, alpha=0.6,
                 label="%s  RMSE %.1f / MAE %.1f meV/atom" % (name, rmse(de), mae(de)))
    if len(r["f_dft"]):
        df = r["f_mace"] - r["f_dft"]
        pick = (rng.choice(len(r["f_dft"]), MAX_POINTS, replace=False)
                if len(r["f_dft"]) > MAX_POINTS else slice(None))
        right.scatter(r["f_dft"][pick], r["f_mace"][pick], s=2, alpha=0.3, rasterized=True,
                      label="%s  RMSE %.3f / MAE %.3f eV/A" % (name, rmse(df), mae(df)))
for ax, title, unit in ((left, "energy per atom", "eV/atom"), (right, "force components", "eV/A")):
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("DFT  [%s]" % unit)
    ax.set_ylabel("MACE  [%s]" % unit)
    ax.set_title(title + ("  (shifted)" if shift and ax is left else ""))
    ax.legend(frameon=False, fontsize=8, markerscale=3)
fig.tight_layout()
out = os.path.join(root, "mace_vs_dft_parity.png")
fig.savefig(out, dpi=150)
print("\nwrote %s" % out)