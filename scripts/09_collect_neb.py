#!/usr/bin/env python
"""Step 9 -- collect the initial state, TS and final state of every guess
that has all three into NEB/, and list the activation energies they give.

A guess is collected only when
    initial   TS_harmonic/.../initial/endpoint.json: verdict ok, free relax converged
    final     TS_harmonic/.../final/endpoint.json:   verdict ok, free relax converged
    TS        TS_sella/.../<start>/result.json: verdict "ts" for at least one start;
              with both starts a TS, the lower-energy one is taken

    <RUN_DIR>/NEB/<i>_rxn/pair_<k>/<stem>/initial.xyz   from TS_harmonic/.../initial/initial.xyz
    <RUN_DIR>/NEB/<i>_rxn/pair_<k>/<stem>/ts.xyz        from TS_sella/.../<start>/ts_sella.xyz
    <RUN_DIR>/NEB/<i>_rxn/pair_<k>/<stem>/final.xyz     from TS_harmonic/.../final/final.xyz
    <RUN_DIR>/NEB/<i>_rxn/pair_<k>/<stem>/neb.json      sources, energies, TS start,
                                                        n_framework, seated oxygens, bonds
    <RUN_DIR>/NEB/barriers.csv   one row per collected guess
    <RUN_DIR>/NEB/barriers.png   forward and reverse barriers per reaction

All three structures come from the same step 5 guess, so the atom order
matches, as NEB needs. Energies are MACE (the model of steps 7 and 8):
Ea_fwd = E_TS - E_initial, Ea_rev = E_TS - E_final, dE_rxn = E_final -
E_initial. They are guesses until NEB / DFT confirm the TS.

Then the starting band for step 10, in the same folder:

    <RUN_DIR>/NEB/<i>_rxn/pair_<k>/<stem>/band.traj   IS, NEB_IMAGES_BEFORE images, TS,
                                                       NEB_IMAGES_AFTER images, FS
    <RUN_DIR>/NEB/<i>_rxn/pair_<k>/<stem>/band.json   the settings it was built with,
                                                       the TS index, the free atoms

Two half-bands, IS -> TS and TS -> FS (NEB_INTERPOLATION, minimum image),
so the band starts on the Sella saddle. The free atoms are the adsorbate
and the framework within NEB_FREE_RADIUS of it in any of the three states
(each state relaxed its own region, so the NEB frees the union); every
other atom carries the same FixAtoms in every image (pyntaz/neb.py).

A guess folder that already has neb.json keeps its three structures, but
its energies still go into the table; delete the folder to collect it
again. Its band is rebuilt whenever the four NEB band settings change, as
long as the NEB has not been submitted (no job.id, no neb_result.json);
a submitted band is never touched. Submits nothing; runs on the login node
(the IDPP interpolation takes a few seconds per guess).
"""

import csv
import json
import os
import shutil
import sys

import _common  # noqa: F401
import layout
import settings
from ase.io import read, write
from pyntaz.neb import build_band

BARRIERS_CSV = "barriers.csv"
BARRIERS_PNG = "barriers.png"
COLUMNS = ["reaction", "reaction_name", "pair", "stem", "ts_start", "E_initial", "E_ts",
           "E_final", "Ea_fwd", "Ea_rev", "dE_rxn", "imag_cm"]


def subdirs(path):
    if not os.path.isdir(path):
        return []
    return sorted(n for n in os.listdir(path) if os.path.isdir(os.path.join(path, n)))


def load(path):
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def endpoint(guess_dir, state):
    """(endpoint.json dict, structure path) of a valid endpoint, else None."""
    job_dir = layout.harmonic_job_dir(guess_dir, state)
    result = load(os.path.join(job_dir, layout.ENDPOINT_RESULT))
    structure = os.path.join(job_dir, layout.harmonic_files(state)[0])
    if result is None or not os.path.isfile(structure):
        return None
    if result["verdict"] != "ok" or not result.get("free_converged"):
        return None
    return result, structure


def best_ts(sella_dir):
    """(start, result.json dict, ts_sella.xyz path) of the lowest-energy start
    whose verdict is "ts", else None."""
    found = []
    for start in subdirs(sella_dir):
        run_dir = os.path.join(sella_dir, start)
        result = load(os.path.join(run_dir, layout.TS_SELLA_RESULT))
        structure = os.path.join(run_dir, layout.TS_SELLA_STRUCTURE)
        if result and result["verdict"] == "ts" and os.path.isfile(structure):
            found.append((result["energy"], start, result, structure))
    if not found:
        return None
    _, start, result, structure = min(found, key=lambda item: item[0])
    return start, result, structure


def collect(target, initial_path, ts_path, final_path, record, relative, index,
            reaction_name, start, energies, imaginary):
    """Copy the three structures into ``target`` and write neb.json."""
    os.makedirs(target, exist_ok=True)
    shutil.copy2(initial_path, os.path.join(target, layout.NEB_STATES["initial"]))
    shutil.copy2(ts_path, os.path.join(target, layout.NEB_STATES["ts"]))
    shutil.copy2(final_path, os.path.join(target, layout.NEB_STATES["final"]))
    e_is, e_ts, e_fs = energies
    with open(os.path.join(target, layout.NEB_RECORD), "w") as handle:
        json.dump({"guess": relative, "reaction": reaction_name,
                   "index": index, "n_framework": record["n_framework"],
                   "oxygens": record["oxygens"], "bonds": record["bonds"],
                   "ts_start": start,
                   "sources": {"initial": os.path.relpath(initial_path, target),
                               "ts": os.path.relpath(ts_path, target),
                               "final": os.path.relpath(final_path, target)},
                   "energies": {"initial": e_is, "ts": e_ts, "final": e_fs},
                   "Ea_fwd": round(e_ts - e_is, 4), "Ea_rev": round(e_ts - e_fs, 4),
                   "dE_rxn": round(e_fs - e_is, 4), "imaginary": imaginary},
                  handle, indent=2)


def band_settings():
    """The settings a band is built with; a band built with others is stale."""
    return {"n_before": settings.NEB_IMAGES_BEFORE, "n_after": settings.NEB_IMAGES_AFTER,
            "interpolation": settings.NEB_INTERPOLATION,
            "free_radius": float(settings.NEB_FREE_RADIUS)}


def make_band(target, n_framework):
    """Write band.traj / band.json into ``target`` unless they are up to
    date or the NEB has started. Returns "built", "up to date" or "started"."""
    if any(os.path.exists(os.path.join(target, name))
           for name in (layout.JOB_ID, layout.NEB_RESULT)):
        return "started"
    wanted = band_settings()
    band_path = os.path.join(target, layout.NEB_BAND)
    info = load(os.path.join(target, layout.NEB_BAND_INFO))
    if info and os.path.isfile(band_path) and {k: info.get(k) for k in wanted} == wanted:
        return "up to date"
    initial, ts, final = (read(os.path.join(target, layout.NEB_STATES[state]))
                          for state in ("initial", "ts", "final"))
    images, free = build_band(initial, ts, final, n_framework,
                              n_before=wanted["n_before"], n_after=wanted["n_after"],
                              method=wanted["interpolation"],
                              free_radius=wanted["free_radius"])
    write(band_path, images)
    with open(os.path.join(target, layout.NEB_BAND_INFO), "w") as handle:
        json.dump(dict(wanted, n_images=len(images), ts_index=wanted["n_before"] + 1,
                       n_framework=n_framework, n_free=len(free),
                       n_free_framework=sum(1 for i in free if i < n_framework),
                       free=free), handle, indent=2)
    return "built"


def plot_barriers(rows, path, title):
    """Forward (filled) and reverse (open) barrier of every collected guess,
    one column per reaction; the lowest forward barrier is labelled."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    reactions = sorted({r["reaction"] for r in rows})
    names = {r["reaction"]: r["reaction_name"] for r in rows}
    rng = np.random.default_rng(0)                  # fixed jitter, same figure every run
    fig, ax = plt.subplots(figsize=(max(6.5, 1.3 * len(reactions) + 2), 5.2), dpi=150)
    for column, reaction in enumerate(reactions):
        part = [r for r in rows if r["reaction"] == reaction]
        jitter = rng.uniform(-.05, .05, len(part))
        ax.scatter(column - .13 + jitter, [r["Ea_fwd"] for r in part], s=28, color="#1b9e77",
                   zorder=3, label="forward  E$_{TS}$ − E$_{IS}$" if column == 0 else None)
        ax.scatter(column + .13 + jitter, [r["Ea_rev"] for r in part], s=28, facecolor="white",
                   edgecolor="#1b9e77", linewidth=1.1, zorder=3,
                   label="reverse  E$_{TS}$ − E$_{FS}$" if column == 0 else None)
        lowest = min(r["Ea_fwd"] for r in part)
        ax.annotate("%.2f" % lowest, (column - .13, lowest), xytext=(-8, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=8,
                    color="#262626")
        ax.text(column, 1.0, "n=%d" % len(part), transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color="#6b6b6b")
    ax.set_xticks(range(len(reactions)))
    ax.set_xticklabels(["R%d" % r for r in reactions])
    ax.set_xlim(-.6, len(reactions) - .4)
    ax.axhline(0, color="#6b6b6b", lw=.6)
    ax.set_ylabel("barrier (eV, MACE)")
    ax.set_title("Guessed barriers   ·   %s" % title, loc="left", fontweight="bold", pad=18)
    ax.grid(axis="y", color="#e9e9e9")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return names


run = layout.RunLayout(settings.RUN_DIR)
if not os.path.isdir(run.ts_harmonic):
    sys.exit("%s not found -- run step 7 first" % run.ts_harmonic)
if not os.path.isdir(run.ts_sella):
    sys.exit("%s not found -- run step 8 first" % run.ts_sella)

rows, counts = [], {}
bands = {"built": 0, "up to date": 0, "started": 0}
for reaction_name in subdirs(run.ts_harmonic):
    head = reaction_name.split("_")[0]
    if not head.isdigit():
        continue
    index = int(head)
    info = load(os.path.join(run.ts_harmonic, reaction_name, layout.REACTION_INFO)) or {}
    count = counts.setdefault(index, {"guesses": 0, "endpoints": 0, "collected": 0})
    for pair in subdirs(os.path.join(run.ts_harmonic, reaction_name)):
        for stem in subdirs(os.path.join(run.ts_harmonic, reaction_name, pair)):
            guess_dir = os.path.join(run.ts_harmonic, reaction_name, pair, stem)
            record = load(os.path.join(guess_dir, layout.TS_BONDS_JSON))
            if record is None:
                continue
            count["guesses"] += 1
            initial, final = endpoint(guess_dir, "initial"), endpoint(guess_dir, "final")
            if not (initial and final):
                continue
            count["endpoints"] += 1
            ts = best_ts(os.path.join(run.ts_sella, reaction_name, pair, stem))
            if ts is None:
                continue
            count["collected"] += 1
            start, ts_result, ts_path = ts

            e_is, e_ts, e_fs = initial[0]["energy"], ts_result["energy"], final[0]["energy"]
            imaginary = ts_result.get("imaginary") or []
            relative = os.path.join(reaction_name, pair, stem)
            rows.append({"reaction": index, "reaction_name": info.get("reaction", reaction_name),
                         "pair": pair, "stem": stem, "ts_start": start,
                         "E_initial": e_is, "E_ts": e_ts, "E_final": e_fs,
                         "Ea_fwd": e_ts - e_is, "Ea_rev": e_ts - e_fs, "dE_rxn": e_fs - e_is,
                         "imag_cm": imaginary[0] if imaginary else None})

            target = os.path.join(run.neb, relative)
            if not os.path.isfile(os.path.join(target, layout.NEB_RECORD)):
                collect(target, initial[1], ts_path, final[1], record, relative, index,
                        rows[-1]["reaction_name"], start, (e_is, e_ts, e_fs), imaginary)
            outcome = make_band(target, record["n_framework"])
            bands[outcome] += 1
            if outcome == "built":
                print("  band built   %s" % relative)

print("%-9s %8s %10s %10s   %-8s %7s   %s"
      % ("reaction", "guesses", "IS+FS ok", "collected", "min Ea", "dE", "lowest-barrier guess"))
for index in sorted(counts):
    c = counts[index]
    part = [r for r in rows if r["reaction"] == index]
    if part:
        low = min(part, key=lambda r: r["Ea_fwd"])
        tail = "%-8.3f %7.3f   %s/%s (%s)" % (low["Ea_fwd"], low["dE_rxn"], low["pair"],
                                              low["stem"], low["ts_start"])
    else:
        tail = "-"
    print("%-9s %8d %10d %10d   %s" % ("R%d" % index, c["guesses"], c["endpoints"],
                                       c["collected"], tail))

if not rows:
    sys.exit("\nno guess has a valid IS, TS and FS yet -- nothing collected")
os.makedirs(run.neb, exist_ok=True)
rows.sort(key=lambda r: (r["reaction"], r["Ea_fwd"]))
csv_path = os.path.join(run.neb, BARRIERS_CSV)
with open(csv_path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
png_path = os.path.join(run.neb, BARRIERS_PNG)
plot_barriers(rows, png_path, os.path.relpath(settings.RUN_DIR))
print("\nenergies in eV (MACE); min Ea = lowest E_TS - E_IS of that reaction, dE = its E_FS - E_IS")
print("bands: %d built, %d up to date, %d left alone (NEB submitted or done)   "
      "[%d + %d images around the TS, %s]"
      % (bands["built"], bands["up to date"], bands["started"], settings.NEB_IMAGES_BEFORE,
         settings.NEB_IMAGES_AFTER, settings.NEB_INTERPOLATION))
print("collected into %s\nwrote %s\n      %s" % (run.neb, csv_path, png_path))