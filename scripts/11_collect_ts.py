#!/usr/bin/env python
"""Step 11 -- collect the TS of every finished NEB worth a DFT search into
DFT_TS/, with everything about it in one place. Submits nothing; runs on the
login node.

    python scripts/11_collect_ts.py

A band is collected when its neb_result.json verdict is in
TS_COLLECT_VERDICTS (default connected, different_ts: the climbing stage got
below NEB_FMAX and the band has a single barrier). Everything comes from
neb_band.traj -- one model, one free region, end points re-relaxed -- never
mixed with the step 7/8 energies in neb.json, which used smaller free regions:

    <RUN_DIR>/DFT_TS/<i>_rxn/pair_<k>/<stem>/ts.xyz        the top (climbing) image
    <RUN_DIR>/DFT_TS/<i>_rxn/pair_<k>/<stem>/initial.xyz   the band's initial state
    <RUN_DIR>/DFT_TS/<i>_rxn/pair_<k>/<stem>/final.xyz     the band's final state
                      (extxyz, MACE energy + full forces, no constraint)
    <RUN_DIR>/DFT_TS/<i>_rxn/pair_<k>/<stem>/ts_info.json  facts about this TS, below
    <RUN_DIR>/DFT_TS/ts_collected.csv                      one row per TS + the ranking

ts_info.json
    where     framework code, sites, run dir, round; reaction index and name,
              pair, stem, the NEB folder it came from
    site      the seated oxygens and the framework Al next to them
    atoms     n_atoms, n_framework, adsorbate formula, the band's free atoms
              (the DFT search should free the same region)
    bonds     every breaking / forming bond: indices, elements, length in the
              initial state, the TS and the final state (A, minimum image)
    neb       verdict, converged, final band fmax and its limit, steps per
              stage, top image, top vs Sella TS (RMSD, bond change, energy),
              model, and the step 8 Sella imaginary frequency
    energies  MACE E of initial, TS, final; Ea_fwd, Ea_rev, dE_rxn; ts_fmax,
              the largest force on a free atom of the TS

ts_collected.csv adds the ranking, per reaction (same atoms, same cell, so
absolute energies compare across pairs and stems):
    dE_ts_min     E_TS - lowest E_TS of the reaction
    Ea_common     E_TS - lowest E_initial of the reaction
    duplicate_of  a lower TS of the same reaction within TS_DUPLICATE_RMSD
                  (adsorbate, minimum image, no superposition) and
                  TS_DUPLICATE_ENERGY, else empty. Atoms are compared by index,
                  so a TS that only differs by swapped equivalent atoms (two H
                  of a CH3) is not caught: duplicates can be missed, never
                  invented.

A folder with a job.id (a DFT job was submitted from it) is never rewritten;
its ts_info.json and ts.xyz still go into the table. Every other folder is
written fresh on every run, so rerunning after new NEBs updates it. Folders
of bands that no longer pass are left alone.

Needs in scripts/settings.py: TS_COLLECT_VERDICTS, TS_COLLECT_REACTIONS,
TS_DUPLICATE_RMSD, TS_DUPLICATE_ENERGY; in scripts/layout.py:
TS_COLLECT_STATES, TS_COLLECT_INFO, TS_COLLECT_CSV, RunLayout.dft_ts.
"""

import csv
import json
import os
import sys
from collections import Counter, defaultdict

import _common  # noqa: F401
import layout
import numpy as np
import settings
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from pyntaz.neb import displacements

REQUIRED_SETTINGS = ("RUN_DIR", "CODE", "SITES", "TS_COLLECT_VERDICTS",
                     "TS_COLLECT_REACTIONS", "TS_DUPLICATE_RMSD", "TS_DUPLICATE_ENERGY")
REQUIRED_LAYOUT = ("TS_COLLECT_STATES", "TS_COLLECT_INFO", "TS_COLLECT_CSV")
STATES = ("initial", "ts", "final")
AL_O_MAX = 2.2          # A, a framework Al this close to a seated oxygen is the site's Al

COLUMNS = ["id", "reaction", "reaction_name", "pair", "stem", "oxygens", "al", "verdict",
           "neb_fmax", "ts_fmax", "E_initial", "E_ts", "E_final", "Ea_fwd", "Ea_rev",
           "dE_rxn", "dE_ts_min", "Ea_common", "duplicate_of", "bonds_ts", "model"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def check_setup():
    """Stop before touching anything when a setting or layout name is missing."""
    missing = ["settings." + n for n in REQUIRED_SETTINGS if not hasattr(settings, n)]
    missing += ["layout." + n for n in REQUIRED_LAYOUT if not hasattr(layout, n)]
    if "settings.RUN_DIR" not in missing and not hasattr(
            layout.RunLayout(settings.RUN_DIR), "dft_ts"):
        missing.append("layout.RunLayout.dft_ts")
    if missing:
        sys.exit("missing: %s\nadd them first (see the docstring of this script)"
                 % ", ".join(missing))


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def jsonable(value):
    """``value`` if json can write it, else its str (SITES may be anything)."""
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def single_point(atoms):
    """A clean copy of a band image: no constraint, its MACE energy and the
    forces on every atom (frozen ones included)."""
    copy = atoms.copy()
    copy.set_constraint()
    copy.pbc = True
    copy.calc = SinglePointCalculator(copy, energy=float(atoms.get_potential_energy()),
                                      forces=atoms.get_forces(apply_constraint=False))
    return copy


def site_al(atoms, oxygens, n_framework):
    """Framework Al atoms within AL_O_MAX of any seated oxygen."""
    al = [i for i in range(n_framework) if atoms[i].symbol == "Al"]
    near = set()
    for o in (oxygens if al else []):
        distances = atoms.get_distances(o, al, mic=True)
        near.update(int(al[k]) for k in np.flatnonzero(distances < AL_O_MAX))
    return sorted(near)


def bond_records(record, states):
    """Every reacting bond of neb.json with its length in the three states."""
    out = []
    for bond in record["bonds"]:
        i, j = bond["indices"]
        entry = {"name": "%s %s" % (bond["change"], "-".join(bond["tags"])),
                 "change": bond["change"], "tags": bond["tags"], "indices": [i, j],
                 "elements": [states["ts"][i].symbol, states["ts"][j].symbol]}
        for state in STATES:
            entry[state] = round(float(states[state].get_distance(i, j, mic=True)), 3)
        out.append(entry)
    return out


# --------------------------------------------------------------------------
# one band
# --------------------------------------------------------------------------

def collect_band(folder, relative, target, record, result):
    """(info, states) of one finished NEB folder; writes nothing."""
    band = load_json(os.path.join(folder, layout.NEB_BAND_INFO))
    images = read(os.path.join(folder, layout.NEB_FINAL_BAND), ":")
    top = int(result["top_image"])
    energies = [float(a.get_potential_energy()) for a in images]
    if top != int(np.argmax(energies[1:-1]) + 1):
        print("  warning %s: top_image %d is not the highest image of neb_band.traj"
              % (relative, top))
    states = {"initial": single_point(images[0]), "ts": single_point(images[top]),
              "final": single_point(images[-1])}

    n_framework = record["n_framework"]
    free = band["free"]
    ts_forces = states["ts"].get_forces()[free]
    e_is, e_ts, e_fs = (float(states[s].get_potential_energy()) for s in STATES)
    parts = relative.split(os.sep)
    climb = result.get("climb") or {}

    info = {
        "id": relative,
        "where": {"framework": jsonable(settings.CODE),
                  "sites": jsonable(settings.SITES),
                  "run_dir": settings.RUN_DIR,
                  "round": jsonable(getattr(settings, "ROUND", None)),
                  "reaction_index": record["index"],
                  "reaction": record["reaction"],
                  "pair": parts[1],
                  "stem": os.path.join(*parts[2:]),
                  "neb_folder": os.path.relpath(folder, target)},
        "site": {"oxygens": record["oxygens"],
                 "al": site_al(states["ts"], record["oxygens"], n_framework)},
        "atoms": {"n_atoms": len(states["ts"]),
                  "n_framework": n_framework,
                  "adsorbate": states["ts"][n_framework:].get_chemical_formula(),
                  "free": free},
        "bonds": bond_records(record, states),
        "neb": {"verdict": result["verdict"],
                "converged": result["converged"],
                "fmax": result["fmax"],
                "fmax_limit": result["settings"]["fmax"],
                "steps_rough": result["rough"]["n_steps"],
                "steps_climb": climb.get("n_steps"),
                "n_images": result["n_images"],
                "top_image": top,
                "top_vs_ts_rmsd": result.get("top_vs_ts_rmsd"),
                "top_vs_ts_bond_max": result.get("top_vs_ts_bond_max"),
                "top_minus_ts_energy": result.get("top_minus_ts_energy"),
                "model": result["model"],
                "sella_imaginary": record.get("imaginary")},
        "energies": {"initial": e_is, "ts": e_ts, "final": e_fs,
                     "Ea_fwd": e_ts - e_is, "Ea_rev": e_ts - e_fs, "dE_rxn": e_fs - e_is,
                     "ts_fmax": float(np.linalg.norm(ts_forces, axis=1).max())},
    }
    return info, states


def write_band(target, info, states):
    os.makedirs(target, exist_ok=True)
    for state in STATES:
        write(os.path.join(target, layout.TS_COLLECT_STATES[state]), states[state],
              format="extxyz")
    with open(os.path.join(target, layout.TS_COLLECT_INFO), "w") as handle:
        json.dump(info, handle, indent=2)


# --------------------------------------------------------------------------
# ranking within each reaction
# --------------------------------------------------------------------------

def duplicate_of(info, kept, structures, rmsd_tol, energy_tol):
    """id of the first kept TS this one repeats, or None."""
    atoms = structures[info["id"]]
    adsorbate = np.arange(info["atoms"]["n_framework"], len(atoms))
    for other in kept:
        ref = structures[other["id"]]
        if len(ref) != len(atoms):
            continue
        if abs(info["energies"]["ts"] - other["energies"]["ts"]) > energy_tol:
            continue
        rmsd = float(np.sqrt((displacements(ref, atoms, adsorbate) ** 2).mean()))
        if rmsd < rmsd_tol:
            return other["id"]
    return None


def rank(infos, structures):
    """{id: ranking dict}; within a reaction, lowest E_TS first."""
    by_reaction = defaultdict(list)
    for info in infos:
        by_reaction[info["where"]["reaction_index"]].append(info)
    ranking = {}
    for index, group in by_reaction.items():
        formulas = {(i["atoms"]["n_atoms"], i["atoms"]["adsorbate"]) for i in group}
        if len(formulas) > 1:
            print("  warning R%d: TSs with different atoms %s -- energies not comparable"
                  % (index, sorted(formulas)))
        group.sort(key=lambda i: i["energies"]["ts"])
        e_ts_min = group[0]["energies"]["ts"]
        e_is_min = min(i["energies"]["initial"] for i in group)
        kept = []
        for info in group:
            same = duplicate_of(info, kept, structures, settings.TS_DUPLICATE_RMSD,
                                settings.TS_DUPLICATE_ENERGY)
            ranking[info["id"]] = {"dE_ts_min": info["energies"]["ts"] - e_ts_min,
                                   "Ea_common": info["energies"]["ts"] - e_is_min,
                                   "duplicate_of": same}
            if same is None:
                kept.append(info)
    return ranking


def table_row(info, ranking):
    e = info["energies"]
    r = ranking[info["id"]]
    return {"id": info["id"],
            "reaction": info["where"]["reaction_index"],
            "reaction_name": info["where"]["reaction"],
            "pair": info["where"]["pair"],
            "stem": info["where"]["stem"],
            "oxygens": " ".join(str(o) for o in info["site"]["oxygens"]),
            "al": " ".join(str(a) for a in info["site"]["al"]),
            "verdict": info["neb"]["verdict"],
            "neb_fmax": round(info["neb"]["fmax"], 4),
            "ts_fmax": round(e["ts_fmax"], 4),
            "E_initial": round(e["initial"], 6),
            "E_ts": round(e["ts"], 6),
            "E_final": round(e["final"], 6),
            "Ea_fwd": round(e["Ea_fwd"], 4),
            "Ea_rev": round(e["Ea_rev"], 4),
            "dE_rxn": round(e["dE_rxn"], 4),
            "dE_ts_min": round(r["dE_ts_min"], 4),
            "Ea_common": round(r["Ea_common"], 4),
            "duplicate_of": r["duplicate_of"] or "",
            "bonds_ts": "; ".join("%s %.3f" % (b["name"], b["ts"]) for b in info["bonds"]),
            "model": info["neb"]["model"]}


# --------------------------------------------------------------------------

def main():
    check_setup()
    run = layout.RunLayout(settings.RUN_DIR)
    if not os.path.isdir(run.neb):
        sys.exit("no NEB folder at %s" % run.neb)
    verdicts = set(settings.TS_COLLECT_VERDICTS)
    reactions = settings.TS_COLLECT_REACTIONS

    infos, structures = [], {}
    names, counts, status = {}, defaultdict(Counter), Counter()
    for root, dirs, files in os.walk(run.neb):
        dirs.sort()
        if layout.NEB_RESULT not in files:
            continue
        relative = os.path.relpath(root, run.neb)
        record = load_json(os.path.join(root, layout.NEB_RECORD))
        index = record["index"]
        if reactions is not None and index not in reactions:
            continue
        result = load_json(os.path.join(root, layout.NEB_RESULT))
        names[index] = record["reaction"]
        counts[index][result["verdict"]] += 1
        if result["verdict"] not in verdicts:
            continue

        target = os.path.join(run.dft_ts, relative)
        if os.path.isfile(os.path.join(target, layout.JOB_ID)):
            info = load_json(os.path.join(target, layout.TS_COLLECT_INFO))
            ts = read(os.path.join(target, layout.TS_COLLECT_STATES["ts"]))
            status["kept (DFT job submitted)"] += 1
        else:
            info, states = collect_band(root, relative, target, record, result)
            write_band(target, info, states)
            ts = states["ts"]
            status["written"] += 1
        infos.append(info)
        structures[info["id"]] = ts

    if not names:
        sys.exit("no finished NEB (neb_result.json) under %s" % run.neb)
    ranking = rank(infos, structures)
    rows = sorted((table_row(i, ranking) for i in infos),
                  key=lambda row: (row["reaction"], row["Ea_common"]))
    if rows:
        os.makedirs(run.dft_ts, exist_ok=True)
        csv_path = os.path.join(run.dft_ts, layout.TS_COLLECT_CSV)
        with open(csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    print("collecting verdicts %s from %s\n" % (", ".join(sorted(verdicts)), run.neb))
    for index in sorted(names):
        part = [row for row in rows if row["reaction"] == index]
        found = ", ".join("%s %d" % (v, n) for v, n in sorted(counts[index].items()))
        print("R%-3d %s" % (index, names[index]))
        print("     NEB verdicts: %s" % found)
        if not part:
            print("     -> nothing collected: no DFT barrier for this reaction from this step")
            continue
        unique = [row for row in part if not row["duplicate_of"]]
        best = part[0]
        print("     -> %d collected, %d unique; lowest Ea_common %.3f eV (%s, %s)"
              % (len(part), len(unique), best["Ea_common"], best["id"], best["verdict"]))
    print("\n" + ", ".join("%s %d" % (k, n) for k, n in sorted(status.items())))
    if rows:
        print("table: %s" % csv_path)


if __name__ == "__main__":
    main()