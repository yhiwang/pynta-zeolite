#!/usr/bin/env python
"""DFT step 2: pick the structures that get a DFT single point.

    python dft/02_select_configs.py          # SELECT_WRITE = False: pick and report only

For every reaction that has a pool (dft/01), it picks SELECT_N structures
that are as different from each other as possible:

  1. SOAP (DScribe) on the adsorbate atoms, the site Al and its first-shell
     O, averaged over those atoms: one vector per structure, scaled to
     length 1. The distance between two structures is the distance between
     their vectors.
  2. Farthest point sampling: each pick is the structure farthest from all
     structures already picked. Already picked are the job folders of the
     reaction in this round and in the earlier rounds of the model
     (settings.MODELS). With none, the first pick is the structure nearest
     to the average of the pool.
  3. It stops at SELECT_N picks in this round, or earlier when every
     structure left is a copy of a picked one (distance below SELECT_SAME).

Each pick gets a job folder:

    runs/<MODEL>/<ROUND>/<FRAMEWORK>/DFT_sp/<i>_rxn/<site>/<pair>/<stem>/<name>/
        config.extxyz   the structure from the pool (MACE values as mace_*)
        source.json     its pool.csv row, the pick number, the SOAP distance when picked
    runs/<MODEL>/<ROUND>/<FRAMEWORK>/DFT_sp/<i>_rxn/selected.csv   every pick, in pick order

A folder that exists is never written again (a DFT job may be running in
it), so a rerun with a larger SELECT_N, or after the pool grew, only adds
picks. "different" in the report is the pool without exact copies. Runs on
the login node and submits nothing.
"""

import csv
import json
import os
import sys
from collections import Counter

import _common  # noqa: F401
import dft_settings as ds
import layout
import numpy as np
import settings
from ase.io import read, write

try:
    from dscribe.descriptors import SOAP
except ImportError:
    sys.exit("DScribe is not installed in this environment: pip install dscribe")

SELECTED_FIELDS = ["pick", "config", "site", "pair", "stem", "file", "frame",
                   "soap_distance", "mace_energy", "mace_fmax"]
SITE_ATOMS = {}          # (round, site) -> [Al index, its first-shell O indices]


def site_atoms(round_name, site):
    """Indices of the site Al and its first-shell O, from the site's framework.json."""
    if (round_name, site) not in SITE_ATOMS:
        path = os.path.join(layout.framework_dir(settings.MODEL, round_name, settings.FRAMEWORK),
                            site, layout.FRAMEWORK_JSON)
        with open(path) as handle:
            record = json.load(handle)
        SITE_ATOMS[round_name, site] = (list(record["al_indices"])
                                        + [s["indices"][0] for s in record["mono_sites"]])
    return SITE_ATOMS[round_name, site]


def picked(round_name, reaction):
    """[(source.json dict, structure)] of every job folder of one reaction in one round."""
    root = os.path.join(layout.framework_dir(settings.MODEL, round_name, settings.FRAMEWORK),
                        layout.DFT_SP, reaction)
    found = []
    for folder, _, files in os.walk(root):
        if layout.DFT_CONFIG in files:
            with open(os.path.join(folder, layout.DFT_SOURCE)) as handle:
                source = json.load(handle)
            atoms = read(os.path.join(folder, layout.DFT_CONFIG))
            atoms.pbc = True
            found.append((source, atoms))
    return sorted(found, key=lambda item: item[0]["pick"])


def soap_vectors(soap, structures, round_names):
    """One SOAP vector of length 1 per structure, as rows."""
    rows = []
    for atoms, round_name in zip(structures, round_names):
        centres = (list(range(atoms.info["n_framework"], len(atoms)))
                   + site_atoms(round_name, atoms.info["site"]))
        rows.append(soap.create(atoms, centers=centres))
    rows = np.array(rows)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def plain(value):
    return value.item() if hasattr(value, "item") else value


if ds.SELECT_N is None:
    sys.exit("set SELECT_N in dft/dft_settings.py")
frame_dir = layout.framework_dir(settings.MODEL, settings.ROUND, settings.FRAMEWORK)
dft_sp = os.path.join(frame_dir, layout.DFT_SP)
pool_dir = os.path.join(dft_sp, layout.DFT_POOL)
if not os.path.isdir(pool_dir):
    sys.exit("%s not found -- run dft/01_collect_configs.py first" % pool_dir)
rounds = list(settings.MODELS[settings.MODEL])
earlier = rounds[:rounds.index(settings.ROUND)]
print("%s   N = %d   earlier rounds: %s" % (dft_sp, ds.SELECT_N, ", ".join(earlier) or "none"))

report = []              # one dict per reaction
for reaction in sorted(os.listdir(pool_dir)):
    pool_file = os.path.join(pool_dir, reaction, layout.DFT_POOL_CONFIGS)
    if not os.path.isfile(pool_file):
        continue
    index = int(reaction.split("_")[0])
    pool = read(pool_file, ":")
    for atoms in pool:
        atoms.pbc = True
    names = [atoms.info["config"] for atoms in pool]

    before = picked(settings.ROUND, reaction)                       # this round
    start = [(atoms, settings.ROUND) for _, atoms in before]
    for round_name in earlier:
        start += [(atoms, round_name) for _, atoms in picked(round_name, reaction)]

    species = sorted({symbol for atoms in pool + [a for a, _ in start]
                      for symbol in atoms.get_chemical_symbols()})
    soap = SOAP(species=species, periodic=True, average="outer", **ds.SELECT_SOAP)
    vectors = soap_vectors(soap, pool, [settings.ROUND] * len(pool))

    # the pool without exact copies: a structure counts when no earlier one is the same
    squared = np.clip(2.0 - 2.0 * vectors @ vectors.T, 0.0, None)
    different = int((~np.tril(squared < ds.SELECT_SAME ** 2, k=-1).any(axis=1)).sum())

    # distance of every pool structure to the nearest structure already picked
    distance = np.full(len(pool), np.inf)
    if start:
        for vector in soap_vectors(soap, [a for a, _ in start], [r for _, r in start]):
            distance = np.minimum(distance, np.linalg.norm(vectors - vector, axis=1))
    taken = {source["config"] for source, _ in before}
    distance[[i for i, name in enumerate(names) if name in taken]] = 0.0

    picks = []           # (pool index, distance when picked)
    while len(before) + len(picks) < ds.SELECT_N:
        if np.isinf(distance).all():                # nothing picked yet: start in the middle
            i, d = int(np.linalg.norm(vectors - vectors.mean(axis=0), axis=1).argmin()), None
        else:
            i = int(distance.argmax())
            d = float(distance[i])
            if d < ds.SELECT_SAME:                   # only copies of picked structures are left
                break
        picks.append((i, d))
        distance = np.minimum(distance, np.linalg.norm(vectors - vectors[i], axis=1))

    sources = [source for source, _ in before]
    for number, (i, d) in enumerate(picks, start=len(before) + 1):
        source = {key: plain(value) for key, value in pool[i].info.items()}
        source.update(model=settings.MODEL, round=settings.ROUND, framework=settings.FRAMEWORK,
                      pick=number, soap_distance=d)
        sources.append(source)
        if ds.SELECT_WRITE:
            folder = os.path.join(dft_sp, reaction, *names[i].split("/"))
            os.makedirs(folder)                     # stops if the folder exists: never overwrite
            write(os.path.join(folder, layout.DFT_CONFIG), pool[i], format="extxyz")
            with open(os.path.join(folder, layout.DFT_SOURCE), "w") as handle:
                json.dump(source, handle, indent=2)
    if ds.SELECT_WRITE and sources:
        with open(os.path.join(dft_sp, reaction, layout.DFT_SELECTED), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SELECTED_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(sources)

    distances = [d for _, d in picks if d is not None]
    report.append({"index": index, "pool": len(pool), "different": different,
                   "before": len(before), "new": len(picks),
                   "first": max(distances) if distances else None,
                   "last": min(distances) if distances else None,
                   "files": Counter(source["file"] for source in sources),
                   "sites": Counter(source["site"] for source in sources)})

if not report:
    sys.exit("no pool under %s -- run dft/01_collect_configs.py first" % pool_dir)

# -- report ------------------------------------------------------------------
print("\n%-9s %6s %10s %7s %5s %6s %12s %12s"
      % ("reaction", "pool", "different", "before", "new", "total", "largest d", "smallest d"))
for r in report:
    print("%-9s %6d %10d %7d %5d %6d %12s %12s"
          % ("R%d" % r["index"], r["pool"], r["different"], r["before"], r["new"],
             r["before"] + r["new"],
             "-" if r["first"] is None else "%.4f" % r["first"],
             "-" if r["last"] is None else "%.4f" % r["last"]))

files = [name for name in layout.POOL_FILES if any(r["files"][name] for r in report)]
print("\npicks by file")
print("%-9s " % "reaction" + " ".join("%12s" % name for name in files))
for r in report:
    print("%-9s " % ("R%d" % r["index"]) + " ".join("%12d" % r["files"][name] for name in files))

print("\npicks by site")
print("%-9s " % "reaction" + " ".join("%6s" % site for site in settings.SITES))
for r in report:
    print("%-9s " % ("R%d" % r["index"]) + " ".join("%6d" % r["sites"][site] for site in settings.SITES))

if ds.SELECT_WRITE:
    print("\nwrote the job folders of %d new picks under %s" % (sum(r["new"] for r in report), dft_sp))
else:
    print("\ndry run: nothing written. Set SELECT_WRITE = True in dft/dft_settings.py")