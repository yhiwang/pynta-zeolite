#!/usr/bin/env python
"""DFT step 1: collect the pool of structures that dft/02 picks from.

    python dft/01_collect_configs.py          # COLLECT_WRITE = False: count only

For every site of settings.MODEL and settings.FRAMEWORK and every guess of
step 3, it takes every structure in the files named in COLLECT_FILES
(dft/dft_settings.py; the paths are in layout.pool_files):

    ts_harmonic   TS_harmonic/.../<stem>/ts_harmonic.xyz       step 3 TS, where step 4 starts
    initial       TS_harmonic/.../<stem>/initial/initial.xyz   step 3 initial state
    final         TS_harmonic/.../<stem>/final/final.xyz       step 3 final state
    ts_sella      TS_sella/.../<stem>/harmonic/ts_sella.xyz    step 4 result
    neb_start     NEB/.../<stem>/neb.traj, first band only     the band step 6 starts from,
                                                               with its MACE energies and forces
    neb_band      NEB/.../<stem>/neb_band.traj                 the band where step 6 stopped

    raw           TS_harmonic/.../<stem>/<stem>_init.xyz       the step 1 guess   } no MACE values,
    band          NEB/.../<stem>/band.traj                     the start band     } not in the list now

A file that does not exist is counted as "no file" (a guess without a band
has no NEB files). Every structure gets three checks, in this order, and is
removed at the first one it fails:

    close   two atoms are closer than COLLECT_MIN_DIST
    force   a force stored in the file is above COLLECT_MAX_FORCE (a file
            that stores no force does not get this check)
    bonds   pyntaz.connectivity.image_problems against the raw guess of the
            same guess: a bond that is not a reacting bond broke or formed,
            or an adsorbate atom is alone (COLLECT_ALONE). The reacting
            bonds are in ts_bonds.json

The rest go into the pool of their reaction, all sites together:

    runs/<MODEL>/<ROUND>/<FRAMEWORK>/DFT_sp/pool/<i>_rxn/pool.extxyz
    runs/<MODEL>/<ROUND>/<FRAMEWORK>/DFT_sp/pool/<i>_rxn/pool.csv
    runs/<MODEL>/<ROUND>/<FRAMEWORK>/DFT_sp/pool/<i>_rxn/rejected.extxyz, rejected.csv

pool.extxyz holds the structures without calculator or constraint; the
stored MACE energy and forces are kept as mace_energy (info) and
mace_forces (arrays). pool.csv has one row per structure, in the same
order. ``config`` is <site>/<pair>/<stem>/<name>, the name of the structure
in every later step. The rejected files hold the removed structures in the
same way, with the reason, to look at; no later step reads them. All four
files of a reaction are written again on every run.

The report counts structures, except "guesses" and "no file" (the guesses
of the reaction, and how many of them do not have that file). Runs on the
login node and submits nothing.
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
from pyntaz.connectivity import image_problems
from sampling import min_distance

POOL_FIELDS = ["config", "site", "reaction", "pair", "stem", "file", "frame", "n_frames",
               "n_atoms", "n_framework", "min_dist", "mace_fmax", "mace_energy"]
KINDS = ("broken", "extra", "alone")


def subdirs(path):
    return sorted(name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)))


def problem_text(atoms, problem):
    """'broken C291-H297', 'alone H300': the kind and the atoms, symbol + index."""
    return "%s %s" % (problem[0], "-".join("%s%d" % (atoms[i].symbol, i) for i in problem[1:]))


def write_structures(folder, configs_name, table_name, fields, entries):
    """Write [(row, structure)] as an extxyz and a csv, or delete old ones when there are none."""
    for name in (configs_name, table_name):
        if os.path.exists(os.path.join(folder, name)):
            os.remove(os.path.join(folder, name))
    if not entries:
        return
    write(os.path.join(folder, configs_name), [structure for _, structure in entries], format="extxyz")
    with open(os.path.join(folder, table_name), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row for row, _ in entries)


unknown = [name for name in ds.COLLECT_FILES if name not in layout.POOL_FILES]
if unknown:
    sys.exit("unknown COLLECT_FILES %s, expected some of %s" % (unknown, list(layout.POOL_FILES)))

frame_dir = layout.framework_dir(settings.MODEL, settings.ROUND, settings.FRAMEWORK)
pool_dir = os.path.join(frame_dir, layout.DFT_SP, layout.DFT_POOL)
print("%s -> %s" % (frame_dir, pool_dir))
print("files: %s" % ", ".join(ds.COLLECT_FILES))

pool = {}               # reaction index -> [(pool.csv row, structure)]
rejected = {}           # reaction index -> [(rejected.csv row, structure)]
counts = Counter()      # (reaction index, file name, what) -> number of structures
kinds = Counter()       # (reaction index, kind of bond problem) -> number of structures
guesses = Counter()     # reaction index -> number of guesses
for site, run_dir in layout.run_dirs(settings.MODEL, settings.ROUND, settings.FRAMEWORK, settings.SITES):
    run = layout.RunLayout(run_dir)
    if not os.path.isdir(run.ts_harmonic):
        print("%s not found -- site skipped" % run.ts_harmonic)
        continue
    for reaction in subdirs(run.ts_harmonic):
        head = reaction.split("_")[0]
        if not head.isdigit():
            continue
        index = int(head)
        if ds.COLLECT_REACTIONS is not None and index not in ds.COLLECT_REACTIONS:
            continue
        for pair in subdirs(os.path.join(run.ts_harmonic, reaction)):
            for stem in subdirs(os.path.join(run.ts_harmonic, reaction, pair)):
                bonds_json = os.path.join(run.ts_harmonic, reaction, pair, stem, layout.TS_BONDS_JSON)
                if not os.path.isfile(bonds_json):
                    continue                     # step 3 did not set this guess up
                with open(bonds_json) as handle:
                    record = json.load(handle)
                n_framework = record["n_framework"]
                reacting = [tuple(bond["indices"]) for bond in record["bonds"]]
                guesses[index] += 1
                files = layout.pool_files(run, reaction, pair, stem)
                raw = read(files["raw"])         # the raw guess: the bonds every image is checked against
                raw.pbc = True
                for name in ds.COLLECT_FILES:
                    if not os.path.isfile(files[name]):
                        counts[index, name, "no file"] += 1
                        continue
                    which = ":"
                    if name in layout.POOL_FIRST_BAND:
                        band_json = os.path.join(os.path.dirname(files[name]), layout.NEB_BAND_INFO)
                        with open(band_json) as handle:
                            which = ":%d" % json.load(handle)["n_images"]
                    frames = read(files[name], which)
                    for frame, atoms in enumerate(frames):
                        atoms.pbc = True         # the checks must see across the cell edge
                        stored = {} if atoms.calc is None else atoms.calc.results
                        forces = stored.get("forces")
                        energy = stored.get("energy")
                        fmax = None if forces is None else float(np.linalg.norm(forces, axis=1).max())
                        dmin = min_distance(atoms)
                        counts[index, name, "found"] += 1
                        if forces is not None:
                            counts[index, name, "with force"] += 1

                        failed, reason = None, None
                        if dmin < ds.COLLECT_MIN_DIST:
                            failed, reason = "close", "close %.2f A" % dmin
                        elif fmax is not None and fmax > ds.COLLECT_MAX_FORCE:
                            failed, reason = "force", "force %.1f eV/A" % fmax
                        else:
                            problems = image_problems(atoms, raw, reacting, n_framework,
                                                      sites=record["oxygens"],
                                                      cutoff=settings.TS_HARMONIC_BOND_CUTOFF,
                                                      alone=ds.COLLECT_ALONE)
                            if problems:
                                failed = "bonds"
                                reason = "; ".join(problem_text(atoms, p) for p in problems)
                                for kind in {p[0] for p in problems}:
                                    kinds[index, kind] += 1

                        label = name if len(frames) == 1 else "%s_%04d" % (name, frame)
                        row = {"config": "/".join((site, pair, stem, label)), "site": site,
                               "reaction": index, "pair": pair, "stem": stem, "file": name,
                               "frame": frame, "n_frames": len(frames), "n_atoms": len(atoms),
                               "n_framework": n_framework, "min_dist": round(dmin, 3),
                               "mace_fmax": None if fmax is None else round(fmax, 4),
                               "mace_energy": None if energy is None else float(energy)}
                        if failed:
                            row["reason"] = reason
                        structure = atoms.copy()             # a copy has no calculator
                        structure.set_constraint()
                        structure.info.clear()
                        structure.info.update({key: value for key, value in row.items()
                                               if value is not None})
                        if forces is not None:
                            structure.arrays["mace_forces"] = np.array(forces)
                        (rejected if failed else pool).setdefault(index, []).append((row, structure))
                        counts[index, name, failed or "pool"] += 1

if not guesses:
    sys.exit("no guess found under %s" % frame_dir)

# -- report ------------------------------------------------------------------
print("\n%-9s %-12s %8s %8s %7s %6s %6s %6s %6s"
      % ("reaction", "file", "guesses", "no file", "found", "close", "force", "bonds", "pool"))
for index in sorted(guesses):
    for name in ds.COLLECT_FILES:
        force = "%6d" % counts[index, name, "force"] if counts[index, name, "with force"] else "%6s" % "-"
        print("%-9s %-12s %8d %8d %7d %6d %s %6d %6d"
              % ("R%d" % index, name, guesses[index], counts[index, name, "no file"],
                 counts[index, name, "found"], counts[index, name, "close"], force,
                 counts[index, name, "bonds"], counts[index, name, "pool"]))
    print("%-9s %-12s %8s %8s %7d %6d %6d %6d %6d"
          % ("R%d" % index, "all", "", "",
             sum(counts[index, name, "found"] for name in ds.COLLECT_FILES),
             sum(counts[index, name, "close"] for name in ds.COLLECT_FILES),
             sum(counts[index, name, "force"] for name in ds.COLLECT_FILES),
             sum(counts[index, name, "bonds"] for name in ds.COLLECT_FILES),
             len(pool.get(index, []))))
    print("%-9s bonds check: %s\n"
          % ("R%d" % index, ", ".join("%s %d" % (kind, kinds[index, kind]) for kind in KINDS)))

print("pool by site")
print("%-9s " % "reaction" + " ".join("%6s" % site for site in settings.SITES) + " %6s" % "all")
for index in sorted(guesses):
    by_site = Counter(row["site"] for row, _ in pool.get(index, []))
    print("%-9s " % ("R%d" % index) + " ".join("%6d" % by_site[site] for site in settings.SITES)
          + " %6d" % len(pool.get(index, [])))

if not ds.COLLECT_WRITE:
    print("\ndry run: nothing written. Set COLLECT_WRITE = True in dft/dft_settings.py")
    sys.exit()

# -- write ---------------------------------------------------------------------
print()
for index in sorted(guesses):
    folder = os.path.join(pool_dir, layout.reaction_dirname(index))
    os.makedirs(folder, exist_ok=True)
    write_structures(folder, layout.DFT_POOL_CONFIGS, layout.DFT_POOL_TABLE, POOL_FIELDS,
                     pool.get(index, []))
    write_structures(folder, layout.DFT_REJECTED_CONFIGS, layout.DFT_REJECTED_TABLE,
                     POOL_FIELDS + ["reason"], rejected.get(index, []))
    print("wrote %s  (%d in the pool, %d removed)"
          % (folder, len(pool.get(index, [])), len(rejected.get(index, []))))