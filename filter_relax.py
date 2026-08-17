#!/usr/bin/env python
"""
Filter relaxed configs: drop the ones that fell apart, then deduplicate.

    python filter_relax.py

Two passes over test_run/Adsorbates_relax/<species>/<site>/<stem>/relax.traj.

Survival compares the first and last frame and keeps a config only if every
bond in the adsorbate, plus the binder-to-framework bond, held. Survivors go
to Adsorbates_relax_filtered/<species>/<site>/<stem>.xyz -- flat per site,
the orientation stem carried through unchanged from placement.

Deduplication then clusters the survivors within each (species, site) by
positional RMSD and keeps the lowest-energy member of each cluster, writing
Adsorbates_relax_unique/ in the same layout.
"""

import os
import json
import shutil

import numpy as np
from ase.io import read
from ase.geometry import find_mic
from ase.io.trajectory import Trajectory

from functions_v2 import (find_framework_indices, extra_framework_indices,
                          neighbor_map, check_config_survived)

RELAX_DIR = os.path.join("test_run", "Adsorbates_relax")
INFO_DIR = os.path.join("test_run", "Adsorbates")
FILTERED_DIR = os.path.join("test_run", "Adsorbates_relax_filtered")
UNIQUE_DIR = os.path.join("test_run", "Adsorbates_relax_unique")
STRUCT_NAME = "relax.xyz"
BOND_THRESH = 0.5    # A, a bond that moved more than this has broken
RMSD_THRESH = 2.0    # A, below this two configs are the same minimum


def find_configs(species_dir):
    """[(site, stem, dirpath)] for one species.

    A config directory is any that holds a relax.traj. os.walk, not glob:
    species names like C[CH2][Pt] contain brackets, which glob reads as
    character classes."""
    configs = []
    for dirpath, dirs, files in os.walk(species_dir):
        if "relax.traj" not in files:
            continue
        rel = os.path.relpath(dirpath, species_dir).split(os.sep)
        if len(rel) != 2:
            continue
        configs.append((rel[0], rel[1], dirpath))
    return sorted(configs)


def survived(dirpath, info):
    """Whether this config came through relaxation with every bond intact.
    A traj with fewer than two frames is a job that died before it moved."""
    traj = Trajectory(os.path.join(dirpath, "relax.traj"))
    if len(traj) < 2:
        return False

    initial, relaxed = traj[0], traj[-1]
    framework = find_framework_indices(initial)
    binders = sorted(len(framework) + int(k)
                     for k in info["gratom_to_molecule_surface_atom_map"])
    ok, _ = check_config_survived(initial, relaxed,
                                  neighbor_map(initial, binders), BOND_THRESH)
    return ok


def load(path):
    """(atoms, energy); energy is None when the file carries none."""
    atoms = read(path)
    try:
        return atoms, atoms.get_potential_energy()
    except (RuntimeError, AttributeError):
        return atoms, None


def rmsd(a, b, indices):
    """Positional RMSD over `indices`, minimum-image.

    Valid without superposition only because the framework is frozen: both
    structures sit in the same frame, so each displacement is real and not a
    rigid-body offset."""
    delta = b.get_positions()[indices] - a.get_positions()[indices]
    _, lengths = find_mic(delta, a.cell, a.pbc)
    return float(np.sqrt((lengths ** 2).mean()))


def dedupe(entries, threshold):
    """[(representative, [members])] by greedy clustering.

    `entries` is [(stem, atoms, energy)], already energy-ordered, so every
    representative is the lowest-energy member of its own cluster."""
    indices = extra_framework_indices(entries[0][1])
    clusters = []
    for entry in entries:
        for representative, members in clusters:
            if rmsd(representative[1], entry[1], indices) < threshold:
                members.append(entry)
                break
        else:
            clusters.append((entry, [entry]))
    return clusters


if not os.path.isdir(RELAX_DIR):
    raise SystemExit("%s not found -- run from the repo root" % RELAX_DIR)

for species in sorted(os.listdir(RELAX_DIR)):
    species_dir = os.path.join(RELAX_DIR, species)
    if not os.path.isdir(species_dir):
        continue

    info_path = os.path.join(INFO_DIR, species, "info.json")
    if not os.path.isfile(info_path):
        print("%s: no info.json at %s" % (species, info_path))
        continue
    with open(info_path) as f:
        info = json.load(f)

    configs = find_configs(species_dir)
    if not configs:
        continue

    print("\n%s" % species)

    by_site = {}
    failed = {}
    for site, stem, dirpath in configs:
        src = os.path.join(dirpath, STRUCT_NAME)
        # a running job has a traj but no final structure yet -- it is
        # written only after the optimizer returns
        if not os.path.isfile(src) or not survived(dirpath, info):
            failed.setdefault(site, []).append(stem)
            continue

        dst_dir = os.path.join(FILTERED_DIR, species, site)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, stem + ".xyz")
        shutil.copy2(src, dst)

        atoms, energy = load(dst)
        by_site.setdefault(site, []).append((stem, atoms, energy))

    for site in sorted(set(by_site) | set(failed)):
        entries = by_site.get(site, [])
        n_total = len(entries) + len(failed.get(site, []))
        if not entries:
            print("  site %s: 0/%d survived" % (site, n_total))
            continue

        # energy first, missing energies last in stem order; the placeholder
        # keeps the key comparable when a field is None
        entries.sort(key=lambda e: (e[2] is None,
                                    0.0 if e[2] is None else e[2], e[0]))
        clusters = dedupe(entries, RMSD_THRESH)

        print("  site %s: %d/%d survived -> %d unique"
              % (site, len(entries), n_total, len(clusters)))

        dst_dir = os.path.join(UNIQUE_DIR, species, site)
        os.makedirs(dst_dir, exist_ok=True)
        for representative, members in clusters:
            stem, _, energy = representative
            shutil.copy2(os.path.join(FILTERED_DIR, species, site,
                                      stem + ".xyz"),
                         os.path.join(dst_dir, stem + ".xyz"))
            merged = "" if len(members) == 1 else \
                "  <- %s" % ", ".join(m[0] for m in members[1:])
            print("      %-24s %s%s"
                  % (stem,
                     "E=%.3f eV" % energy if energy is not None else "no E",
                     merged))

    for tree in (FILTERED_DIR, UNIQUE_DIR):
        species_dst = os.path.join(tree, species)
        if os.path.isdir(species_dst):
            shutil.copy2(info_path, os.path.join(species_dst, "info.json"))