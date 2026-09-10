#!/usr/bin/env python
"""Step 3 -- keep the relaxed configs that held together, then deduplicate.

Survivors are copied to <RUN_DIR>/Adsorbates_relax_filtered/<species>/<site>/<stem>.xyz;
within each (species, site) they are clustered by adsorbate RMSD and the
lowest-energy member of each cluster goes to Adsorbates_relax_unique/.
"""

import os
import shutil
import sys

import _common  # noqa: F401
import layout
import settings
from ase.io.trajectory import Trajectory
from pyntaz.filtering import (binder_indices, relaxation_survived, sort_by_energy,
                              cluster_by_rmsd)
from pyntaz.geometry import framework_indices


def survived(config_dir, info):
    """Whether the relaxation in ``config_dir`` kept every bond. A trajectory
    with fewer than two frames is a job that died before it moved."""
    trajectory = Trajectory(os.path.join(config_dir, layout.RELAX_TRAJECTORY))
    if len(trajectory) < 2:
        return False
    initial, relaxed = trajectory[0], trajectory[-1]
    binders = binder_indices(info, len(framework_indices(initial)))
    return relaxation_survived(initial, relaxed, binders,
                               threshold=settings.BOND_CHANGE_THRESHOLD)


run = layout.RunLayout(settings.RUN_DIR)

if not os.path.isdir(run.relaxed):
    sys.exit("%s not found -- run step 2 first" % run.relaxed)

for species, species_dir in layout.species_dirs(run.relaxed):
    if not os.path.isfile(layout.species_info_path(run.adsorbates, species)):
        print("%s: no info.json in %s" % (species, run.adsorbates))
        continue
    info = layout.load_species_info(run.adsorbates, species)

    configs = layout.config_dirs(species_dir)
    if not configs:
        continue
    print("\n%s" % species)

    survivors_by_site, failed_by_site = {}, {}
    for site, stem, config_dir in configs:
        relaxed_xyz = os.path.join(config_dir, layout.RELAXED_STRUCTURE)
        if not os.path.isfile(relaxed_xyz) or not survived(config_dir, info):
            failed_by_site.setdefault(site, []).append(stem)
            continue

        filtered_dir = os.path.join(run.filtered, species, site)
        os.makedirs(filtered_dir, exist_ok=True)
        filtered_xyz = os.path.join(filtered_dir, stem + ".xyz")
        shutil.copy2(relaxed_xyz, filtered_xyz)

        atoms, energy = layout.read_with_energy(filtered_xyz)
        survivors_by_site.setdefault(site, []).append((stem, atoms, energy))

    for site in sorted(set(survivors_by_site) | set(failed_by_site)):
        survivors = survivors_by_site.get(site, [])
        n_total = len(survivors) + len(failed_by_site.get(site, []))
        if not survivors:
            print("  site %s: 0/%d survived" % (site, n_total))
            continue

        clusters = cluster_by_rmsd(sort_by_energy(survivors),
                                   threshold=settings.RMSD_THRESHOLD)
        print("  site %s: %d/%d survived -> %d unique"
              % (site, len(survivors), n_total, len(clusters)))

        unique_dir = os.path.join(run.unique, species, site)
        os.makedirs(unique_dir, exist_ok=True)
        for representative, members in clusters:
            stem, _, energy = representative
            shutil.copy2(os.path.join(run.filtered, species, site, stem + ".xyz"),
                         os.path.join(unique_dir, stem + ".xyz"))
            merged = "" if len(members) == 1 else \
                "  <- %s" % ", ".join(member[0] for member in members[1:])
            print("      %-24s %s%s"
                  % (stem, "E=%.3f eV" % energy if energy is not None else "no E",
                     merged))

    for tree in (run.filtered, run.unique):
        layout.copy_species_info(run.adsorbates, tree, species)
