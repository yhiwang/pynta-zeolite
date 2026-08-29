#!/usr/bin/env python
"""Step 3 -- keep the relaxed configs that held together, then deduplicate.

    python scripts/03_filter_relax.py

Survivors (every adsorbate bond and the binder-framework bond changed by less
than BOND_CHANGE_THRESHOLD) are copied to
<run_dir>/Adsorbates_relax_filtered/<species>/<site>/<stem>.xyz; within each
(species, site) they are clustered by adsorbate RMSD and the lowest-energy
member of each cluster goes to Adsorbates_relax_unique/ in the same layout.
"""

import os
import shutil
import sys

from _common import step_parser, layout_from, config


def main():
    parser = step_parser(__doc__)
    args = parser.parse_args()
    layout = layout_from(args)

    from pyntaz import runtree
    from pyntaz.filtering import config_survived, sort_by_energy, cluster_by_rmsd

    if not os.path.isdir(layout.relaxed):
        sys.exit("%s not found -- run step 2 first" % layout.relaxed)

    for species, species_dir in runtree.species_dirs(layout.relaxed):
        if not os.path.isfile(runtree.species_info_path(layout.adsorbates, species)):
            print("%s: no info.json in %s" % (species, layout.adsorbates))
            continue
        info = runtree.load_species_info(layout.adsorbates, species)

        configs = runtree.config_dirs(species_dir)
        if not configs:
            continue
        print("\n%s" % species)

        survivors_by_site, failed_by_site = {}, {}
        for site, stem, config_dir in configs:
            relaxed_xyz = os.path.join(config_dir, config.RELAXED_STRUCTURE)
            # a running job has a traj but no final structure yet -- it is
            # written only after the optimizer returns
            if not os.path.isfile(relaxed_xyz) or not config_survived(config_dir, info):
                failed_by_site.setdefault(site, []).append(stem)
                continue

            filtered_dir = os.path.join(layout.filtered, species, site)
            os.makedirs(filtered_dir, exist_ok=True)
            filtered_xyz = os.path.join(filtered_dir, stem + ".xyz")
            shutil.copy2(relaxed_xyz, filtered_xyz)

            atoms, energy = runtree.read_with_energy(filtered_xyz)
            survivors_by_site.setdefault(site, []).append((stem, atoms, energy))

        for site in sorted(set(survivors_by_site) | set(failed_by_site)):
            survivors = survivors_by_site.get(site, [])
            n_total = len(survivors) + len(failed_by_site.get(site, []))
            if not survivors:
                print("  site %s: 0/%d survived" % (site, n_total))
                continue

            clusters = cluster_by_rmsd(sort_by_energy(survivors))
            print("  site %s: %d/%d survived -> %d unique"
                  % (site, len(survivors), n_total, len(clusters)))

            unique_dir = os.path.join(layout.unique, species, site)
            os.makedirs(unique_dir, exist_ok=True)
            for representative, members in clusters:
                stem, _, energy = representative
                shutil.copy2(os.path.join(layout.filtered, species, site, stem + ".xyz"),
                             os.path.join(unique_dir, stem + ".xyz"))
                merged = "" if len(members) == 1 else \
                    "  <- %s" % ", ".join(member[0] for member in members[1:])
                print("      %-24s %s%s"
                      % (stem, "E=%.3f eV" % energy if energy is not None else "no E",
                         merged))

        for tree in (layout.filtered, layout.unique):
            runtree.copy_species_info(layout.adsorbates, tree, species)


if __name__ == "__main__":
    main()
