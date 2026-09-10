#!/usr/bin/env python
"""Step 4 (optional) -- plot relaxation energy across the placement sweep,
one PNG per species, written into the run directory.

Reads <RUN_DIR>/Adsorbates_relax/<species>/<site>/<stem>/relax.log and marks
the survivors found in Adsorbates_relax_filtered/ (unmarked if step 3 has
not run yet).
"""

import os
import sys

import _common  # noqa: F401
import layout
import settings
from pyntaz.plotting import parse_optimizer_log, plot_species_sweep


def collect_energies(species_relax_dir):
    """({site: {stem: (e_initial, e_relaxed)}}, [unparsable logs])."""
    energies, skipped = {}, []
    for site, site_path in layout.site_dirs(species_relax_dir):
        for stem in sorted(os.listdir(site_path)):
            log = os.path.join(site_path, stem, layout.RELAX_LOG)
            if not os.path.isfile(log):
                continue
            with open(log) as handle:
                e_initial, e_relaxed = parse_optimizer_log(handle)
            if e_initial is None:
                skipped.append(log)
                continue
            energies.setdefault(site, {})[stem] = (e_initial, e_relaxed)
    return energies, skipped


def collect_survivors(species_filtered_dir):
    """{site: {stem}} that survived filtering; empty when the filter has not
    run yet, which just leaves the plots unmarked."""
    if not os.path.isdir(species_filtered_dir):
        return {}
    return {site: layout.stems_in(site_path)
            for site, site_path in layout.site_dirs(species_filtered_dir)}


run = layout.RunLayout(settings.RUN_DIR)

if not os.path.isdir(run.relaxed):
    sys.exit("%s not found -- run step 2 first" % run.relaxed)

for species, species_dir in layout.species_dirs(run.relaxed):
    energies, skipped = collect_energies(species_dir)
    if not energies:
        print("%-20s no site logs (gas phase, or not yet run)" % species)
        continue
    survivors = collect_survivors(os.path.join(run.filtered, species))

    out_path = os.path.join(run.plots, layout.safe_filename(species) + "_sweep.png")
    kind = plot_species_sweep(species, energies, survivors, out_path)
    if kind == "gas":
        print("%-20s gas phase, nothing to plot" % species)
        continue
    if kind is None:
        print("%-20s mixed or unrecognized stems, skipped" % species)
        continue

    n_configs = sum(len(stems) for stems in energies.values())
    n_survived = sum(len(stems) for stems in survivors.values())
    print("%-20s %d sites, %d configs, %d survived -> %s"
          % (species, len(energies), n_configs, n_survived, out_path))
    for log in skipped:
        print("    unparsable: %s" % log)
