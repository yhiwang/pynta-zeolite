#!/usr/bin/env python
"""Step 4 (optional) -- plot relaxation energy across the placement sweep,
one PNG per species, written into the run directory.

    python scripts/04_plot_relax_sweep.py

Reads <run_dir>/Adsorbates_relax/<species>/<site>/<stem>/relax.log and marks
the survivors found in Adsorbates_relax_filtered/ (unmarked if step 3 has
not run yet).
"""

import os
import sys

from _common import step_parser, layout_from


def main():
    parser = step_parser(__doc__)
    args = parser.parse_args()
    layout = layout_from(args)

    from pyntaz import runtree
    from pyntaz.plotting import collect_energies, collect_survivors, plot_species_sweep

    if not os.path.isdir(layout.relaxed):
        sys.exit("%s not found -- run step 2 first" % layout.relaxed)

    for species, species_dir in runtree.species_dirs(layout.relaxed):
        energies, skipped = collect_energies(species_dir)
        if not energies:
            print("%-20s no site logs (gas phase, or not yet run)" % species)
            continue
        survivors = collect_survivors(os.path.join(layout.filtered, species))

        out_path = os.path.join(layout.plots, runtree.safe_filename(species) + "_sweep.png")
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


if __name__ == "__main__":
    main()
