#!/usr/bin/env python
"""Site x site table of how many TS guesses each oxygen pair produced, one
PNG per reaction, from the info.json step 6 writes.

    <RUN_DIR>/TS_unique/<i>_rxn/pair_counts.png    kept / collected per cell

The table covers every first-shell oxygen of every Al: rows are the oxygen
seating X14, columns the oxygen seating X15, so flip0 and flip1 of one pair
fill mirror cells. X is a pair that was never tried (out of span, or
excluded by the site rule), 0/n a pair that was tried and gave nothing.
"""

import json
import os
import sys

import _common  # noqa: F401
import layout
import settings
from plotting import plot_pair_counts

run = layout.RunLayout(settings.RUN_DIR)
if not os.path.isdir(run.ts_unique):
    sys.exit("%s not found -- run step 6 first" % run.ts_unique)

# table axes: every first-shell O of every Al, Al by Al, whatever the site
# rule -- the rule only decides which cells are X
framework = layout.load_framework(settings.RUN_DIR)
oxygens = framework.monodentate_sites()
al_order = {al: k for k, al in enumerate(framework.al_indices)}
oxygens.sort(key=lambda site: (al_order[site["al_index"]], site["indices"][0]))
sites = [(site["t_label"], site["label"], site["indices"][0]) for site in oxygens]
row_of = {site["indices"][0]: k for k, site in enumerate(oxygens)}

table_title = "%s %s" % (framework.code, "-".join(framework.t_labels))
if len(framework.al_indices) > 1:
    table_title += ", site rule %s" % framework.site_rule


def pair_counts(info):
    """{(row, col): (kept, collected)} per directed oxygen pair. Every tried
    pair seeds both directions with (0, 0); each guess then lands on its
    manifest (X14 oxygen, X15 oxygen)."""
    counts = {}
    for record in info["pairs"].values():
        a, b = record["oxygens"]
        counts[(row_of[a], row_of[b])] = [0, 0]
        counts[(row_of[b], row_of[a])] = [0, 0]
        for guess in record["guesses"].values():
            first, second = guess["oxygens"]
            cell = counts[(row_of[first], row_of[second])]
            cell[1] += 1
            cell[0] += guess["kept"]
    return {key: tuple(value) for key, value in counts.items()}


for reaction_name, reaction_dir in layout.species_dirs(run.ts_unique):
    with open(os.path.join(reaction_dir, layout.REACTION_INFO)) as handle:
        info = json.load(handle)
    table_path = os.path.join(reaction_dir, layout.PAIR_TABLE)
    plot_pair_counts("%s  %s   (%s)" % (reaction_name, info["reaction"], table_title),
                     sites, pair_counts(info), table_path)
    print("%s -> %s" % (reaction_name, table_path))
