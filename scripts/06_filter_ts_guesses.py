#!/usr/bin/env python
"""Step 6 -- filter the TS guesses step 5 wrote down to distinct ones, and
table how many each site pair produced.

Reads every <RUN_DIR>/TS_guesses/<i>_rxn/pair_<k>/<stem>/<stem>_init.xyz
together with its manifest record from the reaction's info.json. Within
each pair and each direction (flip0 / flip1 are never merged), guesses
closer than settings.TS_RMSD_THRESHOLD in adsorbate RMSD are one guess,
and the roomiest member (best clearance) stands for the cluster. Keepers
are laid out in the same shape under

    <RUN_DIR>/TS_unique/<i>_rxn/info.json          manifest + kept / merged
    <RUN_DIR>/TS_unique/<i>_rxn/pair_counts.png    site x site kept/collected
    <RUN_DIR>/TS_unique/<i>_rxn/pair_<k>/<stem>/<stem>_init.xyz

so the relaxation submitter can be pointed at either tree.

pair_counts.png covers every first-shell oxygen of every Al: rows are the
oxygen seating X14, columns the oxygen seating X15, so flip0 and flip1 of
one pair fill mirror cells. X is a pair that was never tried (out of span,
or excluded by the site rule), 0/n a pair that was tried and gave nothing.

settings.TS_KEEP_PER_DIRECTION, when set, caps the clusters kept per pair
and direction (1 = only the roomiest guess of each direction survives).
"""

import json
import os
import shutil
import sys

import _common  # noqa: F401
import layout
import settings
from ase.io import read
from pyntaz.filtering import cluster_by_rmsd
from pyntaz.plotting import plot_pair_counts

run = layout.RunLayout(settings.RUN_DIR)

if not os.path.isdir(run.ts_guesses):
    sys.exit("%s not found -- run step 5 first" % run.ts_guesses)

# --------------------------------------------------------------------------
# table axes: every first-shell O of every Al, Al by Al, whatever the site
# rule -- the rule only decides which cells are X
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# collect, filter, copy, table -- one reaction at a time
# --------------------------------------------------------------------------

n_collected = n_kept = 0
for reaction_name, reaction_dir in layout.species_dirs(run.ts_guesses):
    with open(os.path.join(reaction_dir, layout.REACTION_INFO)) as handle:
        info = json.load(handle)
    print("\n%s  %s" % (reaction_name, info["reaction"]))

    out_reaction_dir = os.path.join(run.ts_unique, reaction_name)
    os.makedirs(out_reaction_dir, exist_ok=True)

    # (stem, atoms, -clearance) per pair -- the tuple shape cluster_by_rmsd
    # expects, sorted so the roomiest guess leads and represents its cluster
    guesses_by_pair = {pair: [] for pair in info["pairs"]}
    for relative, xyz in layout.initial_guess_files(reaction_dir):
        pair, stem = relative.split(os.sep)
        clearance = info["pairs"][pair]["guesses"][stem]["clearance"]
        guesses_by_pair[pair].append((stem, read(xyz), -clearance))

    for pair, guesses in guesses_by_pair.items():
        record = info["pairs"][pair]
        for guess in record["guesses"].values():
            guess["kept"] = False
            guess["merged"] = []
        if not guesses:
            print("  %s  %-7s %-7s %.2f A   0 guesses"
                  % (pair, record["sites"][0], record["sites"][1], record["span"]))
            continue

        summary = []
        merges = []
        for flip in (0, 1):
            entries = sorted((entry for entry in guesses
                              if record["guesses"][entry[0]]["flip"] == flip),
                             key=lambda entry: entry[2])
            if not entries:
                continue
            clusters = cluster_by_rmsd(entries, settings.TS_RMSD_THRESHOLD)
            if settings.TS_KEEP_PER_DIRECTION:
                clusters = clusters[:settings.TS_KEEP_PER_DIRECTION]
            for (stem, atoms, _), members in clusters:
                record["guesses"][stem]["kept"] = True
                record["guesses"][stem]["merged"] = [m[0] for m in members[1:]]
                if len(members) > 1:
                    merges.append((stem, [m[0] for m in members[1:]]))
            summary.append("flip%d %d -> %d" % (flip, len(entries), len(clusters)))

        kept = [stem for stem, guess in record["guesses"].items() if guess["kept"]]
        print("  %s  %-7s %-7s %.2f A   %s"
              % (pair, record["sites"][0], record["sites"][1], record["span"],
                 "   ".join(summary)))
        for stem, members in merges:
            print("      %s  <- %s" % (stem, ", ".join(members)))

        for stem in kept:
            config_dir = os.path.join(out_reaction_dir, pair, stem)
            os.makedirs(config_dir, exist_ok=True)
            shutil.copy2(layout.initial_guess_path(os.path.join(reaction_dir, pair, stem)),
                         layout.initial_guess_path(config_dir))
        n_collected += len(guesses)
        n_kept += len(kept)

    info["rmsd_threshold"] = settings.TS_RMSD_THRESHOLD
    info["keep_per_direction"] = settings.TS_KEEP_PER_DIRECTION
    with open(os.path.join(out_reaction_dir, layout.REACTION_INFO), "w") as handle:
        json.dump(info, handle, indent=2)

    table_path = os.path.join(out_reaction_dir, layout.PAIR_TABLE)
    plot_pair_counts("%s  %s   (%s)" % (reaction_name, info["reaction"], table_title),
                     sites, pair_counts(info), table_path)
    print("  table -> %s" % table_path)

print("\n%d of %d guesses kept in %s" % (n_kept, n_collected, run.ts_unique))