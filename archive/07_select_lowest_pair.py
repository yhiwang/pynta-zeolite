#!/usr/bin/env python
"""Step 7 -- find, per reaction, the endpoint pair whose initial + final
energies are lowest, and write ts_guesses/best_pairs.json. COPY = True also
copies each best pair to ts_guesses/best/<i>_rxn/.
"""

import json
import os
import shutil

from _common import config
from pyntaz.runtree import read_with_energy
from pyntaz.ts_guess import reaction_dirs, pair_dirs
from pyntaz.ts_pairs import INITIAL_XYZ, FINAL_XYZ, REACTION_INFO

RUN_DIR = os.path.join("runs", "MOR_T2-T4_5.65")
COPY = False
TOP = 5


layout = config.RunLayout(RUN_DIR)
best = {}
for name in reaction_dirs(layout.ts_guesses):
    reaction_dir = os.path.join(layout.ts_guesses, name)
    with open(os.path.join(reaction_dir, REACTION_INFO)) as handle:
        reaction = json.load(handle)

    rows = []
    for pair in pair_dirs(reaction_dir):
        pair_dir = os.path.join(reaction_dir, pair)
        _, e_initial = read_with_energy(os.path.join(pair_dir, INITIAL_XYZ))
        _, e_final = read_with_energy(os.path.join(pair_dir, FINAL_XYZ))
        if e_initial is None or e_final is None:
            continue
        endpoints = reaction["pairs"][pair]
        rows.append({"pair": pair, "e_initial": e_initial, "e_final": e_final,
                     "e_sum": e_initial + e_final,
                     "span": endpoints["span"],
                     "initial": "%(species)s %(site)s/%(stem)s" % endpoints["initial"],
                     "final": "%(species)s %(site)s/%(stem)s" % endpoints["final"]})
    if not rows:
        print("\n[%d] %s   no energies found" % (reaction["index"], reaction["reaction"]))
        continue

    rows.sort(key=lambda row: row["e_sum"])
    lowest_initial = min(rows, key=lambda row: row["e_initial"])
    lowest_final = min(rows, key=lambda row: row["e_final"])

    print("\n[%d] %s   %d pairs" % (reaction["index"], reaction["reaction"], len(rows)))
    print("  lowest initial anywhere: %.4f eV (%s)   lowest final anywhere: %.4f eV (%s)"
          % (lowest_initial["e_initial"], lowest_initial["initial"],
             lowest_final["e_final"], lowest_final["final"]))
    print("  %-10s %-12s %-12s %-12s %-6s %-26s %s"
          % ("pair", "E_initial", "E_final", "E_sum", "span", "initial", "final"))
    for row in rows[:TOP]:
        print("  %-10s %-12.4f %-12.4f %-12.4f %-6.2f %-26s %s"
              % (row["pair"], row["e_initial"], row["e_final"], row["e_sum"],
                 row["span"], row["initial"], row["final"]))
    winner = rows[0]
    print("  -> %s  (E_sum %.4f eV, %.4f eV above the unconstrained minimum)"
          % (winner["pair"], winner["e_sum"],
             winner["e_sum"] - lowest_initial["e_initial"] - lowest_final["e_final"]))
    best[name] = dict(winner, reaction=reaction["reaction"],
                      path=os.path.join(reaction_dir, winner["pair"]))

    if COPY:
        target = os.path.join(layout.ts_guesses, "best", name)
        if os.path.isdir(target):
            shutil.rmtree(target)
        shutil.copytree(os.path.join(reaction_dir, winner["pair"]), target)
        print("  copied to %s" % target)

out_path = os.path.join(layout.ts_guesses, "best_pairs.json")
with open(out_path, "w") as handle:
    json.dump(best, handle, indent=2)
print("\nwrote %s" % out_path)