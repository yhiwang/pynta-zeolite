#!/usr/bin/env python
"""Find, per reaction, the endpoint pair whose initial + final energies are
lowest, and collect it.

    python scripts/07_select_lowest_pairs.py              # table + best_pairs.json
    python scripts/07_select_lowest_pairs.py --copy       # also copy each best pair to ts_guesses/best/<i>_rxn/

The energies are the MACE energies stored in initial.xyz / final.xyz (the
relaxed minima). Gas-phase species are the same for every pair of a reaction,
so they do not change the ranking and are left out. Because the initial and
final energies are independent, the lowest pair is simply the lowest-energy
initial config combined with the lowest-energy final config -- among the
combinations step 5 allowed (same-site or different-site pairing).
"""

import json
import os
import shutil

from _common import step_parser, layout_from


def main():
    parser = step_parser(__doc__)
    parser.add_argument("--copy", action="store_true",
                        help="copy each best pair folder to ts_guesses/best/<i>_rxn/")
    parser.add_argument("--top", type=int, default=5,
                        help="how many pairs to list per reaction (default: %(default)s)")
    args = parser.parse_args()
    layout = layout_from(args)

    from pyntaz.runtree import read_with_energy
    from pyntaz.ts_guess import reaction_dirs, pair_dirs
    from pyntaz.ts_pairs import INITIAL_XYZ, FINAL_XYZ, REACTION_INFO

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
        print("  %-10s %-12s %-12s %-12s %-26s %s"
              % ("pair", "E_initial", "E_final", "E_sum", "initial", "final"))
        for row in rows[:args.top]:
            print("  %-10s %-12.4f %-12.4f %-12.4f %-26s %s"
                  % (row["pair"], row["e_initial"], row["e_final"], row["e_sum"],
                     row["initial"], row["final"]))
        winner = rows[0]
        print("  -> %s  (E_sum %.4f eV, %.4f eV above the unconstrained minimum)"
              % (winner["pair"], winner["e_sum"],
                 winner["e_sum"] - lowest_initial["e_initial"] - lowest_final["e_final"]))
        best[name] = dict(winner, reaction=reaction["reaction"],
                          path=os.path.join(reaction_dir, winner["pair"]))

        if args.copy:
            target = os.path.join(layout.ts_guesses, "best", name)
            if os.path.isdir(target):
                shutil.rmtree(target)
            shutil.copytree(os.path.join(reaction_dir, winner["pair"]), target)
            print("  copied to %s" % target)

    out_path = os.path.join(layout.ts_guesses, "best_pairs.json")
    with open(out_path, "w") as handle:
        json.dump(best, handle, indent=2)
    print("\nwrote %s" % out_path)


if __name__ == "__main__":
    main()