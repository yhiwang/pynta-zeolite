#!/usr/bin/env python
"""Step 6 -- build a TS guess for every endpoint pair.

    python scripts/06_build_ts_guesses.py
    python scripts/06_build_ts_guesses.py --rxn 1_rxn --pair pair_0001   # verbose trace of one pair

Writes endpoint_<role>.xyz and ts_guess.xyz into every pair folder (see
pyntaz/ts_guess.py for the construction). Selecting a single --pair turns on
the full verbose trace unless --quiet is given.
"""

from _common import step_parser, layout_from


def main():
    parser = step_parser(__doc__)
    parser.add_argument("--rxn", help="only this reaction folder, e.g. 1_rxn")
    parser.add_argument("--pair", help="only this pair folder, e.g. pair_0001")
    parser.add_argument("--verbose", action="store_true", help="print the full trace")
    parser.add_argument("--quiet", action="store_true",
                        help="no trace even when a single --pair is selected")
    parser.add_argument("--roll-table", action="store_true",
                        help="print the clearance of every roll angle, not just the winner")
    args = parser.parse_args()

    from pyntaz.ts_guess import build_all_guesses

    verbose = args.verbose or (args.pair is not None and not args.quiet)
    build_all_guesses(layout_from(args), only_reaction=args.rxn,
                      only_pair=args.pair, verbose=verbose,
                      print_roll_table=args.roll_table)


if __name__ == "__main__":
    main()
