#!/usr/bin/env python
"""Step 5 -- pair reactant and product minima into TS endpoint folders.

    python scripts/05_setup_ts_pairs.py

For every reaction in reaction.yaml writes
<run_dir>/ts_guesses/<i>_rxn/info.json and pair_NNNN/{initial,final}.xyz from
the configs in Adsorbates_relax_unique/. See pyntaz/ts_pairs.py for the
pairing rules.
"""

import os
import sys

from _common import step_parser, layout_from, config


def main():
    parser = step_parser(__doc__)
    parser.add_argument("--reactions", default=config.DEFAULT_REACTIONS_FILE,
                        help="reaction yaml (default: %(default)s)")
    args = parser.parse_args()
    layout = layout_from(args)

    from pyntaz.reactions import load_reaction_set
    from pyntaz.ts_pairs import build_all_pairs

    if not os.path.isdir(layout.unique):
        sys.exit("%s not found -- run step 3 first" % layout.unique)

    reaction_set = load_reaction_set(args.reactions)
    build_all_pairs(reaction_set, layout)


if __name__ == "__main__":
    main()
