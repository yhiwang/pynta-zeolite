#!/usr/bin/env python
"""Step 6 -- build a TS guess for every endpoint pair.

Set RXN / PAIR to restrict to one reaction folder ("1_rxn") or one pair
("pair_0001"); a single PAIR turns on the full trace unless QUIET is True.
Writes endpoint_<role>.xyz and ts_guess.xyz into every pair folder.
"""

import os

from _common import config
from pyntaz.ts_guess import build_all_guesses

RUN_DIR = os.path.join("runs", "MOR_T2-T4_5.65")
RXN = None
PAIR = None
VERBOSE = False
QUIET = False
ROLL_TABLE = False


verbose = VERBOSE or (PAIR is not None and not QUIET)
build_all_guesses(config.RunLayout(RUN_DIR), only_reaction=RXN, only_pair=PAIR,
                  verbose=verbose, print_roll_table=ROLL_TABLE)