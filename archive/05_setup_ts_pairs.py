#!/usr/bin/env python
"""Step 5 -- pair reactant and product minima into TS endpoint folders.

For every reaction in reaction.yaml writes
<RUN_DIR>/ts_guesses/<i>_rxn/info.json and pair_NNNN/{initial,final}.xyz from
the configs in Adsorbates_relax_unique/. See pyntaz/ts_pairs.py for the
pairing rules, including the site-span filter.
"""

import os
import sys

from _common import config
from pyntaz.framework import load_framework
from pyntaz.reactions import load_reaction_set
from pyntaz.ts_pairs import build_all_pairs

RUN_DIR = os.path.join("runs", "MOR_T2-T4_5.65")
REACTIONS = config.DEFAULT_REACTIONS_FILE
MAX_SPAN = config.TS_PAIR_MAX_SPAN


layout = config.RunLayout(RUN_DIR)

if not os.path.isdir(layout.unique):
    sys.exit("%s not found -- run step 3 first" % layout.unique)

framework = load_framework(RUN_DIR)
reaction_set = load_reaction_set(REACTIONS)
build_all_pairs(reaction_set, layout, framework, MAX_SPAN)