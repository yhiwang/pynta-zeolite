#!/usr/bin/env python
"""Step 1 -- place every species of reaction.yaml on the saved framework.

Reads <RUN_DIR>/bare.xyz + framework.json (step 0) and writes
<RUN_DIR>/Adsorbates/<species>/<site>/<stem>/<stem>_init.xyz plus info.json.
Species folders that already exist are reused untouched.
"""

import os

from _common import config
from pyntaz.framework import load_framework
from pyntaz.reactions import load_reaction_set
from pyntaz.adsorbates import write_all_guesses

RUN_DIR = os.path.join("runs", "MOR_T2-T4_5.65")
REACTIONS = config.DEFAULT_REACTIONS_FILE


framework = load_framework(RUN_DIR)
print("%s %s: %d atoms, Al at %s, %d monodentate sites, %d site pairs"
      % (framework.code, "-".join(framework.t_labels), len(framework.atoms),
         framework.al_indices, len(framework.mono_sites), len(framework.bi_sites)))
framework.describe()

reaction_set = load_reaction_set(REACTIONS)
print("\nspecies: %s" % ", ".join(reaction_set.species))

written = write_all_guesses(reaction_set, RUN_DIR, framework)
print("\n%d structures in %s"
      % (sum(len(paths) for paths in written.values()),
         config.RunLayout(RUN_DIR).adsorbates))