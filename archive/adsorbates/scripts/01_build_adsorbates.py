#!/usr/bin/env python
"""Step 1 -- place every species of reaction.yaml on the saved framework.

Reads <RUN_DIR>/bare.xyz + framework.json (step 0) and writes
<RUN_DIR>/Adsorbates/<species>/<site>/<stem>/<stem>_init.xyz plus info.json.
Species folders that already exist are reused untouched.
"""

import os

import _common  # noqa: F401
import layout
import settings
from pyntaz.adsorbates import species_guesses

run = layout.RunLayout(settings.RUN_DIR)

framework = layout.load_framework(settings.RUN_DIR)
print("%s %s: %d atoms, Al at %s, %d monodentate sites, %d site pairs"
      % (framework.code, "-".join(framework.t_labels), len(framework.atoms),
         framework.al_indices, len(framework.mono_sites), len(framework.bi_sites)))
print(framework.report())

reaction_set = layout.load_reactions(settings.REACTIONS)
print("\nspecies: %s" % ", ".join(reaction_set.species))

n_written = 0
for name, mol in reaction_set.species.items():
    species_dir = os.path.join(run.adsorbates, name)
    if os.path.exists(species_dir):
        print("%s: reusing %s" % (name, species_dir))
        n_written += len(layout.initial_guess_files(species_dir))
        continue

    guesses = species_guesses(mol, framework, gas_vacuum=settings.GAS_VACUUM)

    # one line per site: how many orientations, and how far the tail clears
    if guesses.site_ids != [None]:
        print("\n%s" % name)
        for site_id in sorted(set(guesses.site_ids)):
            scores = [s for s, k in zip(guesses.scores, guesses.site_ids) if k == site_id]
            tag = guesses.tags[guesses.site_ids.index(site_id)]
            print("  %-12s %3d orientations  tail clearance %.2f-%.2f A"
                  % ("-".join("O%d" % o for o in tag["site_indices"]),
                     len(scores), min(scores), max(scores)))

    manifest = layout.write_species_guesses(run.adsorbates, name, mol, guesses,
                                            len(framework.atoms))
    n_written += len(manifest)

print("\n%d structures in %s" % (n_written, run.adsorbates))
