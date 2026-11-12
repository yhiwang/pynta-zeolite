#!/usr/bin/env python
"""Place one adsorbate, given as an adjacency list, on the saved framework --
a quick way to test a single species without editing reaction.yaml. Writes
into <RUN_DIR>/Adsorbates/ exactly like step 1 (an existing species folder
is overwritten).

Set ADJLIST_FILE to a file holding an RMG adjacency list, or leave it None
to use the example below (ethylene bridging two framework oxygens).
"""

import _common  # noqa: F401
import layout
import settings
from pyntaz.adsorbates import species_guesses
from pyntaz.pynta_mol import get_name
from pyntaz.reactions import molecule_from_adjlist

ADJLIST_FILE = None

EXAMPLE_ADJLIST = """
multiplicity 1
1 C u0 p0 c0 {2,S} {3,S} {4,S} {7,S}
2 C u0 p0 c0 {1,S} {5,S} {6,S} {8,S}
3 H u0 p0 c0 {1,S}
4 H u0 p0 c0 {1,S}
5 H u0 p0 c0 {2,S}
6 H u0 p0 c0 {2,S}
7 X u0 p0 c0 {1,S}
8 X u0 p0 c0 {2,S}
"""

adjlist = EXAMPLE_ADJLIST if ADJLIST_FILE is None else open(ADJLIST_FILE).read()
mol = molecule_from_adjlist(adjlist)
name = get_name(mol)

framework = layout.load_framework(settings.RUN_DIR)
n_binders = len(mol.get_adatoms())
print("%s  %d adatoms -> %s"
      % (name, n_binders, ["gas", "monodentate", "bidentate"][n_binders]))
print("%d single sites, %d pairs\n"
      % (len(framework.mono_sites), len(framework.bi_sites)))

run = layout.RunLayout(settings.RUN_DIR)
guesses = species_guesses(mol, framework, gas_vacuum=settings.GAS_VACUUM)
manifest = layout.write_species_guesses(run.adsorbates, name, mol, guesses,
                                        len(framework.atoms))
print("%d structures in %s" % (len(manifest), run.adsorbates))
