#!/usr/bin/env python
"""Place one adsorbate, given as an adjacency list, on the saved framework --
a quick way to test a single species without editing reaction.yaml.

Set ADJLIST_FILE to a file holding an RMG adjacency list, or leave it None
to use the example below (ethylene bridging two framework oxygens).
"""

import os

from _common import config
from pyntaz.pynta_mol import get_name
from pyntaz.framework import load_framework
from pyntaz.reactions import molecule_from_adjlist
from pyntaz.adsorbates import write_species_guesses

RUN_DIR = os.path.join("runs", "MOR_T2-T4_5.65")
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

framework = load_framework(RUN_DIR)
n_binders = len(mol.get_adatoms())
print("%s  %d adatoms -> %s"
      % (name, n_binders, ["gas", "monodentate", "bidentate"][n_binders]))
print("%d single sites, %d pairs\n"
      % (len(framework.mono_sites), len(framework.bi_sites)))

xyz_paths = write_species_guesses(mol, name, RUN_DIR, framework)
print("\n%d structures" % len(xyz_paths))