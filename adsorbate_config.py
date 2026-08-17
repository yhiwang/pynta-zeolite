#!/usr/bin/env python
"""
Place one adsorbate, given as an adjacency list, on the bare zeolite.

    python adsorbate_config.py

No reaction.yaml and no PyntaZ: the adjacency list below is parsed straight
into a Molecule and handed to construct_initial_guess_files, which does the
rest. Use this to test a single species without editing the reaction set.
"""

from molecule.molecule import Molecule
from pynta.mol import get_name
from zeolite_bare import make_zeolite_bare
from adsorbate import construct_initial_guess_files

# no *N labels: those mark the atoms a reaction family acts on, and nothing
# below reads them. The leading integer is the atom id {2,S} points at.
ADJLIST = """
multiplicity 1
1  C u0 p0 c0 {2,S} {5,S} {6,S} {7,S}
2  C u0 p0 c0 {1,S} {3,D} {8,S}
3  C u0 p0 c0 {2,D} {4,S} {9,S}
4  C u0 p0 c0 {3,S} {10,S} {11,S} {13,S}
5  H u0 p0 c0 {1,S} {12,S}
6  H u0 p0 c0 {1,S}
7  H u0 p0 c0 {1,S}
8  H u0 p0 c0 {2,S}
9  H u0 p0 c0 {3,S}
10 H u0 p0 c0 {4,S}
11 H u0 p0 c0 {4,S}
12 X u0 p0 c0 {5,S}
13 X u0 p0 c0 {4,S}
"""

CODE = "MOR"
LABEL = "T4"
PATH = "test_run"

mol = Molecule().from_adjacency_list(ADJLIST)
mol.multiplicity = mol.get_radical_count() + 1
name = get_name(mol)

bare = make_zeolite_bare(CODE, LABEL)
bare.analyze_zeolite()

n = len(mol.get_adatoms())
print("%s  %d adatoms -> %s" % (name, n, ["gas", "monodentate", "bidentate"][n]))
print("%d single sites, %d pairs\n"
      % (len(bare.single_sites_lists), len(bare.double_sites_lists)))

xyzs = construct_initial_guess_files(
    mol, name, PATH, bare.atoms,
    bare.single_sites_lists, bare.double_sites_lists)

print("\n%d structures" % len(xyzs))