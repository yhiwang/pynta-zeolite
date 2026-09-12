#!/usr/bin/env python
"""Isolate reaction 5 (2-butoxide -> isobutoxide methyl shift) and print what
the TS graph looks like before PairSweep gets hold of it.

Run from anywhere:  python tests/test_methyl_shift_graph.py
"""

import os
import sys
from itertools import product

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from pyntaz.adjlist import AdjacencyStructure, AdjacencyListError
from pyntaz.reactions import molecule_from_adjlist
from pyntaz.ts_graph import TSGraph, axial_options, effective_torsions

REACTANT = """multiplicity 1
1  C u0 p0 c0 {2,S} {5,S} {6,S} {7,S}
2  *1 C u0 p0 c0 {1,S} {3,S} {8,S} {14,S}
3  *2 C u0 p0 c0 {2,S} {4,S} {9,S} {10,S}
4  *3 C u0 p0 c0 {3,S} {11,S} {12,S} {13,S}
5  H u0 p0 c0 {1,S}
6  H u0 p0 c0 {1,S}
7  H u0 p0 c0 {1,S}
8  H u0 p0 c0 {2,S}
9  H u0 p0 c0 {3,S}
10 H u0 p0 c0 {3,S}
11 H u0 p0 c0 {4,S}
12 H u0 p0 c0 {4,S}
13 H u0 p0 c0 {4,S}
14 *4 X u0 p0 c0 {2,S}
15 *5 X u0 p0 c0
"""

PRODUCT = """multiplicity 1
1  C u0 p0 c0 {2,S} {5,S} {6,S} {7,S}
2  *1 C u0 p0 c0 {1,S} {3,S} {4,S} {8,S}
3  *2 C u0 p0 c0 {2,S} {9,S} {10,S} {15,S}
4  *3 C u0 p0 c0 {2,S} {11,S} {12,S} {13,S}
5  H u0 p0 c0 {1,S}
6  H u0 p0 c0 {1,S}
7  H u0 p0 c0 {1,S}
8  H u0 p0 c0 {2,S}
9  H u0 p0 c0 {3,S}
10 H u0 p0 c0 {3,S}
11 H u0 p0 c0 {4,S}
12 H u0 p0 c0 {4,S}
13 H u0 p0 c0 {4,S}
14 *4 X u0 p0 c0
15 *5 X u0 p0 c0 {3,S}
"""


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


ts = TSGraph(molecule_from_adjlist(REACTANT), molecule_from_adjlist(PRODUCT))

section("bond changes (ts.report)")
print(ts.report())

section("rings of the merged graph, classified (ts.rings)")
for ring, where in ts.rings():
    print("  %-14s %s" % ("-".join(ts.tag(i) for i in ring), where))
print("  ts_rings():", ts.ts_rings())
assert ts.ts_rings() == [[1, 3, 2]] or sorted(ts.ts_rings()[0]) == [1, 2, 3], \
    "expected the C2-C3-C4 triangle to be a TS-only ring"

# sanity check: a real ring must NOT be flagged. cyclopropane -> cyclopropane
# (no bond changes at all) should come back as "both".
CYCLOPROPANE = """multiplicity 1
1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}
2 C u0 p0 c0 {1,S} {3,S} {6,S} {7,S}
3 C u0 p0 c0 {1,S} {2,S} {8,S} {9,S}
4 H u0 p0 c0 {1,S}
5 H u0 p0 c0 {1,S}
6 H u0 p0 c0 {2,S}
7 H u0 p0 c0 {2,S}
8 H u0 p0 c0 {3,S}
9 H u0 p0 c0 {3,S}
"""
real = TSGraph(molecule_from_adjlist(CYCLOPROPANE), molecule_from_adjlist(CYCLOPROPANE))
print("  cyclopropane control:", real.rings())
assert real.rings() and real.rings()[0][1] == "both"
assert real.ts_rings() == []
print("  ring classification OK")

section("merged TS adjacency list (ts.adjlist)")
adjlist = ts.adjlist()
print(adjlist)

section("coordination in the merged graph")
for i, element in enumerate(ts.elements):
    neighbors = sorted(ts.adj[i])
    flag = "  <-- five-coordinate" if len(neighbors) == 5 else ""
    print("  %-7s %d neighbours: %s%s"
          % (ts.tag(i), len(neighbors),
             " ".join(ts.tag(j) for j in neighbors), flag))
print("  sites (X):", [ts.tag(i) for i in ts.sites()])

section("what AdjacencyStructure sees")
struct = AdjacencyStructure.from_adjlist(adjlist)
print("  rings:        ", struct.rings)
print("  ring systems: ", struct.ring_systems)
print("  ring bonds:   ", sorted(struct.ring_bonds))
print("  hypervalent:  ", struct.hypervalent_atoms())
print("  rotatable:    ", list(struct.rotatable_bonds()))
print("  effective:    ", effective_torsions(struct))
options = axial_options(struct)
print("  axial options per atom:")
for label, pairs in options.items():
    print("    atom %d: %d pairs  %s" % (label, len(pairs), pairs))
n_axial = 1
for pairs in options.values():
    n_axial *= len(pairs)
print("  axial grid size: %d" % n_axial)

section("build with stretched changing bonds (1.2) and the anti rule")
import numpy as np
from pyntaz.ts_graph import build_at
ts12 = TSGraph(molecule_from_adjlist(REACTANT), molecule_from_adjlist(PRODUCT),
               stretch={"form": 1.2, "break": 1.2})
print("  bond_scales:   ", ts12.bond_scales())
print("  changing_bonds:", sorted(ts12.changing_bonds()))
survey = AdjacencyStructure.from_adjlist(ts12.adjlist(), changing=ts12.changing_bonds())
print("  axial options: ", axial_options(survey), " (ring atoms are left out)")
print("  torsions:      ", effective_torsions(survey))
site_elements = {14: "O", 15: "O"}
n = len(ts12.elements)
pos = build_at(ts12.adjlist(), site_elements, 1.0, n,
               bond_scales=ts12.bond_scales(), changing=ts12.changing_bonds())
d = lambda a, b: np.linalg.norm(pos[a - 1] - pos[b - 1])
def ang(a, b, c):
    u, v = pos[a - 1] - pos[b - 1], pos[c - 1] - pos[b - 1]
    return np.degrees(np.arccos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)))
print("  C2-C3 %.2f  C2-C4 %.2f  C3-C4 %.2f  (want ~1.5 / 1.8 / 1.8)"
      % (d(2, 3), d(2, 4), d(3, 4)))
print("  C2-X14 %.2f  C3-X15 %.2f  X14..X15 %.2f" % (d(2, 14), d(3, 15), d(14, 15)))
print("  C4-C2-X14 %.0f  C4-C3-X15 %.0f  (want 180: form and break anti)"
      % (ang(4, 2, 14), ang(4, 3, 15)))
print("  C3-C2-C1 %.0f  C3-C2-H8 %.0f" % (ang(3, 2, 1), ang(3, 2, 8)))
labels = range(1, n + 1)
closest = min((d(a, b), a, b) for a in labels for b in labels
              if a < b and (b - 1) not in ts12.adj[a - 1])
print("  closest non-bonded pair: %.2f A (%d, %d)" % closest)
assert d(2, 4) > d(2, 3) and d(3, 4) > d(2, 3), "bridge bonds should be the long ones"
assert ang(4, 2, 14) > 170 and ang(4, 3, 15) > 170
assert closest[0] > 1.5
print("  bridge geometry OK")

section("write the built TS for viewing")
from ase import Atoms
from ase.io import write
symbols = ["O" if e == "X" else e for e in ts12.elements]   # X shown as O
atoms = Atoms(symbols, positions=pos)
out = os.path.join(REPO, "tests", "ts5_bridge.xyz")
write(out, atoms)
print("  wrote", out, "(X14/X15 drawn as O)")
print("  view with:  ase gui", os.path.relpath(out, REPO))