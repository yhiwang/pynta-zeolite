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

section("one build with the 'chemical' axial choice")
# leaving / arriving bond on the axis at every centre
chemical = {2: (14, 4), 3: (4, 15), 4: (3, 2)}
print("  axial =", chemical)
try:
    struct = AdjacencyStructure.from_adjlist(adjlist)
    struct.build(axial=chemical)
    print("  built OK")
    for label in (2, 3, 4, 14, 15):
        print("    atom %2d at %s" % (label, struct.positions[label].round(3)))
except AdjacencyListError as error:
    print("  AdjacencyListError:", error)

section("does every axial combination fail?")
labels = sorted(options)
failures = 0
first_error = None
for combo in product(*(options[label] for label in labels)):
    axial = dict(zip(labels, combo))
    try:
        AdjacencyStructure.from_adjlist(adjlist).build(axial=axial)
    except AdjacencyListError as error:
        failures += 1
        first_error = first_error or str(error)
print("  %d / %d combinations raise" % (failures, n_axial))
if first_error:
    print("  first error:", first_error)