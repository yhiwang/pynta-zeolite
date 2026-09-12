#!/usr/bin/env python
"""Reaction 2 (butoxide beta-H elimination) end to end, on a real MOR
framework, to see why PairSweep finds nothing.

Builds the MOR supercell from data/MOR.cif with two Al (same choice as
tests/test_framework_sites.py), takes the cross O-O pair, builds the TS
graph, and for every torsion/axial build reports the X-X span the fit can
reach against the O-O distance. Then seats the closest build anyway and
writes it with the framework so it can be looked at.

Run from anywhere:  python tests/test_beta_h_elim.py
Writes tests/r2_molecule.xyz (built TS, X as O) and tests/r2_seated.xyz
(closest build seated on the pair, roll 0, inside the framework).
"""

import os
import sys
from itertools import product

import numpy as np
from ase import Atoms
from ase.io import read, write

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from pyntaz.adjlist import AdjacencyStructure
from pyntaz.framework import ZeoliteFramework
from pyntaz.reactions import molecule_from_adjlist
from pyntaz.ts_graph import (TSGraph, PairSweep, axial_options, build_at,
                             effective_torsions, fit_site_scale, seat)

STRETCH = {"form": 1.0, "break": 1.0}
SPAN_TOLERANCE = 0.2

REACTANT = """multiplicity 1
1  *1 C u0 p0 c0 {2,S} {5,S} {6,S} {14,S}
2  *2 C u0 p0 c0 {1,S} {3,S} {7,S} {8,S}
3  C u0 p0 c0 {2,S} {4,S} {9,S} {10,S}
4  C u0 p0 c0 {3,S} {11,S} {12,S} {13,S}
5  H u0 p0 c0 {1,S}
6  H u0 p0 c0 {1,S}
7  *3 H u0 p0 c0 {2,S}
8  H u0 p0 c0 {2,S}
9  H u0 p0 c0 {3,S}
10 H u0 p0 c0 {3,S}
11 H u0 p0 c0 {4,S}
12 H u0 p0 c0 {4,S}
13 H u0 p0 c0 {4,S}
14 *4 X u0 p0 c0 {1,S}
15 *5 X u0 p0 c0
"""

PRODUCT = """multiplicity 1
1  *1 C u0 p0 c0 {2,D} {5,S} {6,S}
2  *2 C u0 p0 c0 {1,D} {3,S} {8,S}
3  C u0 p0 c0 {2,S} {4,S} {9,S} {10,S}
4  C u0 p0 c0 {3,S} {11,S} {12,S} {13,S}
5  H u0 p0 c0 {1,S}
6  H u0 p0 c0 {1,S}
7  *3 H u0 p0 c0 {15,S}
8  H u0 p0 c0 {2,S}
9  H u0 p0 c0 {3,S}
10 H u0 p0 c0 {3,S}
11 H u0 p0 c0 {4,S}
12 H u0 p0 c0 {4,S}
13 H u0 p0 c0 {4,S}
14 *4 X u0 p0 c0
15 *5 X u0 p0 c0 {7,S}
"""


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def angle(a, b, c):
    u, v = a - b, c - b
    return np.degrees(np.arccos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)))


# --------------------------------------------------------------------------
# framework: MOR 1x1x2 with two Al, cross rule -> one O-O pair
# --------------------------------------------------------------------------
section("framework")
atoms = read(os.path.join(REPO, "data", "MOR.cif")).repeat([1, 1, 2])
t_sites = {"T": [i for i, a in enumerate(atoms) if a.symbol == "Si"]}
o_sites = {"O": [i for i, a in enumerate(atoms) if a.symbol == "O"]}
centre = atoms.cell.sum(axis=0) / 2
first = min(t_sites["T"], key=lambda i: np.linalg.norm(atoms.positions[i] - centre))
probe = ZeoliteFramework("MOR", atoms, [], [], [1, 1, 2], t_sites, o_sites)
_, second = probe.second_order_sites(first)[0]
for i in (first, second):
    atoms[i].symbol = "Al"
framework = ZeoliteFramework("MOR", atoms, ["T", "T"], [first, second],
                             [1, 1, 2], t_sites, o_sites).find_sites(3.5, "cross")
pairs = framework.bi_sites
print("  Al at", framework.al_indices, "-", len(pairs), "cross pair(s)")
for site_a, site_b in pairs:
    print("  %s(%d)  %s(%d)  %.2f A" % (site_a["site"], site_a["indices"][0],
                                        site_b["site"], site_b["indices"][0],
                                        framework.span(site_a, site_b)))
site_a, site_b = pairs[0]
oxygens = (site_a["indices"][0], site_b["indices"][0])
target = framework.span(site_a, site_b)

# --------------------------------------------------------------------------
# the TS graph
# --------------------------------------------------------------------------
section("TS graph")
ts = TSGraph(molecule_from_adjlist(REACTANT), molecule_from_adjlist(PRODUCT),
             stretch=STRETCH)
print(ts.report())
survey = AdjacencyStructure.from_adjlist(ts.adjlist(), changing=ts.changing_bonds())
print("  rotatable:", list(survey.rotatable_bonds()))
print("  effective:", effective_torsions(survey))
print("  axial:    ", axial_options(survey))
print("  sites:    ", [ts.tag(i) for i in ts.sites()])

# --------------------------------------------------------------------------
# one build, scale 1.0: where do the X end up?
# --------------------------------------------------------------------------
section("built molecule at site-bond scale 1.0")
site_elements = {15: "O", 14: "O"}
n = len(ts.elements)
pos = build_at(ts.adjlist(), site_elements, 1.0, n,
               bond_scales=ts.bond_scales(), changing=ts.changing_bonds())
P = lambda label: pos[label - 1]
print("  C1=C2 %.2f  C1-X14 %.2f  C2-H7 %.2f  H7-X15 %.2f"
      % tuple(np.linalg.norm(P(a) - P(b)) for a, b in ((1, 2), (1, 14), (2, 7), (7, 15))))
print("  dihedral X14-C1-C2-H7 = %.0f deg   (syn would be ~0)"
      % dihedral(P(14), P(1), P(2), P(7)))
print("  angle C2-H7-X15 = %.0f deg" % angle(P(2), P(7), P(15)))
print("  X14..X15 = %.2f A   target O-O = %.2f A" % (np.linalg.norm(P(14) - P(15)), target))
symbols = ["O" if e == "X" else e for e in ts.elements]
out = os.path.join(REPO, "tests", "r2_molecule.xyz")
write(out, Atoms(symbols, positions=pos))
print("  wrote", os.path.relpath(out, REPO))

# --------------------------------------------------------------------------
# stage one, by hand: closest span over every build and scale
# --------------------------------------------------------------------------
section("stage one: span fit against %.2f A, tolerance %.2f" % (target, SPAN_TOLERANCE))
sweep = PairSweep(ts, framework.atoms, oxygens, span_tolerance=SPAN_TOLERANCE)
print("  torsion keys %s, axial grid %d, %d builds per scale"
      % (sweep.torsion_keys, len(sweep.axial_grid), sweep.n_builds))
step = 360.0 / sweep.torsion_steps
best = None
spans = []
for combo in product(range(sweep.torsion_steps), repeat=len(sweep.torsion_keys)):
    torsions = {key: k * step for key, k in zip(sweep.torsion_keys, combo)}
    for axial_combo in sweep.axial_grid:
        axial = {label: pair for label, pair in zip(sweep.axial_labels, axial_combo)}
        scale, span, error = fit_site_scale(
            sweep.adjlist, sweep.site_elements, sweep.n_atoms,
            sweep.site_a, sweep.site_b, sweep.target_a, sweep.target_b,
            torsions, axial, sweep.bond_scales, sweep.scale_range, sweep.changing)
        spans.append(span)
        if best is None or error < best[3]:
            best = (torsions, axial, scale, error, span)
survivors = sweep.survivors()
print("  survivors within tolerance: %d" % len(survivors))
print("  closest miss %.2f A  (span %.2f at scale %.2f, torsions %s)"
      % (best[3], best[4], best[2], best[0]))
print("  fitted spans range %.2f - %.2f A" % (min(spans), max(spans)))

# --------------------------------------------------------------------------
# seat the closest build anyway so it can be looked at in the framework
# --------------------------------------------------------------------------
section("seat the closest build (roll 0) and write it")
angles = tuple(round(a) for a in best[0].values())
positions = sweep._positions(angles, best[1], best[2])
placed = seat(positions, sweep.site_a, sweep.site_b,
              sweep.target_a, sweep.target_b, 0.0)
molecule = Atoms([ts.elements[i] for i in sweep.keep], positions=placed[sweep.keep],
                 cell=framework.atoms.cell, pbc=framework.atoms.pbc)
out = os.path.join(REPO, "tests", "r2_seated.xyz")
write(out, framework.atoms + molecule)
for x, o in ((sweep.site_a, sweep.oxygen_a), (sweep.site_b, sweep.oxygen_b)):
    print("  %s sits %.2f A from its oxygen %d"
          % (ts.tag(x), np.linalg.norm(placed[x] - framework.atoms.positions[o]), o))
print("  wrote", os.path.relpath(out, REPO))
print("  view with:  ase gui tests/r2_seated.xyz")