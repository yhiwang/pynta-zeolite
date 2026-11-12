#!/usr/bin/env python
"""Rotate one chain of an adsorbate and save the frames as a .traj.

    python scripts/rotate_chain.py

Edit ADJLIST, BOND and ANGLES below. ``mol.report()`` prints every bond you
are allowed to put in BOND.
"""

import _common                      # noqa: F401   puts the repo on sys.path

from ase.io import Trajectory
from pyntaz.adjlist import AdjacencyStructure

# X-H-C-C-C-C : butane held on a site through one of its hydrogens
ADJLIST = """
1  C u0 p0 c0 {2,S}
2  H u0 p0 c0 {1,S} {3,S}
3  C u0 p0 c0 {2,S} {4,S} {7,S} {8,S}
4  C u0 p0 c0 {3,S} {5,S} {9,S} {10,S}
5  C u0 p0 c0 {4,S} {6,S} {11,S} {12,S}
6  C u0 p0 c0 {5,S} {13,S} {14,S} {15,S}
7  H u0 p0 c0 {3,S}
8  H u0 p0 c0 {3,S}
9  H u0 p0 c0 {4,S}
10 H u0 p0 c0 {4,S}
11 H u0 p0 c0 {5,S}
12 H u0 p0 c0 {5,S}
13 H u0 p0 c0 {6,S}
14 H u0 p0 c0 {6,S}
15 H u0 p0 c0 {6,S}
"""

BOND = (4, 5)                 # C4-C5: turning it swings the third carbon
ANGLES = range(0, 360, 30)    # degrees
OUT = "rotate_c4c5.traj"

mol = AdjacencyStructure.from_adjlist(ADJLIST)
print(mol.report())           # which bonds you can put in BOND

with Trajectory(OUT, "w") as traj:
    for angle in ANGLES:
        mol.build(torsions={BOND: angle})
        traj.write(mol.to_ase())        # to_ase(include_sites=True) keeps the X

print("\nwrote %d frames to %s" % (len(ANGLES), OUT))
print("view with:  ase gui %s" % OUT)