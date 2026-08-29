"""Tests for pyntaz.adjlist: parser, rings, placement, torsions, X sites.

Run:  python3 test_adjlist.py
Every check prints PASS/FAIL; exits nonzero on any failure. Also writes
xyz files of every built structure into ./structures for visual inspection.
"""

import os
import sys
import numpy as np
from ase.io import write as ase_write

HERE = os.path.dirname(os.path.abspath(__file__))
# works both from the repo root (pyntaz/ next to this file) and from
# scripts/ (pyntaz/ one level up), like _common.py does
for candidate in (HERE, os.path.dirname(HERE)):
    if os.path.isdir(os.path.join(candidate, "pyntaz")):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
        break

from pyntaz.adjlist import AdjacencyStructure, AdjacencyListError

FAILURES = []
os.makedirs("structures", exist_ok=True)


def check(name, condition, detail=""):
    tag = "PASS" if condition else "FAIL"
    print("  [%s] %s %s" % (tag, name, detail))
    if not condition:
        FAILURES.append(name)


def save(mol, name, include_sites=False):
    atoms = mol.to_ase(include_sites=include_sites)
    ase_write(os.path.join("structures", name + ".xyz"), atoms)
    return atoms


def bond_errors(mol, atoms):
    """Max |actual - target| over every bond, in Angstrom."""
    labels = atoms.info["adjlist_labels"]
    index = {label: k for k, label in enumerate(labels)}
    worst = 0.0
    for a in mol.labels:
        for b in mol.neighbors[a]:
            if a < b and a in index and b in index:
                actual = np.linalg.norm(atoms.positions[index[a]]
                                        - atoms.positions[index[b]])
                worst = max(worst, abs(actual - mol.bond_length(a, b)))
    return worst


def planarity(atoms):
    """Max distance of any atom from the best-fit plane."""
    centered = atoms.positions - atoms.positions.mean(axis=0)
    return abs(centered @ np.linalg.svd(centered)[2][2]).max()


def angle(atoms, i, j, k):
    return atoms.get_angle(i, j, k)


ETHYLENE = """
1 C u0 p0 c0 {2,D} {3,S} {4,S}
2 C u0 p0 c0 {1,D} {5,S} {6,S}
3 H u0 p0 c0 {1,S}
4 H u0 p0 c0 {1,S}
5 H u0 p0 c0 {2,S}
6 H u0 p0 c0 {2,S}
"""

PROPANE = """
1 C u0 p0 c0 {2,S} {4,S} {5,S} {6,S}
2 C u0 p0 c0 {1,S} {3,S} {7,S} {8,S}
3 C u0 p0 c0 {2,S} {9,S} {10,S} {11,S}
4 H u0 p0 c0 {1,S}
5 H u0 p0 c0 {1,S}
6 H u0 p0 c0 {1,S}
7 H u0 p0 c0 {2,S}
8 H u0 p0 c0 {2,S}
9 H u0 p0 c0 {3,S}
10 H u0 p0 c0 {3,S}
11 H u0 p0 c0 {3,S}
"""

WATER = """
1 O u0 p2 c0 {2,S} {3,S}
2 H u0 p0 c0 {1,S}
3 H u0 p0 c0 {1,S}
"""

BUTADIENE = """
1 C u0 p0 c0 {2,D} {5,S} {6,S}
2 C u0 p0 c0 {1,D} {3,S} {7,S}
3 C u0 p0 c0 {2,S} {4,D} {8,S}
4 C u0 p0 c0 {3,D} {9,S} {10,S}
5 H u0 p0 c0 {1,S}
6 H u0 p0 c0 {1,S}
7 H u0 p0 c0 {2,S}
8 H u0 p0 c0 {3,S}
9 H u0 p0 c0 {4,S}
10 H u0 p0 c0 {4,S}
"""

CYCLOHEXANE = """
1 C u0 p0 c0 {2,S} {6,S} {7,S} {8,S}
2 C u0 p0 c0 {1,S} {3,S} {9,S} {10,S}
3 C u0 p0 c0 {2,S} {4,S} {11,S} {12,S}
4 C u0 p0 c0 {3,S} {5,S} {13,S} {14,S}
5 C u0 p0 c0 {4,S} {6,S} {15,S} {16,S}
6 C u0 p0 c0 {1,S} {5,S} {17,S} {18,S}
7 H u0 p0 c0 {1,S}
8 H u0 p0 c0 {1,S}
9 H u0 p0 c0 {2,S}
10 H u0 p0 c0 {2,S}
11 H u0 p0 c0 {3,S}
12 H u0 p0 c0 {3,S}
13 H u0 p0 c0 {4,S}
14 H u0 p0 c0 {4,S}
15 H u0 p0 c0 {5,S}
16 H u0 p0 c0 {5,S}
17 H u0 p0 c0 {6,S}
18 H u0 p0 c0 {6,S}
"""

CYCLOPENTANE = """
1 C u0 p0 c0 {2,S} {5,S} {6,S} {7,S}
2 C u0 p0 c0 {1,S} {3,S} {8,S} {9,S}
3 C u0 p0 c0 {2,S} {4,S} {10,S} {11,S}
4 C u0 p0 c0 {3,S} {5,S} {12,S} {13,S}
5 C u0 p0 c0 {1,S} {4,S} {14,S} {15,S}
6 H u0 p0 c0 {1,S}
7 H u0 p0 c0 {1,S}
8 H u0 p0 c0 {2,S}
9 H u0 p0 c0 {2,S}
10 H u0 p0 c0 {3,S}
11 H u0 p0 c0 {3,S}
12 H u0 p0 c0 {4,S}
13 H u0 p0 c0 {4,S}
14 H u0 p0 c0 {5,S}
15 H u0 p0 c0 {5,S}
"""

BENZENE = """
1 C u0 p0 c0 {2,B} {6,B} {7,S}
2 C u0 p0 c0 {1,B} {3,B} {8,S}
3 C u0 p0 c0 {2,B} {4,B} {9,S}
4 C u0 p0 c0 {3,B} {5,B} {10,S}
5 C u0 p0 c0 {4,B} {6,B} {11,S}
6 C u0 p0 c0 {1,B} {5,B} {12,S}
7 H u0 p0 c0 {1,S}
8 H u0 p0 c0 {2,S}
9 H u0 p0 c0 {3,S}
10 H u0 p0 c0 {4,S}
11 H u0 p0 c0 {5,S}
12 H u0 p0 c0 {6,S}
"""

# furan: heteroatom aromatic ring (edge lengths become irregular)
FURAN = """
1 O u0 p2 c0 {2,S} {5,S}
2 C u0 p0 c0 {1,S} {3,D} {6,S}
3 C u0 p0 c0 {2,D} {4,S} {7,S}
4 C u0 p0 c0 {3,S} {5,D} {8,S}
5 C u0 p0 c0 {1,S} {4,D} {9,S}
6 H u0 p0 c0 {2,S}
7 H u0 p0 c0 {3,S}
8 H u0 p0 c0 {4,S}
9 H u0 p0 c0 {5,S}
"""

NAPHTHALENE = """
1 C u0 p0 c0 {2,B} {6,B} {11,S}
2 C u0 p0 c0 {1,B} {3,B} {12,S}
3 C u0 p0 c0 {2,B} {4,B} {13,S}
4 C u0 p0 c0 {3,B} {5,B} {14,S}
5 C u0 p0 c0 {4,B} {6,B} {7,B}
6 C u0 p0 c0 {1,B} {5,B} {10,B}
7 C u0 p0 c0 {5,B} {8,B} {15,S}
8 C u0 p0 c0 {7,B} {9,B} {16,S}
9 C u0 p0 c0 {8,B} {10,B} {17,S}
10 C u0 p0 c0 {6,B} {9,B} {18,S}
11 H u0 p0 c0 {1,S}
12 H u0 p0 c0 {2,S}
13 H u0 p0 c0 {3,S}
14 H u0 p0 c0 {4,S}
15 H u0 p0 c0 {7,S}
16 H u0 p0 c0 {8,S}
17 H u0 p0 c0 {9,S}
18 H u0 p0 c0 {10,S}
"""

METHYL_X = """
1 X u0 p0 c0 {2,S}
2 C u0 p0 c0 {1,S} {3,S} {4,S} {5,S}
3 H u0 p0 c0 {2,S}
4 H u0 p0 c0 {2,S}
5 H u0 p0 c0 {2,S}
"""

# methylcyclohexane: chain BFS entering a ring system through one atom
METHYLCYCLOHEXANE = """
1 C u0 p0 c0 {2,S} {8,S} {9,S} {10,S}
2 C u0 p0 c0 {1,S} {3,S} {7,S} {11,S}
3 C u0 p0 c0 {2,S} {4,S} {12,S} {13,S}
4 C u0 p0 c0 {3,S} {5,S} {14,S} {15,S}
5 C u0 p0 c0 {4,S} {6,S} {16,S} {17,S}
6 C u0 p0 c0 {5,S} {7,S} {18,S} {19,S}
7 C u0 p0 c0 {2,S} {6,S} {20,S} {21,S}
8 H u0 p0 c0 {1,S}
9 H u0 p0 c0 {1,S}
10 H u0 p0 c0 {1,S}
11 H u0 p0 c0 {2,S}
12 H u0 p0 c0 {3,S}
13 H u0 p0 c0 {3,S}
14 H u0 p0 c0 {4,S}
15 H u0 p0 c0 {4,S}
16 H u0 p0 c0 {5,S}
17 H u0 p0 c0 {5,S}
18 H u0 p0 c0 {6,S}
19 H u0 p0 c0 {6,S}
20 H u0 p0 c0 {7,S}
21 H u0 p0 c0 {7,S}
"""

# vinyl adsorbate CH2=CH-X: locked pi plane next to a site
VINYL_X = """
1 X u0 p0 c0 {2,S}
2 C u0 p0 c0 {1,S} {3,D} {4,S}
3 C u0 p0 c0 {2,D} {5,S} {6,S}
4 H u0 p0 c0 {2,S}
5 H u0 p0 c0 {3,S}
6 H u0 p0 c0 {3,S}
"""


def test_ethylene():
    print("ethylene")
    mol = AdjacencyStructure.from_adjlist(ETHYLENE)
    check("no rings", not mol.rings)
    check("sp2 carbons", mol.hybridization(1) == "sp2"
          and mol.hybridization(2) == "sp2")
    atoms = mol.build()
    save(mol, "ethylene")
    check("6 atoms", len(atoms) == 6)
    check("bond lengths", bond_errors(mol, atoms) < 1e-6,
          "max err %.2e" % bond_errors(mol, atoms))
    check("planar", planarity(atoms) < 1e-6,
          "%.2e A off plane" % planarity(atoms))
    check("C=C ~1.32", abs(atoms.get_distance(0, 1) - 1.32) < 0.01,
          "%.3f" % atoms.get_distance(0, 1))
    check("H-C=C angle 120", abs(angle(atoms, 2, 0, 1) - 120.0) < 1.0,
          "%.1f deg" % angle(atoms, 2, 0, 1))
    check("no rotatable bonds", not mol.rotatable_bonds())


def test_propane_torsions():
    print("propane + torsions")
    mol = AdjacencyStructure.from_adjlist(PROPANE)
    rotatable = mol.rotatable_bonds()
    check("two rotatable bonds", set(rotatable) == {(1, 2), (2, 3)},
          str(sorted(rotatable)))
    atoms = mol.build()
    save(mol, "propane")
    check("bond lengths", bond_errors(mol, atoms) < 1e-6)
    check("C-C-C tetra angle", abs(angle(atoms, 0, 1, 2) - 109.47) < 1.0,
          "%.1f deg" % angle(atoms, 0, 1, 2))
    info = rotatable[(1, 2)]
    r0, r1 = info["refs"]
    labels = atoms.info["adjlist_labels"]
    ix = {label: k for k, label in enumerate(labels)}
    default_dihedral = atoms.get_dihedral(ix[r0], ix[1], ix[2], ix[r1])
    check("default anti", min(abs(default_dihedral - 180.0),
                              abs(default_dihedral + 180.0)) < 1.0,
          "%.1f deg" % default_dihedral)
    atoms60 = mol.build(torsions={(1, 2): 60.0})
    d60 = atoms60.get_dihedral(ix[r0], ix[1], ix[2], ix[r1])
    check("torsion 60 applied", min(abs(d60 - 60.0), abs(d60 - 300.0)) < 1.0,
          "%.1f deg" % d60)
    try:
        mol.build(torsions={(1, 4): 10.0})
        check("bad torsion key raises", False)
    except AdjacencyListError:
        check("bad torsion key raises", True)


def test_water():
    print("water (lone pairs)")
    mol = AdjacencyStructure.from_adjlist(WATER)
    check("O is sp3 by VSEPR", mol.hybridization(1) == "sp3")
    atoms = mol.build()
    save(mol, "water")
    check("bent, tetra angle", abs(angle(atoms, 1, 0, 2) - 109.47) < 1.0,
          "%.1f deg" % angle(atoms, 1, 0, 2))


def test_butadiene():
    print("butadiene (sp2-sp2 single bond, default s-trans)")
    mol = AdjacencyStructure.from_adjlist(BUTADIENE)
    atoms = mol.build()
    save(mol, "butadiene")
    check("bond lengths", bond_errors(mol, atoms) < 1e-6)
    check("planar", planarity(atoms) < 1e-4,
          "%.2e A off plane" % planarity(atoms))
    d = atoms.get_dihedral(0, 1, 2, 3)
    check("s-trans backbone", min(abs(d - 180.0), abs(d + 180.0)) < 1.0,
          "%.1f deg" % d)
    check("C2-C3 rotatable", (2, 3) in mol.rotatable_bonds()
          or (3, 2) in mol.rotatable_bonds())


def test_cyclohexane():
    print("cyclohexane (sp3 ring, chair pucker)")
    mol = AdjacencyStructure.from_adjlist(CYCLOHEXANE)
    check("one 6-ring", len(mol.rings) == 1 and len(mol.rings[0]) == 6)
    atoms = mol.build()
    save(mol, "cyclohexane")
    err = bond_errors(mol, atoms)
    check("bond lengths (incl. pucker comp.)", err < 0.02,
          "max err %.3f A" % err)
    ring_z = atoms.positions[:6, 2]
    check("puckered (alternating z)",
          np.std(ring_z) > 0.1 and planarity(atoms[:6]) > 0.15,
          "z spread %.2f" % np.ptp(ring_z))
    check("no rotatable bonds in ring", not mol.rotatable_bonds())


def test_cyclopentane():
    print("cyclopentane (odd sp3 ring)")
    mol = AdjacencyStructure.from_adjlist(CYCLOPENTANE)
    atoms = mol.build()
    save(mol, "cyclopentane")
    err = bond_errors(mol, atoms)
    check("ring closes", err < 0.05, "max err %.3f A" % err)


def test_benzene():
    print("benzene (aromatic ring)")
    mol = AdjacencyStructure.from_adjlist(BENZENE)
    atoms = mol.build()
    save(mol, "benzene")
    check("bond lengths", bond_errors(mol, atoms) < 1e-6)
    check("planar", planarity(atoms) < 1e-6,
          "%.2e A off plane" % planarity(atoms))
    check("aromatic C-C ~1.38",
          abs(atoms.get_distance(0, 1) - 1.383) < 0.01,
          "%.3f" % atoms.get_distance(0, 1))


def test_furan():
    print("furan (heteroatom ring, irregular polygon)")
    mol = AdjacencyStructure.from_adjlist(FURAN)
    atoms = mol.build()
    save(mol, "furan")
    check("bond lengths", bond_errors(mol, atoms) < 1e-6,
          "max err %.2e" % bond_errors(mol, atoms))
    ring = atoms[:5]
    check("ring planar", planarity(ring) < 1e-6)
    labels = atoms.info["adjlist_labels"]
    ix = {label: k for k, label in enumerate(labels)}
    co = atoms.get_distance(ix[1], ix[2])
    cc = atoms.get_distance(ix[2], ix[3])
    check("C-O shorter than C=C? (radii-based)", co != cc,
          "C-O %.3f, C=C %.3f" % (co, cc))


def test_naphthalene():
    print("naphthalene (fused aromatic rings)")
    mol = AdjacencyStructure.from_adjlist(NAPHTHALENE)
    check("two 6-rings", len(mol.rings) == 2
          and all(len(r) == 6 for r in mol.rings),
          str([len(r) for r in mol.rings]))
    check("one ring system", len(mol.ring_systems) == 1)
    atoms = mol.build()
    save(mol, "naphthalene")
    err = bond_errors(mol, atoms)
    check("bond lengths", err < 0.05, "max err %.3f A" % err)
    check("planar (aromatic fusion)", planarity(atoms) < 0.01,
          "%.3f A off plane" % planarity(atoms))
    # no atom pair should collide
    d = atoms.get_all_distances()
    np.fill_diagonal(d, 10.0)
    check("no clashes", d.min() > 1.0, "min pair %.2f A" % d.min())


def test_methyl_x():
    print("CH3-X adsorbate (site anchor)")
    mol = AdjacencyStructure.from_adjlist(METHYL_X)
    check("X detected", mol.site_labels == [1])
    atoms = mol.build()
    save(mol, "methyl_x")
    save(mol, "methyl_x_with_site", include_sites=True)
    check("4 real atoms", len(atoms) == 4)
    with_site = mol.to_ase(include_sites=True)
    check("5 with placeholder", len(with_site) == 5
          and "X" in with_site.get_chemical_symbols())
    anchor = mol.site_anchors[1]
    c_pos = mol.positions[2]
    dist = np.linalg.norm(anchor["position"] - c_pos)
    check("X-C at site bond length", abs(dist - 2.0) < 1e-6,
          "%.3f A" % dist)
    check("anchor bonded_to", anchor["bonded_to"] == [2])
    # binding direction: X->C vector vs C's H's roughly on far side
    x_to_c = c_pos - anchor["position"]
    h_mean = np.mean([mol.positions[h] for h in (3, 4, 5)], axis=0) - c_pos
    cosine = np.dot(x_to_c, h_mean) / (np.linalg.norm(x_to_c)
                                       * np.linalg.norm(h_mean))
    check("H umbrella points away from X", cosine > 0.9, "cos %.2f" % cosine)


def test_vinyl_x():
    print("CH2=CH-X (pi system on a site)")
    mol = AdjacencyStructure.from_adjlist(VINYL_X)
    atoms = mol.build()
    save(mol, "vinyl_x", include_sites=False)
    check("5 real atoms", len(atoms) == 5)
    check("vinyl group planar", planarity(atoms) < 1e-4,
          "%.2e" % planarity(atoms))


def test_methylcyclohexane():
    print("methylcyclohexane (chain into ring system)")
    mol = AdjacencyStructure.from_adjlist(METHYLCYCLOHEXANE)
    atoms = mol.build()
    save(mol, "methylcyclohexane")
    err = bond_errors(mol, atoms)
    check("bond lengths", err < 0.02, "max err %.3f A" % err)
    rotatable = mol.rotatable_bonds()
    check("methyl-ring bond rotatable",
          (2, 1) in rotatable or (1, 2) in rotatable, str(sorted(rotatable)))
    d = atoms.get_all_distances()
    np.fill_diagonal(d, 10.0)
    check("no clashes", d.min() > 0.9, "min pair %.2f A" % d.min())


def test_report():
    print("report()")
    mol = AdjacencyStructure.from_adjlist(METHYLCYCLOHEXANE)
    text = mol.report()
    check("mentions ring", "ring" in text)
    check("mentions rotatable", "rotatable" in text)
    print("  --- sample report ---")
    for line in text.splitlines():
        print("  |", line)


if __name__ == "__main__":
    for test in (test_ethylene, test_propane_torsions, test_water,
                 test_butadiene, test_cyclohexane, test_cyclopentane,
                 test_benzene, test_furan, test_naphthalene,
                 test_methyl_x, test_vinyl_x, test_methylcyclohexane,
                 test_report):
        test()
        print()
    if FAILURES:
        print("FAILED: %d check(s): %s" % (len(FAILURES), FAILURES))
        sys.exit(1)
    print("all checks passed")