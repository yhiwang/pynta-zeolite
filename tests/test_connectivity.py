"""Tests for pyntaz.connectivity: read_graph, connectivity_problems, matching_maps.

Run:  python3 tests/test_connectivity.py   (from the repo root)
Every check prints PASS/FAIL; exits nonzero on any failure.

The structure is the reactant of reaction 0 (ethylene + H on a seated O) on
a fake 5-atom framework, with the adsorbate atoms in a shuffled order, so the
search has to find which atom is which.
"""

import os
import sys

from ase import Atoms

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)          # tests/ sits directly under the repo
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from pyntaz.connectivity import connectivity_problems, matching_maps, read_graph

FAILURES = []
CUTOFF = 1.2


def check(name, condition, detail=""):
    tag = "PASS" if condition else "FAIL"
    print("  [%s] %s %s" % (tag, name, detail))
    if not condition:
        FAILURES.append(name)


REACTANT = """multiplicity 1
1 *1 C u0 p0 c0 {2,D} {3,S} {4,S}
2 *2 C u0 p0 c0 {1,D} {5,S} {6,S}
3 H u0 p0 c0 {1,S}
4 H u0 p0 c0 {1,S}
5 H u0 p0 c0 {2,S}
6 H u0 p0 c0 {2,S}
7 *3 H u0 p0 c0 {8,S}
8 *4 X u0 p0 c0 {7,S}
9 *5 X u0 p0 c0
"""

# fake framework 0 Si, 1 O (site of X8), 2 Al, 3 O (site of X9), 4 Si,
# then the adsorbate in a shuffled order
POSITIONS = {"Si0": (-1.5, 0, -3.6), "O1": (0, 0, -2.97), "Al2": (1.2, 0, -4.0),
             "O3": (2.5, 0, -2.5), "Si4": (4.0, 0, -2.6),
             "C1": (0, 0, 0), "C2": (1.34, 0, 0), "H3": (-0.55, 0.93, 0),
             "H4": (-0.55, -0.93, 0), "H5": (1.89, 0.93, 0), "H6": (1.89, -0.93, 0),
             "H7": (0, 0, -2.0)}
ORDER = ["Si0", "O1", "Al2", "O3", "Si4", "H6", "C2", "H3", "H7", "C1", "H5", "H4"]
FRAME = [0, 1, 2, 3, 4]
SITES = {8: 1, 9: 3}
TRUTH = {1: ORDER.index("C1"), 2: ORDER.index("C2"), 3: ORDER.index("H3"),
         4: ORDER.index("H4"), 5: ORDER.index("H5"), 6: ORDER.index("H6"),
         7: ORDER.index("H7"), 8: 1, 9: 3}


def structure():
    atoms = Atoms([name.rstrip("0123456789") for name in ORDER],
                  positions=[POSITIONS[name] for name in ORDER],
                  cell=[20, 20, 20], pbc=True)
    atoms.positions += 10
    return atoms


def kinds(problems):
    return sorted((p["kind"], p["labels"]) for p in problems)


def test_read_graph():
    print("read_graph")
    graph = read_graph(REACTANT)
    check("9 labels", len(graph["elements"]) == 9)
    check("X labels 8 and 9", [l for l, e in graph["elements"].items() if e == "X"] == [8, 9])
    check("6 bonds", sorted(graph["bonds"]) == [(1, 2), (1, 3), (1, 4), (2, 5), (2, 6), (7, 8)],
          str(sorted(graph["bonds"])))


def test_identify():
    print("identify: only the X labels given")
    graph = read_graph(REACTANT)
    maps = matching_maps(structure(), graph, FRAME, SITES, True, CUTOFF)
    check("8 maps (C1/C2, H3/H4, H5/H6 can swap)", len(maps) == 8, "%d maps" % len(maps))
    check("true map is one of them", TRUTH in maps)


def test_check_correct():
    print("check: correct full map")
    graph = read_graph(REACTANT)
    check("matching_maps gives [map]",
          matching_maps(structure(), graph, FRAME, TRUTH, True, CUTOFF) == [TRUTH])
    problems = connectivity_problems(structure(), graph, FRAME, TRUTH, True, CUTOFF)
    check("no problems", problems == [], str(problems))


def test_check_swapped():
    print("check: C1 and C2 swapped")
    graph = read_graph(REACTANT)
    wrong = dict(TRUTH)
    wrong[1], wrong[2] = TRUTH[2], TRUTH[1]
    check("no map", matching_maps(structure(), graph, FRAME, wrong, True, CUTOFF) == [])
    problems = connectivity_problems(structure(), graph, FRAME, wrong, True, CUTOFF)
    check("4 missing", sum(p["kind"] == "missing" for p in problems) == 4)
    check("4 extra", sum(p["kind"] == "extra" for p in problems) == 4)


def test_element():
    print("check: H3 label on the C1 atom")
    graph = read_graph(REACTANT)
    bad = dict(TRUTH)
    bad[3], bad[1] = TRUTH[1], TRUTH[3]
    problems = connectivity_problems(structure(), graph, FRAME, bad, False, CUTOFF)
    elements = [p for p in problems if p["kind"] == "element"]
    check("element problems for labels 1 and 3",
          sorted(p["labels"] for p in elements) == [(1,), (3,)], str(elements))


def test_second_molecule():
    print("a second molecule (H2) far away")
    graph = read_graph(REACTANT)
    atoms = structure() + Atoms("H2", positions=[(15, 15, 15), (15, 15, 15.74)])
    maps = matching_maps(atoms, graph, FRAME, SITES, True, CUTOFF)
    check("still 8 maps (bonds of other atoms not checked)", len(maps) == 8, "%d maps" % len(maps))


def test_extra_bond():
    print("an extra H bonded to C1")
    graph = read_graph(REACTANT)
    atoms = structure()
    atoms += Atoms("H", positions=[atoms.positions[TRUTH[1]] + (0, 0, 1.05)])
    strict = matching_maps(atoms, graph, FRAME, SITES, True, CUTOFF)
    easy = matching_maps(atoms, graph, FRAME, SITES, False, CUTOFF)
    check("strict: no map", strict == [], "%d maps" % len(strict))
    check("easy: maps found", len(easy) > 0, "%d maps" % len(easy))
    problems = connectivity_problems(atoms, graph, FRAME, TRUTH, True, CUTOFF)
    check("extra C1 - new H", kinds(problems) == [("extra", (1, None))], str(kinds(problems)))


def test_h_on_other_site():
    print("H7 moved onto the other seated O (X9)")
    graph = read_graph(REACTANT)
    atoms = structure()
    atoms.positions[TRUTH[7]] = atoms.positions[3] + (0, 0, 0.97)
    check("no map", matching_maps(atoms, graph, FRAME, SITES, True, CUTOFF) == [])
    problems = connectivity_problems(atoms, graph, FRAME, TRUTH, True, CUTOFF)
    check("missing 7-8, extra 9-7",
          kinds(problems) == [("extra", (9, 7)), ("missing", (7, 8))], str(kinds(problems)))


def test_missing_x():
    print("X label not in the map")
    graph = read_graph(REACTANT)
    try:
        matching_maps(structure(), graph, FRAME, {8: 1}, True, CUTOFF)
        check("ValueError", False)
    except ValueError as error:
        check("ValueError", True, str(error))


if __name__ == "__main__":
    for test in (test_read_graph, test_identify, test_check_correct, test_check_swapped,
                 test_element, test_second_molecule, test_extra_bond, test_h_on_other_site,
                 test_missing_x):
        test()
        print()
    if FAILURES:
        print("FAILED: %d check(s): %s" % (len(FAILURES), FAILURES))
        sys.exit(1)
    print("all checks passed")