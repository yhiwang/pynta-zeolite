"""Tests for the site rules of pyntaz.framework: "all" -> every first-shell
oxygen and every pair within the span; "cross" with two Al -> only cross
pairs (one O on each Al, the O-Si-O bridges) and their oxygens; one Al is
the same under both.

Run:  python3 tests/test_framework_sites.py   (from the repo root)

Builds the MOR supercell straight from data/MOR.cif with ASE so maze is
not needed; the site labels are dummies, the connectivity is real.
"""

import os
import sys

import numpy as np
from ase.io import read

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from pyntaz.framework import ZeoliteFramework  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    print("  [%s] %s %s" % ("PASS" if condition else "FAIL", name, detail))
    if not condition:
        FAILURES.append(name)


def mor_supercell():
    atoms = read(os.path.join(REPO, "data", "MOR.cif")).repeat([1, 1, 2])
    t_sites = {"T": [i for i, a in enumerate(atoms) if a.symbol == "Si"]}
    o_sites = {"O": [i for i, a in enumerate(atoms) if a.symbol == "O"]}
    return atoms, t_sites, o_sites


def framework_with_al(al_indices, site_rule):
    atoms, t_sites, o_sites = mor_supercell()
    for i in al_indices:
        atoms[i].symbol = "Al"
    return ZeoliteFramework("MOR", atoms, ["T"] * len(al_indices), al_indices,
                            [1, 1, 2], t_sites, o_sites).find_sites(3.5, site_rule)


def pair_keys(pairs):
    return [(a["indices"], b["indices"]) for a, b in pairs]


def test_single_al():
    print("single Al")
    atoms, t_sites, _ = mor_supercell()
    centre = atoms.cell.sum(axis=0) / 2
    first = min(t_sites["T"], key=lambda i: np.linalg.norm(atoms.positions[i] - centre))
    fw = framework_with_al([first], "all")
    check("four first-shell oxygens", len(fw.mono_sites) == 4, str(len(fw.mono_sites)))
    check("six pairs (every two of four)", len(fw.bi_sites) == 6, str(len(fw.bi_sites)))
    check("no pair reports a bridging Si",
          all(not fw.bridging_si(a, b) for a, b in fw.bi_sites))
    cross = framework_with_al([first], "cross")
    check("rule 'cross' is the same as 'all' for one Al",
          pair_keys(cross.bi_sites) == pair_keys(fw.bi_sites)
          and len(cross.mono_sites) == 4 and not cross.cross_only())
    return first


def test_two_al(first):
    print("two Al, second-order neighbours")
    atoms, _, _ = mor_supercell()
    probe = ZeoliteFramework("MOR", atoms, [], [], [1, 1, 2],
                             *mor_supercell()[1:])
    _, second = probe.second_order_sites(first)[0]      # nearest second-order T

    every = framework_with_al([first, second], "all")
    all_pairs = every.site_pairs(every.monodentate_sites(), 3.5)
    check("rule 'all': 8 oxygens, every pair within the span",
          len(every.mono_sites) == 8 and pair_keys(every.bi_sites) == pair_keys(all_pairs),
          "%d sites, %d pairs" % (len(every.mono_sites), len(every.bi_sites)))
    check("rule 'all' round-trips through to_dict",
          ZeoliteFramework.from_dict(every.to_dict(), every.atoms).site_rule == "all")

    fw = framework_with_al([first, second], "cross")
    cross = [(a, b) for a, b in all_pairs if fw.is_cross_pair(a, b)]
    check("some cross pairs exist within 3.5 A", len(cross) >= 1, str(len(cross)))
    check("rule 'cross': bi_sites are exactly the cross pairs",
          pair_keys(fw.bi_sites) == pair_keys(cross))
    check("rule 'cross' round-trips through to_dict",
          ZeoliteFramework.from_dict(fw.to_dict(), fw.atoms).site_rule == "cross")
    check("an explicit rule overrides the framework's own in find_pairs",
          pair_keys(fw.find_pairs(3.5, site_rule="all")) == pair_keys(all_pairs))
    try:
        ZeoliteFramework.from_dict(dict(fw.to_dict(), site_rule="bogus"), fw.atoms)
        check("unknown rule rejected", False)
    except ValueError:
        check("unknown rule rejected", True)
    # every O-Si-O bridge between the two Al (neighborhood's bridge_o) is kept
    hood = fw.neighborhood(first, second)
    bridges = {(min(o1, o2), max(o1, o2))
               for oxygens in hood["bridge_o"].values()
               for o1 in oxygens if o1 in hood["first_o"]
               for o2 in oxygens if o2 in hood["other_first_o"]}
    kept = {(min(a["indices"][0], b["indices"][0]), max(a["indices"][0], b["indices"][0]))
            for a, b in fw.bi_sites}
    check("every O-Si-O bridge is a kept pair", bridges and bridges <= kept,
          "%d bridges, %d kept" % (len(bridges), len(kept)))
    check("bridged pairs report their Si, unbridged ones do not",
          all(bool(fw.bridging_si(a, b)) == (
              (min(a["indices"][0], b["indices"][0]),
               max(a["indices"][0], b["indices"][0])) in bridges)
              for a, b in fw.bi_sites))
    used = {site["indices"] for pair in fw.bi_sites for site in pair}
    check("mono_sites are the oxygens of those pairs",
          {site["indices"] for site in fw.mono_sites} == used,
          "%d sites" % len(fw.mono_sites))
    check("find_pairs agrees with bi_sites",
          [(a["indices"], b["indices"]) for a, b in fw.find_pairs(3.5)]
          == [(a["indices"], b["indices"]) for a, b in fw.bi_sites])
    check("report mentions the rule", "cross pairs" in fw.report())
    print("  --- report ---")
    for line in fw.report().splitlines():
        print("  |", line)


if __name__ == "__main__":
    first = test_single_al()
    print()
    test_two_al(first)
    if FAILURES:
        print("\nFAILED: %d check(s): %s" % (len(FAILURES), FAILURES))
        sys.exit(1)
    print("\nall checks passed")
