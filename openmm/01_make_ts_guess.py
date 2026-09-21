#!/usr/bin/env python
"""One TS guess for the OpenMM test, standalone: build the framework,
seat the reaction below on one of its oxygen pairs, keep the best
clearance over both flips. Needs only the pyntaz package (sweep sizes are
its defaults). Writes, next to this file,

    ts_guess.xyz    framework + adsorbate, as step 5 would
    ts_guess.json   the index bookkeeping 02_openmm_restrain.py reads:
                    nslab, the seated oxygens, and every forming/breaking
                    bond as indices into ts_guess.xyz
"""

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import yaml
from ase.io import write
from pyntaz.framework import build_framework
from pyntaz.reactions import molecule_from_adjlist
from pyntaz.ts_graph import TSGraph, PairSweep

CODE = "MOR"
T_LABELS = ("T4",)          # one label for a single Al, two for a pair
MAX_SPAN = 3.5              # A, longest O-O distance tried
PAIR_ID = 0                 # a row of the pair table printed below
STRETCH = {"form": 1.0, "break": 1.0}
HERE = os.path.dirname(os.path.abspath(__file__))

REACTION = """
reactant: |
  multiplicity 1
  1 *1 C u0 p0 c0 {2,D} {3,S} {4,S}
  2 *2 C u0 p0 c0 {1,D} {5,S} {6,S}
  3 H u0 p0 c0 {1,S}
  4 H u0 p0 c0 {1,S}
  5 H u0 p0 c0 {2,S}
  6 H u0 p0 c0 {2,S}
  7 *3 H u0 p0 c0 {8,S}
  8 *4 X u0 p0 c0 {7,S}
  9 *5 X u0 p0 c0
product: |
  multiplicity 1
  1 *1 C u0 p0 c0 {2,S} {3,S} {4,S} {7,S}
  2 *2 C u0 p0 c0 {1,S} {5,S} {6,S} {9,S}
  3 H u0 p0 c0 {1,S}
  4 H u0 p0 c0 {1,S}
  5 H u0 p0 c0 {2,S}
  6 H u0 p0 c0 {2,S}
  7 *3 H u0 p0 c0 {1,S}
  8 *4 X u0 p0 c0
  9 *5 X u0 p0 c0 {2,S}
reaction: C=C + [H][Pt] <=> CC[Pt]
reaction_family: Surface_Protonation
"""

# -- framework and the pair ------------------------------------------------
framework = build_framework(CODE, T_LABELS, max_span=MAX_SPAN)
nslab = len(framework.atoms)
print(framework.report())
pairs = framework.find_pairs(MAX_SPAN)
print("\n%s %s: %d atoms, %d oxygen pairs within %.2f A"
      % (CODE, "-".join(T_LABELS), nslab, len(pairs), MAX_SPAN))
for pair_id, (site_a, site_b) in enumerate(pairs):
    print("  pair_%02d%s  %-7s O%-4d %-7s O%-4d %.2f A"
          % (pair_id, " *" if pair_id == PAIR_ID else "  ",
             site_a["site"], site_a["indices"][0],
             site_b["site"], site_b["indices"][0], framework.span(site_a, site_b)))

site_a, site_b = pairs[PAIR_ID]
oxygens = (site_a["indices"][0], site_b["indices"][0])

# -- the reaction ----------------------------------------------------------
reaction = yaml.safe_load(REACTION)
print("\n%s   %s" % (reaction["reaction"], reaction["reaction_family"]))
print(REACTION.strip())

ts = TSGraph(molecule_from_adjlist(reaction["reactant"]),
             molecule_from_adjlist(reaction["product"]), stretch=STRETCH)
print("\nbond changes")
print(ts.report())
print("\nmerged graph")
print(ts.adjlist())

# -- the sweep, both flips, keep the best clearance ------------------------
best = None
for flip, ordered in enumerate((oxygens, oxygens[::-1])):
    sweep = PairSweep(ts, framework.atoms, ordered)
    survivors = sweep.survivors()
    print("\nflip%d  X%d on O%d, X%d on O%d   %d/%d fit"
          % (flip, sweep.site_a + 1, ordered[0], sweep.site_b + 1, ordered[1],
             len(survivors), sweep.n_builds))
    for survivor in survivors:
        roll, clearance, molecule = max(sweep.rolls(survivor), key=lambda e: e[1])
        if best is None or clearance > best[0]:
            best = (clearance, flip, ordered, sweep, molecule)

if best is None:
    raise SystemExit("nothing fit the pair")
clearance, flip, ordered, sweep, molecule = best
print("\nbest: flip%d  scale %.2f  span %.2f A  roll %d  clearance %.3f A  torsions %s"
      % (flip, molecule.info["scale"], molecule.info["span"],
         molecule.info["roll"], clearance, molecule.info["angles"]))

frame = framework.atoms + molecule

# -- the bookkeeping -------------------------------------------------------
# graph atom -> index in frame: real atoms follow the framework in sweep.keep
# order; the two X atoms are the seated oxygens
where = {g: nslab + k for k, g in enumerate(sweep.keep)}
where[sweep.site_a], where[sweep.site_b] = ordered
bonds = []
print("\nforming / breaking bonds in ts_guess.xyz")
for change, i, j, before, after, stretch in ts.changes():
    if change not in ("form", "break"):
        continue
    a, b = where[i], where[j]
    d = frame.get_distance(a, b, mic=True)
    bonds.append({"change": change, "tags": [ts.tag(i), ts.tag(j)],
                  "indices": [int(a), int(b)], "distance": round(float(d), 3)})
    print("  %-6s %-6s %-6s  %4d %4d   %.3f A" % (change, ts.tag(i), ts.tag(j), a, b, d))

write(os.path.join(HERE, "ts_guess.xyz"), frame)
with open(os.path.join(HERE, "ts_guess.json"), "w") as handle:
    json.dump({"code": CODE, "t_labels": list(T_LABELS), "pair": "pair_%02d" % PAIR_ID,
               "reaction": reaction["reaction"], "nslab": nslab,
               "flip": flip, "oxygens": [int(o) for o in ordered],
               "clearance": round(clearance, 3), "bonds": bonds}, handle, indent=2)
print("\nwrote ts_guess.xyz and ts_guess.json in %s" % HERE)