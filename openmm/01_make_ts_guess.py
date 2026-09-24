#!/usr/bin/env python
"""One TS guess per reaction in reaction.yaml, standalone: build the
framework once, seat every reaction on the same oxygen pair, keep the
best clearance over both flips. The sweep settings mirror
scripts/settings.py so each guess is the one step 5 would make.

Reads  reaction.yaml   (next to this file)
Writes guesses/<i>_rxn/ts_guess.xyz and ts_guess.json for reaction i:
       nslab, the seated oxygens, and every forming/breaking bond as
       indices into ts_guess.xyz
"""

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from ase.io import write
from pyntaz.framework import build_framework
from pyntaz.reactions import molecule_from_adjlist, parse_reactions
from pyntaz.ts_graph import TSGraph, PairSweep

HERE = os.path.dirname(os.path.abspath(__file__))
REACTIONS = os.path.join(HERE, "reaction.yaml")
OUT = os.path.join(HERE, "guesses")

# the run (settings.py: CODE, SITES)
CODE = "MOR"
T_LABELS = ("T4",)          # one label for a single Al, two for a pair
PAIR_ID = 0                 # every reaction goes on this row of the pair table

# step 0 (settings.py: SUPERCELL_MIN_LENGTH, BIDENTATE_MAX_SPAN, SITE_RULE)
SUPERCELL_MIN_LENGTH = 12.0
MAX_SPAN = 3.5
SITE_RULE = "all"

# step 5 (settings.py: TS_*)
TORSION_STEPS = 6
ROLL_STEPS = 24
SPAN_TOLERANCE = 0.4
SITE_SCALE_RANGE = (1.0, 1.2, 0.1)
STRETCH = {"form": 1.2, "break": 1.2}

# -- framework and the pair ------------------------------------------------
framework = build_framework(CODE, T_LABELS, min_length=SUPERCELL_MIN_LENGTH,
                            max_span=MAX_SPAN, site_rule=SITE_RULE)
nslab = len(framework.atoms)
pairs = framework.find_pairs(MAX_SPAN)
site_a, site_b = pairs[PAIR_ID]
oxygens = (site_a["indices"][0], site_b["indices"][0])
print("%s %s: %d atoms, pair_%02d = %s O%d / %s O%d, %.2f A"
      % (CODE, "-".join(T_LABELS), nslab, PAIR_ID, site_a["site"], oxygens[0],
         site_b["site"], oxygens[1], framework.span(site_a, site_b)))

# -- the reactions ---------------------------------------------------------
with open(REACTIONS) as handle:
    reactions, _ = parse_reactions(handle.read())
print("%d reactions in %s\n" % (len(reactions), os.path.basename(REACTIONS)))


def best_guess(ts):
    """Best-clearance roll over both flips of the pair, or None."""
    best = None
    for flip, ordered in enumerate((oxygens, oxygens[::-1])):
        sweep = PairSweep(ts, framework.atoms, ordered,
                          torsion_steps=TORSION_STEPS, roll_steps=ROLL_STEPS,
                          span_tolerance=SPAN_TOLERANCE, scale_range=SITE_SCALE_RANGE)
        for survivor in sweep.survivors():
            roll, clearance, molecule = max(sweep.rolls(survivor), key=lambda e: e[1])
            if best is None or clearance > best[0]:
                best = (clearance, flip, ordered, sweep, molecule)
    return best


for reaction in reactions:
    i = reaction["index"]
    ts = TSGraph(molecule_from_adjlist(reaction["reactant"]),
                 molecule_from_adjlist(reaction["product"]), stretch=STRETCH)
    best = best_guess(ts)
    if best is None:
        print("%d_rxn  %-40s  nothing fit" % (i, reaction["reaction"]))
        continue
    clearance, flip, ordered, sweep, molecule = best
    frame = framework.atoms + molecule

    # graph atom -> index in frame: real atoms follow the framework in
    # sweep.keep order; the two X atoms are the seated oxygens
    where = {g: nslab + k for k, g in enumerate(sweep.keep)}
    where[sweep.site_a], where[sweep.site_b] = ordered
    bonds = []
    for change, a, b, before, after, stretch in ts.changes():
        if change in ("form", "break"):
            bonds.append({"change": change, "tags": [ts.tag(a), ts.tag(b)],
                          "indices": [int(where[a]), int(where[b])],
                          "distance": round(float(frame.get_distance(where[a], where[b], mic=True)), 3)})

    out = os.path.join(OUT, "%d_rxn" % i)
    os.makedirs(out, exist_ok=True)
    write(os.path.join(out, "ts_guess.xyz"), frame)
    with open(os.path.join(out, "ts_guess.json"), "w") as handle:
        json.dump({"code": CODE, "t_labels": list(T_LABELS), "pair": "pair_%02d" % PAIR_ID,
                   "reaction": reaction["reaction"], "index": i, "nslab": nslab,
                   "flip": flip, "oxygens": [int(o) for o in ordered],
                   "clearance": round(clearance, 3), "bonds": bonds}, handle, indent=2)
    print("%d_rxn  %-40s  flip%d  scale %.2f  clearance %.3f A  %d changing bonds"
          % (i, reaction["reaction"], flip, molecule.info["scale"], clearance, len(bonds)))

print("\nwrote %s" % OUT)