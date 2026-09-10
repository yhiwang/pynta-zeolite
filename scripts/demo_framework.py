#!/usr/bin/env python
"""A tour of the framework API: the unique T pairs of a code, a single-Al
and a double-Al framework, the neighbourhood of the two Al, and the
save / load round trip. Writes into runs/ like step 0 does.
"""

import os

import _common  # noqa: F401
import layout
from pyntaz.framework import build_framework, unique_pairs

CODE = "MOR"
SINGLE = ("T4",)
PAIR = "T2-T4_5.65"
RUNS = "runs"

pairs = unique_pairs(CODE)
print("%d unique second-order pairs in %s:" % (len(pairs), CODE))
for name, record in pairs.items():
    print("  %-14s indices %-12s %.2f A" % (name, record["indices"], record["distance"]))
layout.save_pairs(pairs, RUNS, CODE)

single = build_framework(CODE, SINGLE)
single_dir = os.path.join(RUNS, "%s_%s" % (CODE, SINGLE[0]))
print("\n%s" % single_dir)
print(single.report())
layout.save_framework(single, single_dir)

record = pairs[PAIR]
double = build_framework(CODE, record["t_labels"], indices=record["indices"])
double_dir = os.path.join(RUNS, "%s_%s" % (CODE, PAIR))
print("\n%s" % double_dir)
print(double.report())
hood = double.neighborhood(*double.al_indices)
print("bridge T: %s" % " ".join("%s(%d)" % (lab, t) for t, lab in hood["bridge_t"].items()))
for t, oxygens in hood["bridge_o"].items():
    print("  via %s(%d): %s" % (hood["bridge_t"][t], t,
                                " ".join("%s(%d)" % (lab, o) for o, lab in oxygens.items())))
layout.save_framework(double, double_dir)

check = layout.load_framework(double_dir)
print("\nreload %s: %d atoms, Al %s, %d sites, %d pairs"
      % (double_dir, len(check.atoms), check.al_indices,
         len(check.mono_sites), len(check.bi_sites)))
