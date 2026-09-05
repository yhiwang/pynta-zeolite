import json
import os

from pyntaz.framework import build_framework, load_framework, unique_pairs

CODE = "MOR"
SINGLE = ("T4",)
PAIR = "T2-T4_5.65"
RUNS = "runs"


pairs = unique_pairs(CODE)
print("%d unique second-order pairs in %s:" % (len(pairs), CODE))
for name, record in pairs.items():
    print("  %-14s indices %-12s %.2f A" % (name, record["indices"], record["distance"]))

os.makedirs(RUNS, exist_ok=True)
with open(os.path.join(RUNS, "%s_pairs.json" % CODE), "w") as handle:
    json.dump(pairs, handle, indent=2)

single = build_framework(CODE, SINGLE)
single_dir = os.path.join(RUNS, "%s_%s" % (CODE, SINGLE[0]))
print("\n%s" % single_dir)
single.describe()
single.save(single_dir)

record = pairs[PAIR]
double = build_framework(CODE, record["t_labels"], indices=record["indices"])
double_dir = os.path.join(RUNS, "%s_%s" % (CODE, PAIR))
print("\n%s" % double_dir)
double.describe()
hood = double.neighborhood(*double.al_indices)
print("bridge T: %s" % " ".join("%s(%d)" % (lab, t) for t, lab in hood["bridge_t"].items()))
for t, oxygens in hood["bridge_o"].items():
    print("  via %s(%d): %s" % (hood["bridge_t"][t], t,
                                " ".join("%s(%d)" % (lab, o) for o, lab in oxygens.items())))
double.save(double_dir)

check = load_framework(double_dir)
print("\nreload %s: %d atoms, Al %s, %d sites, %d pairs"
      % (double_dir, len(check.atoms), check.al_indices,
         len(check.mono_sites), len(check.bi_sites)))