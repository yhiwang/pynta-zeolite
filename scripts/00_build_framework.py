#!/usr/bin/env python
"""Step 0 -- build one bare framework and save it into its run directory.

Set CODE and SITES below. SITES is one T label ("T4") for a single Al or a
pair name from the printed table ("T2-T4_5.65") for two Al. Writes
<RUN_DIR>/bare.xyz, framework.json and <CODE>_pairs.json; every later step
reads the framework from there.
"""

import json
import os

from _common import config
from pyntaz.framework import build_framework, unique_pairs

CODE = "MOR"
SITES = "T2-T4_5.65"
RUN_DIR = os.path.join("runs", "%s_%s" % (CODE, SITES))
MAX_SPAN = config.BIDENTATE_MAX_SPAN


pairs = unique_pairs(CODE)
print("%d unique second-order pairs in %s:" % (len(pairs), CODE))
for name, record in pairs.items():
    print("  %-14s indices %-12s %.2f A" % (name, record["indices"], record["distance"]))

if SITES in pairs:
    record = pairs[SITES]
    framework = build_framework(CODE, record["t_labels"], indices=record["indices"])
else:
    framework = build_framework(CODE, (SITES,))
framework.find_sites(MAX_SPAN)

print("\n%s" % RUN_DIR)
framework.describe()
framework.save(RUN_DIR)
with open(os.path.join(RUN_DIR, "%s_pairs.json" % CODE), "w") as handle:
    json.dump(pairs, handle, indent=2)