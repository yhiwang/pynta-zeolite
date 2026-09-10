#!/usr/bin/env python
"""Step 0 -- build one bare framework and save it into its run directory.

Set CODE and SITES in settings.py. SITES is one T label ("T4") for a single
Al or a pair name from the printed table ("T2-T4_5.65") for two Al. With
two Al, SITE_RULE decides whether only the cross pairs (one O on each Al,
the O-Si-O bridges) and their oxygens become sites ("cross") or every
first-shell O and pair does ("all"). Writes <RUN_DIR>/bare.xyz,
framework.json (which records the rule) and <CODE>_pairs.json; every later
step reads the framework from there.
"""

import _common  # noqa: F401
import layout
import settings
from pyntaz.framework import build_framework, unique_pairs

pairs = unique_pairs(settings.CODE, settings.SUPERCELL_MIN_LENGTH)
print("%d unique second-order pairs in %s:" % (len(pairs), settings.CODE))
for name, record in pairs.items():
    print("  %-14s indices %-12s %.2f A" % (name, record["indices"], record["distance"]))

if settings.SITES in pairs:
    record = pairs[settings.SITES]
    framework = build_framework(settings.CODE, record["t_labels"],
                                indices=record["indices"],
                                min_length=settings.SUPERCELL_MIN_LENGTH,
                                max_span=settings.BIDENTATE_MAX_SPAN,
                                site_rule=settings.SITE_RULE)
else:
    framework = build_framework(settings.CODE, (settings.SITES,),
                                min_length=settings.SUPERCELL_MIN_LENGTH,
                                max_span=settings.BIDENTATE_MAX_SPAN,
                                site_rule=settings.SITE_RULE)

print("\n%s" % settings.RUN_DIR)
print(framework.report())
layout.save_framework(framework, settings.RUN_DIR)
layout.save_pairs(pairs, settings.RUN_DIR, settings.CODE)
