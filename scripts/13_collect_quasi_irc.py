#!/usr/bin/env python
"""Step 13 -- collect every converged DFT Sella TS (step 12) into its own
quasi_irc/ folder, where step 14 (vibrations) and step 15 (quasi-IRC) run.
Submits nothing; runs on the login node.

    python scripts/13_collect_quasi_irc.py

Which TSs: every row of DFT_TS/ts_collected.csv whose dft_sella/result.json
says converged. Per TS, only what the next steps need:

    <RUN_DIR>/DFT_TS/<i>_rxn/pair_<k>/<stem>/quasi_irc/start.xyz      copy of dft_sella/ts_dft.xyz
    <RUN_DIR>/DFT_TS/<i>_rxn/pair_<k>/<stem>/quasi_irc/ts_info.json   copy of step 11's ts_info.json

Every row reports one of:

    collected       start.xyz and ts_info.json written
    kept            quasi_irc/vib/job.id present (a step 14 job was submitted):
                    the folder is never rewritten
    not converged   result.json says not_converged
    no result       no dft_sella/result.json (not submitted, running or stopped)

Every folder without a vib/job.id is written fresh on every run, so rerunning
after more DFT Sella jobs converge adds them. Folders of TSs that no longer
pass are left alone.

Needs in scripts/layout.py: QUASI_IRC_DIR, DFT_VIB_DIR, TS_COLLECT_CSV,
TS_COLLECT_INFO, DFT_SELLA_DIR, DFT_SELLA_STRUCTURE.
"""

import csv
import json
import os
import shutil
import sys
from collections import Counter, defaultdict

import _common  # noqa: F401
import layout
import settings

REQUIRED_LAYOUT = ("QUASI_IRC_DIR", "DFT_VIB_DIR", "TS_COLLECT_CSV", "TS_COLLECT_INFO",
                   "DFT_SELLA_DIR", "DFT_SELLA_STRUCTURE")


def check_setup():
    """Stop before touching anything when a layout name is missing."""
    missing = ["layout." + n for n in REQUIRED_LAYOUT if not hasattr(layout, n)]
    if missing:
        sys.exit("missing: %s\nadd them first (see the docstring)" % ", ".join(missing))


def main():
    check_setup()
    run = layout.RunLayout(settings.RUN_DIR)
    table = os.path.join(run.dft_ts, layout.TS_COLLECT_CSV)
    if not os.path.isfile(table):
        sys.exit("%s not found -- run scripts/11_collect_ts.py first" % table)
    with open(table) as handle:
        rows = list(csv.DictReader(handle))

    status = Counter()
    by_reaction = defaultdict(list)
    for row in rows:
        by_reaction[int(row["reaction"])].append(row)
    for index in sorted(by_reaction):
        print("R%-3d %s" % (index, by_reaction[index][0]["reaction_name"]))
        for row in by_reaction[index]:
            ts_dir = os.path.join(run.dft_ts, row["id"])
            sella_dir = os.path.join(ts_dir, layout.DFT_SELLA_DIR)
            target = os.path.join(ts_dir, layout.QUASI_IRC_DIR)
            label = "%-52s" % row["id"]
            result_path = os.path.join(sella_dir, layout.TS_SELLA_RESULT)
            if not os.path.isfile(result_path):
                status["no result"] += 1
                print("  no result      %s" % label)
                continue
            with open(result_path) as handle:
                result = json.load(handle)
            if not result["converged"]:
                status["not converged"] += 1
                print("  not converged  %s  fmax %.3f" % (label, result["fmax"]))
                continue
            if os.path.isfile(os.path.join(target, layout.DFT_VIB_DIR, layout.JOB_ID)):
                status["kept"] += 1
                print("  kept           %s  (step 14 job submitted)" % label)
                continue
            os.makedirs(target, exist_ok=True)
            shutil.copy2(os.path.join(sella_dir, layout.DFT_SELLA_STRUCTURE),
                         os.path.join(target, layout.TS_START))
            shutil.copy2(os.path.join(ts_dir, layout.TS_COLLECT_INFO),
                         os.path.join(target, layout.TS_COLLECT_INFO))
            status["collected"] += 1
            print("  collected      %s  fmax %.3f  E %.4f" % (label, result["fmax"],
                                                             result["energy"]))

    print("\n%d rows in %s" % (len(rows), layout.TS_COLLECT_CSV))
    print(", ".join("%s %d" % item for item in sorted(status.items())))


if __name__ == "__main__":
    main()