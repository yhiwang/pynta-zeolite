#!/usr/bin/env python
"""One table and one figure for the initial and final states of step 7
(see analysis/endpoint_summary.py).

    <RUN_DIR>/TS_harmonic/endpoint_summary.csv   one row per guess
    <RUN_DIR>/TS_harmonic/endpoint_summary.png   A  survivors per O pair and reaction
                                                 B  outcomes by reaction
                                                 C  where each endpoint ended
                                                 D  survivor energies within each reaction

    python summarize_endpoints.py             # green: survived, red: lost
    python summarize_endpoints.py --reasons   # colored by status (why it was lost)

A guess survives when both its initial and final state are ok (bond check
passed, free relax converged). Runs on the login node in seconds and submits
nothing; run it again whenever more step 7 jobs have finished.
"""

import os
import sys
from collections import Counter

import _common  # noqa: F401
import layout
import settings
from endpoint_summary import collect, plot_summary, survivor_counts, write_csv

run = layout.RunLayout(settings.RUN_DIR)
if not os.path.isdir(run.ts_harmonic):
    sys.exit("%s not found -- run step 7 first" % run.ts_harmonic)

files = {"info": layout.REACTION_INFO, "bonds": layout.TS_BONDS_JSON,
         "result": layout.ENDPOINT_RESULT, "err": layout.JOB_ERR,
         "guess_suffix": layout.INITIAL_GUESS_SUFFIX}
rows = collect(run.ts_harmonic, files)
if not rows:
    sys.exit("no step 7 guesses under %s" % run.ts_harmonic)

csv_path = os.path.join(run.ts_harmonic, layout.ENDPOINT_SUMMARY_CSV)
png_path = os.path.join(run.ts_harmonic, layout.ENDPOINT_SUMMARY_PNG)
write_csv(rows, csv_path)
plot_summary(rows, png_path, os.path.relpath(settings.RUN_DIR),
             reasons="--reasons" in sys.argv[1:], cutoff=settings.TS_HARMONIC_BOND_CUTOFF)

counts = survivor_counts(rows)
print("%-10s %8s %11s %9s %9s" % ("reaction", "guesses", "initial ok", "final ok", "survived"))
for reaction in sorted(counts):
    c = counts[reaction]
    print("%-10s %8d %11d %9d %9d"
          % ("R%d" % reaction, c["guesses"], c["initial"], c["final"], c["survived"]))
total = sum(counts.values(), Counter())
print("%-10s %8d %11d %9d %9d"
      % ("total", total["guesses"], total["initial"], total["final"], total["survived"]))
print("\nwrote %s\n      %s" % (csv_path, png_path))