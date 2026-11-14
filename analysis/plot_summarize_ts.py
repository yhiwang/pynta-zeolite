#!/usr/bin/env python
"""One table and one figure for every Sella run of the run
directory (see analysis/ts_summary.py).

    <RUN_DIR>/TS_sella/ts_summary.csv   one row per guess and start
    <RUN_DIR>/TS_sella/ts_summary.png   A  best outcome per O pair and reaction
                                        B  verdicts by reaction and start
                                        C  where each search ended (bond map)
                                        D  TS energies within each reaction

    python summarize_ts.py             # green: TS found, red: no TS
    python summarize_ts.py --reasons   # colored by verdict (why no TS)

Runs on the login node in seconds and submits nothing; run it again whenever
more step 4 jobs have finished. Starts without result.json yet show as
"no result yet" ("job failed" when their job.err has a traceback).
"""

import os
import sys

import _common  # noqa: F401
import layout
import settings
from ts_summary import LABELS, VERDICTS, collect, plot_summary, verdict_counts, write_csv

for site, run_dir in layout.run_dirs(settings.MODEL, settings.ROUND, settings.FRAMEWORK, settings.SITES):
    print("\n=== %s" % run_dir)
    run = layout.RunLayout(run_dir)
    if not os.path.isdir(run.ts_sella):
        print("%s not found -- run step 4 first -- site skipped" % run.ts_sella)
        continue

    files = {"info": layout.REACTION_INFO, "bonds": layout.TS_BONDS_JSON,
             "start": layout.TS_START, "end": layout.TS_SELLA_STRUCTURE,
             "result": layout.TS_SELLA_RESULT, "err": layout.JOB_ERR}
    rows = collect(run.ts_sella, files)
    if not rows:
        print("no Sella runs under %s -- site skipped" % run.ts_sella)
        continue

    csv_path = os.path.join(run.ts_sella, layout.TS_SUMMARY_CSV)
    png_path = os.path.join(run.ts_sella, layout.TS_SUMMARY_PNG)
    write_csv(rows, csv_path)
    plot_summary(rows, png_path, os.path.relpath(run_dir), reasons="--reasons" in sys.argv[1:])

    counts = verdict_counts(rows)
    present = [v for v in VERDICTS if any(row["verdict"] == v for row in rows)]
    print("%-10s %-9s " % ("reaction", "start") + " ".join("%13s" % LABELS[v] for v in present))
    for (reaction, start) in sorted(counts):
        print("%-10s %-9s " % ("R%d" % reaction, start)
              + " ".join("%13d" % counts[(reaction, start)][v] for v in present))
    print("\nwrote %s\n      %s" % (csv_path, png_path))