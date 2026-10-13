#!/usr/bin/env python
"""Move one round's pipeline results out of the way, so steps 07-10 start
clean with the next model.

    python scripts/archive_round.py        # MOVE = False: dry run, shows what would move

    <RUN_DIR>/TS_harmonic  ->  <RUN_DIR>/<ROUND>/TS_harmonic
    <RUN_DIR>/TS_sella     ->  <RUN_DIR>/<ROUND>/TS_sella
    <RUN_DIR>/NEB          ->  <RUN_DIR>/<ROUND>/NEB

Run it after dft/01_collect_configs.py has collected this round (it reads
these folders). DFT_sp/ is never touched: DFT data stays in DFT_sp/<round>/.

Refuses when
  - any job.id in these folders is still in squeue (a job writing there)
  - <RUN_DIR>/<ROUND>/<folder> already exists (never overwrites)
and warns when DFT_sp/<ROUND>/ does not exist yet (not collected?).
"""

import os
import shutil
import subprocess
import sys

import _common  # noqa: F401
import layout
import settings

ROUND = "round2"                              # same name as COLLECT_ROUND
FOLDERS = ("TS_harmonic", "TS_sella", "NEB")  # the folders steps 07-10 write
MOVE = True                                  # False: dry run


def queued_ids():
    """SLURM ids of this user's jobs, or None when squeue is unavailable."""
    try:
        out = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""),
                              "-o", "%i"], capture_output=True, text=True,
                             check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return {line.strip() for line in out.splitlines() if line.strip()}


def job_ids(folder):
    """{job id: folder} for every job.id under ``folder``."""
    found = {}
    for dirpath, _, files in os.walk(folder):
        if layout.JOB_ID in files:
            with open(os.path.join(dirpath, layout.JOB_ID)) as handle:
                found[handle.read().strip()] = dirpath
    return found


run_dir = settings.RUN_DIR
target = os.path.join(run_dir, ROUND)
present = [name for name in FOLDERS if os.path.isdir(os.path.join(run_dir, name))]
print("%s -> %s" % (run_dir, target))
if not present:
    sys.exit("none of %s in %s -- nothing to move" % (", ".join(FOLDERS), run_dir))

clashes = [name for name in present if os.path.exists(os.path.join(target, name))]
if clashes:
    sys.exit("%s already exist in %s -- pick another ROUND"
             % (", ".join(clashes), target))

ids = {}
for name in present:
    ids.update(job_ids(os.path.join(run_dir, name)))
in_queue = queued_ids()
if ids and in_queue is None:
    sys.exit("squeue not available, cannot check %d jobs -- run this on the login node" % len(ids))
running = {job: path for job, path in ids.items() if job in (in_queue or set())}
if running:
    for job, path in sorted(running.items()):
        print("  still in the queue: job %s  %s" % (job, path))
    sys.exit("%d jobs still in the queue -- wait for them (or scancel) first" % len(running))

if not os.path.isdir(os.path.join(layout.RunLayout(run_dir).dft_sp, ROUND)):
    print("warning: DFT_sp/%s does not exist -- has this round been collected?" % ROUND)

for name in present:
    print("  %-12s -> %s" % (name, os.path.join(target, name)))
missing = [name for name in FOLDERS if name not in present]
if missing:
    print("  (not there, skipped: %s)" % ", ".join(missing))

if not MOVE:
    print("\ndry run: nothing moved. Set MOVE = True in scripts/archive_round.py")
    sys.exit()
os.makedirs(target, exist_ok=True)
for name in present:
    shutil.move(os.path.join(run_dir, name), os.path.join(target, name))
print("\nmoved %d folders. Steps 07-10 now start clean in %s" % (len(present), run_dir))