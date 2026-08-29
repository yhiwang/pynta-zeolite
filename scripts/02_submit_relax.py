#!/usr/bin/env python
"""Step 2 -- mirror Adsorbates/ into Adsorbates_relax/ and submit one MACE
relaxation job per config.

    python scripts/02_submit_relax.py            # dry run: writes job dirs + job.sh
    python scripts/02_submit_relax.py --submit   # actually sbatch (on HPC2)

Each job directory gets the initial guess, a copy of relax_one.py and a
job.sh. Configs that already have a relax.xyz are skipped.
"""

import os
import shutil
import subprocess
import sys

from _common import step_parser, layout_from, config


def main():
    parser = step_parser(__doc__)
    parser.add_argument("--submit", action="store_true", help="run sbatch")
    parser.add_argument("--skip", nargs="*", default=[], metavar="SPECIES",
                        help="species names to leave out")
    args = parser.parse_args()
    layout = layout_from(args)

    from pyntaz import runtree
    from pyntaz.relax import slurm_job_script, RELAX_WORKER

    if not config.check_machine_paths():
        sys.exit("fix pyntaz/config.py or set the PYNTAZ_* environment variables")
    if not os.path.isdir(layout.adsorbates):
        sys.exit("%s not found -- run step 1 first" % layout.adsorbates)

    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), RELAX_WORKER)
    n_jobs = 0
    for relative, xyz in runtree.initial_guess_files(layout.adsorbates):
        if relative.split(os.sep)[0] in args.skip:
            continue

        job_dir = os.path.join(layout.relaxed, relative)
        os.makedirs(job_dir, exist_ok=True)
        shutil.copy2(xyz, os.path.join(job_dir, os.path.basename(xyz)))
        shutil.copy2(worker, os.path.join(job_dir, RELAX_WORKER))

        if os.path.exists(os.path.join(job_dir, config.RELAXED_STRUCTURE)):
            print("already done  ", relative)
            continue

        with open(os.path.join(job_dir, "job.sh"), "w") as handle:
            handle.write(slurm_job_script(os.path.basename(xyz)))

        if args.submit:
            # list argv, never a shell: species names contain [ ] =
            result = subprocess.run(["sbatch", "--chdir=" + job_dir, "job.sh"],
                                    capture_output=True, text=True)
            print(relative, "->", result.stdout.strip() or result.stderr.strip())
            if result.returncode != 0:
                sys.exit("sbatch failed, stopping")
        else:
            print("would submit  ", relative)
        n_jobs += 1

    print("\n%d configs %s" % (n_jobs, "submitted" if args.submit else "would be submitted"))


if __name__ == "__main__":
    main()
