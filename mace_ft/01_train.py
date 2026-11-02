#!/usr/bin/env python
"""MACE step 1: fine-tune the foundation model on the DFT training set.

    python mace_ft/01_train.py         # FT_SUBMIT = False: write config.yaml + job.sh only

Writes ``<RUN_DIR>/MACE_ft/<FT_NAME>/config.yaml`` (every mace_run_train
option, readable and kept with the results) and ``job.sh`` (one GPU; the
cluster values come from scripts/machine.py), then submits it. The job
runs in that folder, so the logs, checkpoints, the replay subset and the
final model all land there:

    <FT_NAME>.model            the fine-tuned model (GPU)
    <FT_NAME>_compiled.model   same, compiled
    *_cpu.model                versions for CPU nodes (the pipeline's Sella runs)
    logs/, results/, checkpoints/

Multihead replay fine-tuning: the training data becomes its own head, and
replay structures are trained alongside so the model keeps its general
chemistry. pt_train_file must be set explicitly -- with a foundation model
given as a *file*, MACE otherwise falls back to naive fine-tuning with only
a warning. FT_REPLAY = "mp" uses MACE's MP file, read from ~/.cache/mace/
(download it once on the login node). FT_REPLAY = a path uses that file:
MACE reads it whole (no filter, no sampling, its own 10% validation split),
so it must be filtered and sized before and use REF_energy / REF_forces.

The replay keeps only the MP configs that pass FT_FILTER_TYPE_PT for
FT_ATOMIC_NUMBERS (MACE's filter_atoms rules). mace_run_train has no "keep
all" value for num_samples_pt, so this script counts the passing configs
in the MP file and writes that count (or FT_NUM_SAMPLES_PT, capped at the
count when padding is off). With FT_PSEUDOLABEL_REPLAY the replay labels
come from the foundation model -- needed for a non-MP foundation such as
OMAT-0, because MACE loads the MP file for every pt_train_file name.

E0s come from the config_type=IsolatedAtom structures in train.extxyz (dft
step 4 + step 3); the script stops if any element in the data lacks one.

job.id guards against submitting the same run twice; delete it (and the
folder's outputs) to start over, or rerun with a new FT_NAME.
"""

import os
import subprocess
import sys

import _common  # noqa: F401
import ft_settings as fs
import layout
import machine
import settings
from ase.data import chemical_symbols
from ase.io import read

TRAINING_DIR = "training"                  # DFT_sp/, written by dft step 3
FILES = {"train_file": "train.extxyz", "valid_file": "valid.extxyz",
         "test_file": "test.extxyz"}
FT_DIR = "MACE_ft"                         # <RUN_DIR>/MACE_ft/<FT_NAME>/
CONFIG = "config.yaml"
REPLAY_CACHE = os.path.join(os.environ.get("XDG_CACHE_HOME",
                                           os.path.expanduser("~/.cache")),
                            "mace", "mp_traj_combinedxyz")

JOB_TEMPLATE = """#!/bin/bash
{header}

cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS={cpus}
nvidia-smi
{trainer} --config {config}
"""


def check_e0s(train_path):
    """Elements in the data without an IsolatedAtom entry (empty = fine)."""
    configs = read(train_path, ":")
    with_e0 = {a.get_chemical_symbols()[0] for a in configs
               if len(a) == 1 and a.info.get("config_type") == "IsolatedAtom"}
    needed = {s for a in configs for s in a.get_chemical_symbols()}
    return sorted(needed - with_e0), len(configs) - len(with_e0)


def passes(symbols, elements, filter_type):
    """MACE's filter_atoms rule for one config (symbols, elements: sets)."""
    if filter_type == "none":
        return True
    if filter_type == "combinations":       # only listed elements, not all needed
        return symbols <= elements
    if filter_type == "exclusive":          # exactly the listed elements
        return symbols == elements
    return elements <= symbols              # inclusive: all listed, others allowed


def count_replay(path, filter_type, elements):
    """(configs that pass, all configs) in a replay file. Reads only the
    species column, so it stops if a frame does not start with it."""
    n_pass = n_all = 0
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            n_atoms = int(line)
            if "Properties=species:S:1" not in next(handle):
                sys.exit("%s frame %d: species is not the first column -- cannot "
                         "count the replay" % (path, n_all + 1))
            symbols = {next(handle).split(None, 1)[0] for _ in range(n_atoms)}
            n_all += 1
            n_pass += passes(symbols, elements, filter_type)
    return n_pass, n_all


def yaml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return "'%s'" % value
    return str(value)


def main():
    run = layout.RunLayout(settings.RUN_DIR)
    data_dir = os.path.abspath(os.path.join(run.dft_sp, TRAINING_DIR))
    paths = {key: os.path.join(data_dir, name) for key, name in FILES.items()}
    for path in paths.values():
        if not os.path.isfile(path):
            sys.exit("%s not found -- run dft/03_build_training_set.py first" % path)
    if fs.FT_REPLAY == "mp":
        replay = REPLAY_CACHE
    else:
        replay = os.path.abspath(fs.FT_REPLAY)
    if not os.path.isfile(replay):
        sys.exit("replay data %s not found -- download it on the login node, or check "
                 "FT_REPLAY in mace_ft/ft_settings.py" % replay)
    if not os.path.isfile(fs.FT_FOUNDATION):
        sys.exit("foundation model %s not found" % fs.FT_FOUNDATION)
    if not os.access(machine.FT_MACE_RUN_TRAIN, os.X_OK):
        sys.exit("%s not found -- is the mace-ft env where FT_MACE_RUN_TRAIN in "
                 "scripts/machine.py says?" % machine.FT_MACE_RUN_TRAIN)
    missing, n_train = check_e0s(paths["train_file"])
    if missing:
        sys.exit("no IsolatedAtom E0 in train.extxyz for %s -- run dft/04 and then "
                 "dft/03 again" % ", ".join(missing))

    if fs.FT_FILTER_TYPE_PT not in ("none", "combinations", "exclusive", "inclusive"):
        sys.exit("FT_FILTER_TYPE_PT must be none, combinations, exclusive or inclusive")
    elements = {chemical_symbols[z] for z in fs.FT_ATOMIC_NUMBERS}
    n_pass, n_all = count_replay(replay, fs.FT_FILTER_TYPE_PT, elements)
    if n_pass == 0:
        sys.exit("no replay config in %s passes filter %s %s -- check FT_ATOMIC_NUMBERS"
                 % (replay, fs.FT_FILTER_TYPE_PT, fs.FT_ATOMIC_NUMBERS))
    if fs.FT_REPLAY == "mp":
        n_replay = n_pass if fs.FT_NUM_SAMPLES_PT is None else fs.FT_NUM_SAMPLES_PT
        capped = n_replay > n_pass and not fs.FT_PADDING_PT   # MACE would stop in the job
        if capped:
            n_replay = n_pass
    else:                                   # MACE reads the whole file
        n_replay, capped = n_all, False

    folder = os.path.abspath(os.path.join(settings.RUN_DIR, FT_DIR, fs.FT_NAME))
    id_path = os.path.join(folder, layout.JOB_ID)
    if os.path.isfile(id_path):
        with open(id_path) as handle:
            sys.exit("%s already submitted (job %s) -- use a new FT_NAME, or delete "
                     "job.id to resubmit" % (folder, handle.read().strip()))
    os.makedirs(folder, exist_ok=True)

    options = {
        "name": fs.FT_NAME,
        "foundation_model": os.path.abspath(fs.FT_FOUNDATION),
        "multiheads_finetuning": True,
        "pt_train_file": "mp" if fs.FT_REPLAY == "mp" else replay,
        "num_samples_pt": n_replay,
        "pseudolabel_replay": fs.FT_PSEUDOLABEL_REPLAY,
        "filter_type_pt": fs.FT_FILTER_TYPE_PT,
        "atomic_numbers": str(fs.FT_ATOMIC_NUMBERS),   # a string: MACE refuses a yaml list here
        "disallow_random_padding_pt": not fs.FT_PADDING_PT,
        **paths,
        "energy_key": "REF_energy",
        "forces_key": "REF_forces",
        "compute_stress": False,
        "stress_weight": 0.0,
        "loss": "weighted",
        "energy_weight": fs.FT_ENERGY_WEIGHT,
        "forces_weight": fs.FT_FORCES_WEIGHT,
        "swa": fs.FT_STAGE_TWO,
        "start_swa": fs.FT_STAGE_TWO_START,
        "swa_energy_weight": fs.FT_STAGE_TWO_ENERGY_WEIGHT,
        "swa_forces_weight": fs.FT_STAGE_TWO_FORCES_WEIGHT,
        "max_num_epochs": fs.FT_EPOCHS,
        "batch_size": fs.FT_BATCH_SIZE,
        "valid_batch_size": fs.FT_BATCH_SIZE,
        "default_dtype": fs.FT_DTYPE,
        "device": "cuda",
        "seed": fs.FT_SEED,
        "error_table": "TotalMAE",
        "eval_interval": 1,
        "save_cpu": True,
        "restart_latest": True,
    }
    if not fs.FT_STAGE_TWO:
        for key in ("swa", "start_swa", "swa_energy_weight", "swa_forces_weight"):
            options.pop(key)
    with open(os.path.join(folder, CONFIG), "w") as handle:
        handle.write("# written by mace_ft/01_train.py -- edit mace_ft/ft_settings.py\n")
        for key, value in options.items():
            handle.write("%s: %s\n" % (key, yaml_value(value)))
    with open(os.path.join(folder, layout.JOB_SCRIPT), "w") as handle:
        handle.write(JOB_TEMPLATE.format(
            header=machine.sbatch_header(fs.FT_SLURM_CPUS, fs.FT_SLURM_MEMORY,
                                         fs.FT_SLURM_TIME, name=fs.FT_NAME, gpu=True),
            cpus=fs.FT_SLURM_CPUS, trainer=machine.FT_MACE_RUN_TRAIN, config=CONFIG))

    print("training data  %s  (%d configs + E0s)" % (data_dir, n_train))
    print("foundation     %s" % fs.FT_FOUNDATION)
    print("replay         %d of %d configs in %s pass %s %s; using %d%s, padding %s, "
          "pseudolabels %s"
          % (n_pass, n_all, os.path.basename(replay), fs.FT_FILTER_TYPE_PT,
             fs.FT_ATOMIC_NUMBERS, n_replay,
             " (FT_NUM_SAMPLES_PT = %d capped)" % fs.FT_NUM_SAMPLES_PT if capped else "",
             fs.FT_PADDING_PT, fs.FT_PSEUDOLABEL_REPLAY))
    print("weights        E:F = %g:%g, stage two from epoch %s at %g:%g"
          % (fs.FT_ENERGY_WEIGHT, fs.FT_FORCES_WEIGHT, fs.FT_STAGE_TWO_START,
             fs.FT_STAGE_TWO_ENERGY_WEIGHT, fs.FT_STAGE_TWO_FORCES_WEIGHT)
          if fs.FT_STAGE_TWO else "weights        E:F = %g:%g"
          % (fs.FT_ENERGY_WEIGHT, fs.FT_FORCES_WEIGHT))
    print("wrote          %s/{%s, %s}" % (folder, CONFIG, layout.JOB_SCRIPT))
    if not fs.FT_SUBMIT:
        print("\ndry run: check config.yaml, then set FT_SUBMIT = True in "
              "mace_ft/ft_settings.py")
        return
    out = subprocess.run(["sbatch", layout.JOB_SCRIPT], cwd=folder,
                         capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit("sbatch failed:\n%s" % out.stderr)
    job_id = out.stdout.strip().split()[-1]
    with open(id_path, "w") as handle:
        handle.write(job_id + "\n")
    print("\nsubmitted job %s -- follow it with: tail -f %s/logs/*.log"
          % (job_id, folder))


if __name__ == "__main__":
    main()