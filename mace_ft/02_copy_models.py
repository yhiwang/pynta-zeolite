#!/usr/bin/env python
"""MACE step 2: put the fine-tuned models into the next round.

    python mace_ft/02_copy_models.py

For every ``runs/<MODEL>/<ROUND>/<FRAMEWORK>/MACE_ft/model_<k>/`` whose
training is done (MACE wrote ``<name>.model`` there; <name> is the name in
the folder's config.yaml), it takes the Default head out of the model with
mace_select_head -- the head trained on the DFT data; the other head is
the replay -- and writes it, as a CPU model, to

    runs/<MODEL>/<next round>/<FRAMEWORK>/models/model_<k>.model

settings.py reads the model of round1 and later from there: PIPELINE_MODEL
is the model that runs the steps, the others are the check models.

A model file that is already there is left alone (jobs of the next round
may be using it); delete it to write it again. A model whose training is
not done is reported and skipped, so the script can be run again later.

Runs on the login node of the machine that trained the models. It calls
mace_select_head from the mace-ft env (next to FT_MACE_RUN_TRAIN in
scripts/machine.py). Afterwards copy the models/ folder to the machine
that runs the pipeline, into the same path.
"""

import os
import subprocess
import sys

import _common  # noqa: F401
import layout
import machine
import settings

CONFIG = "config.yaml"                 # MACE_ft/model_<k>/, written by 01_train.py
HEAD = "Default"                       # the head MACE trains on the DFT data


def run_name(folder):
    """The name of the run in ``folder``: the name line of its config.yaml."""
    with open(os.path.join(folder, CONFIG)) as handle:
        for line in handle:
            if line.startswith("name:"):
                return line.split(":", 1)[1].strip().strip("'")
    sys.exit("no name line in %s" % os.path.join(folder, CONFIG))


frame_dir = layout.framework_dir(settings.MODEL, settings.ROUND, settings.FRAMEWORK)
ft_dir = os.path.join(frame_dir, layout.MACE_FT)
next_round = "round%d" % len(settings.ROUNDS)          # the round after settings.ROUND
select_head = os.path.join(os.path.dirname(machine.FT_MACE_RUN_TRAIN), "mace_select_head")
prefix = layout.DFT_MODEL_PREFIX
models = []
if os.path.isdir(ft_dir):
    models = sorted((name for name in os.listdir(ft_dir)
                     if name.startswith(prefix) and name[len(prefix):].isdigit()),
                    key=lambda name: int(name[len(prefix):]))
if not models:
    sys.exit("no %s<k> folder in %s -- run mace_ft/01_train.py first" % (prefix, ft_dir))
if not os.access(select_head, os.X_OK):
    sys.exit("%s not found -- is the mace-ft env where FT_MACE_RUN_TRAIN in "
             "scripts/machine.py says?" % select_head)
print("%s -> %s" % (ft_dir, os.path.dirname(
    layout.round_model(settings.MODEL, next_round, settings.FRAMEWORK, 1))))

written = 0
for model in models:
    folder = os.path.join(ft_dir, model)
    target = layout.round_model(settings.MODEL, next_round, settings.FRAMEWORK,
                                int(model[len(prefix):]))
    if os.path.isfile(target):
        print("%-9s already there: %s" % (model, target))
        continue
    trained = os.path.join(folder, run_name(folder) + ".model")
    if not os.path.isfile(trained):
        print("%-9s not done: no %s" % (model, trained))
        continue
    os.makedirs(os.path.dirname(target), exist_ok=True)
    out = subprocess.run([select_head, "-n", HEAD, "-d", "cpu", "-o", os.path.abspath(target),
                          os.path.abspath(trained)], capture_output=True, text=True)
    if out.returncode != 0 or not os.path.isfile(target):
        sys.exit("mace_select_head failed for %s:\n%s" % (trained, out.stderr))
    written += 1
    print("%-9s wrote %s  (%.1f MB)" % (model, target, os.path.getsize(target) / 1e6))

print("\n%d of %d models written. For the next round set ROUND = \"%s\" in scripts/settings.py; "
      "PIPELINE_MODEL picks the model that runs the steps" % (written, len(models), next_round))