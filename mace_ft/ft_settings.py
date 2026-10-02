"""How the MACE fine-tuning is set up. The run (RUN_DIR) and the foundation
model (MACE_MODEL) come from scripts/settings.py. Edit here, not in the
mace_ft scripts. Each FT_NAME gets its own folder, so changing the weights
and the name gives a second run side by side with the first."""

import os

FT_NAME = "ft_e1f10"            # folder <RUN_DIR>/MACE_ft/<FT_NAME>/
FT_SUBMIT = True               # False: write config.yaml + job.sh only (dry run)

# the mace-ft env's trainer, called directly (conda activate fails silently
# in batch jobs)
FT_MACE_RUN_TRAIN = os.path.expanduser("~/.conda/envs/mace-ft/bin/mace_run_train")

# loss weights: stage one, then stage two (lower learning rate, more energy)
FT_ENERGY_WEIGHT = 1.0
FT_FORCES_WEIGHT = 10.0
FT_STAGE_TWO = True
FT_STAGE_TWO_ENERGY_WEIGHT = 10.0
FT_STAGE_TWO_FORCES_WEIGHT = 10.0

FT_EPOCHS = 50                  # each epoch also replays FT_NUM_SAMPLES_PT MP structures
FT_STAGE_TWO_START = 40         # epoch at which stage two starts
FT_BATCH_SIZE = 4
FT_NUM_SAMPLES_PT = 10000       # MP replay structures (fewer = faster epochs)
FT_DTYPE = "float32"            # float64 is MACE's default, but these GPUs are slow
                                # in double precision; float32 is fine for training
FT_SEED = 123

FT_SLURM_ACCOUNT = "ark245grp"
FT_SLURM_PARTITION = "gpu-6000-blackwell"
FT_SLURM_CPUS = 8
FT_SLURM_MEMORY = "64G"
FT_SLURM_TIME = "24:00:00"