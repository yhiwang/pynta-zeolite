"""How the MACE fine-tuning is set up. The run (RUN_DIR) and the round
(ROUND) come from scripts/settings.py. Edit here, not in the mace_ft
scripts. Each FT_NAME gets its own folder, so changing the weights and the
name gives a second run side by side with the first."""

import os

import settings                 # the mace_ft scripts put scripts/ on sys.path first

FT_NAME = "ft_round2_omat0_e1f10"   # folder <RUN_DIR>/MACE_ft/<FT_NAME>/; data = rounds 1+2
# always start from the original foundation model, not the pipeline's model
FT_FOUNDATION = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "models", "mace-omat-0-medium.model")
FT_SUBMIT = True               # False: write config.yaml + job.sh only (dry run)

# loss weights: stage one, then stage two (lower learning rate, more energy)
FT_ENERGY_WEIGHT = 1.0
FT_FORCES_WEIGHT = 10.0
FT_STAGE_TWO = False
FT_STAGE_TWO_ENERGY_WEIGHT = 10.0
FT_STAGE_TWO_FORCES_WEIGHT = 10.0

FT_EPOCHS = 40                  # each epoch also replays the MP structures
FT_STAGE_TWO_START = 40         # epoch at which stage two starts
FT_BATCH_SIZE = 4
FT_NUM_SAMPLES_PT = None        # MP replay size: None = every config that passes the filter;
                                # a number = that many (capped at the count when padding is off)
FT_PSEUDOLABEL_REPLAY = True    # relabel the replay with the foundation model
FT_FILTER_TYPE_PT = "combinations"     # none, combinations (only these elements),
                                       # exclusive (exactly these), inclusive (all of these + others)
FT_ATOMIC_NUMBERS = [1, 6, 8, 13, 14]  # H, C, O, Al, Si
FT_PADDING_PT = False           # False: never fill the replay with configs that fail the filter
FT_DTYPE = "float32"            # float64 is MACE's default, but these GPUs are slow
                                # in double precision; float32 is fine for training
FT_SEED = 123

FT_SLURM_CPUS = 8
FT_SLURM_MEMORY = "64G"
FT_SLURM_TIME = "24:00:00"