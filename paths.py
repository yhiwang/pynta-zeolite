import os

REPO = os.environ.get(
    "PYNTAZ_REPO",
    "/home/ivanwang/pynta-zeolite")

PYTHON = os.environ.get(
    "PYNTAZ_PYTHON",
    "/nfs/hpc2/apo/kulkarnilab/ivanwang/miniconda3/envs/pyntaz-env/bin/python")

MODEL = os.environ.get(
    "PYNTAZ_MODEL",
    os.path.join(REPO, "models", "mace-mpa-0-medium.model"))

SLURM_ACCOUNT = os.environ.get("PYNTAZ_SLURM_ACCOUNT", "ark245grp")
SLURM_PARTITION = os.environ.get("PYNTAZ_SLURM_PARTITION", "high")


def check():
    missing = [(name, path) for name, path in
               (("PYTHON", PYTHON), ("MODEL", MODEL), ("REPO", REPO))
               if not os.path.exists(path)]
    for name, path in missing:
        print("%s does not exist: %s" % (name, path))
    return not missing