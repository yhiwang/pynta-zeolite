"""Step 2 of the workflow: relax one initial guess with MACE, framework
frozen, and the SLURM job that runs it.

Only ASE, MACE and :mod:`pyntaz.geometry` are needed here so the worker can
run on a compute node without maze or RMG.
"""

import os

from ase.io import read, write
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.calculators.singlepoint import SinglePointCalculator

from . import config
from .geometry import framework_indices

RELAX_WORKER = "relax_one.py"     # scripts/relax_one.py, copied into every job dir


def relax_structure(xyz_path, model_path=config.MACE_MODEL,
                    fmax=config.RELAX_FMAX, max_steps=config.RELAX_MAX_STEPS,
                    device="cpu"):
    """Relax ``xyz_path`` with MACE, freezing the framework.

    The framework is the largest connected cluster of Si/O/Al, found by
    connectivity rather than atom index, so an adsorbate containing O forms
    its own cluster and stays free.

    Writes ``relax.xyz`` (with energy and forces), ``relax.traj`` and
    ``relax.log`` next to the input. Returns (atoms, energy, converged).
    """
    from mace.calculators import mace_mp   # heavy import, only when relaxing

    if not os.path.isfile(model_path):
        raise FileNotFoundError("MACE model file not found: " + model_path)

    atoms = read(xyz_path)
    atoms.pbc = True

    frozen = framework_indices(atoms)
    free = [i for i in range(len(atoms)) if i not in set(frozen)]
    print("frozen  %d atoms  %s"
          % (len(frozen), sorted(set(atoms[i].symbol for i in frozen))))
    print("free    %d atoms  %s" % (len(free), [atoms[i].symbol for i in free]))
    if frozen:
        atoms.set_constraint(FixAtoms(indices=frozen))

    atoms.calc = mace_mp(model=model_path, default_dtype="float64", device=device)

    out_dir = os.path.dirname(os.path.abspath(xyz_path))
    optimizer = BFGS(atoms,
                     logfile=os.path.join(out_dir, config.RELAX_LOG),
                     trajectory=os.path.join(out_dir, config.RELAX_TRAJECTORY))
    converged = bool(optimizer.run(fmax=fmax, steps=max_steps))

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    print("energy    %.3f eV" % energy)
    print("steps     %d" % optimizer.get_number_of_steps())
    print("converged %s" % converged)

    # MACE's results are dropped with the calculator; cache them so extxyz
    # writes energy and forces into the file
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    write(os.path.join(out_dir, config.RELAXED_STRUCTURE), atoms)
    return atoms, energy, converged


SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --account=%(account)s
#SBATCH --partition=%(partition)s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=%(cores)d
#SBATCH --mem=%(memory)s
#SBATCH --time=%(time)s
#SBATCH --output=job.out
#SBATCH --error=job.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=%(repo)s:$PYTHONPATH

# the env python directly -- conda.sh on HPC2 points at an old install path
# and `conda activate` fails silently
%(python)s %(worker)s %(xyz)s
"""


def slurm_job_script(xyz_name, worker=RELAX_WORKER,
                     cores=config.SLURM_CORES, memory=config.SLURM_MEMORY,
                     time=config.SLURM_TIME):
    """Text of ``job.sh`` for one relaxation; ``xyz_name`` and ``worker``
    are relative to the job directory."""
    return SLURM_TEMPLATE % {"account": config.SLURM_ACCOUNT,
                             "partition": config.SLURM_PARTITION,
                             "cores": cores, "memory": memory, "time": time,
                             "repo": config.REPO, "python": config.PYTHON,
                             "worker": worker, "xyz": xyz_name}
