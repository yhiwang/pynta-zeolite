"""Step 2 of the workflow: relax one structure with MACE, framework frozen.

Only ASE, MACE and :mod:`pyntaz.geometry` are needed here so it can run on
a compute node without maze or RMG. Reading the input and writing the
result are the caller's job; the optimizer's own log and trajectory are
written wherever the caller points them.
"""

from ase.optimize import BFGS
from ase.constraints import FixAtoms

from .geometry import framework_indices

RELAX_FMAX = 0.05           # eV/A
RELAX_MAX_STEPS = 100


def relax(atoms, model_path, frozen=None, fmax=RELAX_FMAX,
          max_steps=RELAX_MAX_STEPS, device="cpu", logfile=None,
          trajectory=None):
    """Relax ``atoms`` in place with MACE and BFGS.

    ``frozen`` lists the atoms held fixed; by default the framework, found
    by connectivity (the largest connected cluster of Si/O/Al) rather than by
    index, so an adsorbate containing O forms its own cluster and stays
    free. ``logfile`` / ``trajectory`` are passed straight to the optimizer.

    ``model_path`` must be a file: a bare model name would make MACE try to
    download it, which fails on a compute node without internet.

    Returns (energy, forces, converged, n_steps). ``atoms`` keeps the MACE
    calculator, whose results are dropped when it is replaced, so cache
    what you need before writing.
    """
    from mace.calculators import mace_mp   # heavy import, only when relaxing

    atoms.pbc = True
    if frozen is None:
        frozen = framework_indices(atoms)
    if frozen:
        atoms.set_constraint(FixAtoms(indices=list(frozen)))

    atoms.calc = mace_mp(model=model_path, default_dtype="float64", device=device)
    optimizer = BFGS(atoms, logfile=logfile, trajectory=trajectory)
    converged = bool(optimizer.run(fmax=fmax, steps=max_steps))

    return (atoms.get_potential_energy(), atoms.get_forces(), converged,
            optimizer.get_number_of_steps())
