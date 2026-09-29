"""Step 8 of the workflow: Sella saddle search on one TS start with MACE,
then a frequency check of the saddle it finds.

    saddle_search     Sella (order 1) with every atom fixed except the
                      adsorbate and the framework within ``free_radius`` of it
    frequency_check   finite-difference vibrations of those free atoms: the
                      imaginary modes, and how much the first one stretches
                      each forming / breaking bond
    verdict           one word for the outcome: "ts", "not_converged",
                      "minimum", "higher_order" or "wrong_mode"

Sella does not know which bonds should change: it follows the softest mode
from wherever it starts. The frequency check is what tells a saddle of the
intended reaction from any other one.

Only ASE, Sella, MACE and numpy are needed, so this runs on a compute node
without maze or RMG. Nothing here reads or writes run directories except the
optimizer log / trajectory and the vibration cache the caller points to.
"""

import shutil

import numpy as np
from ase.constraints import FixAtoms

FMAX = 0.05             # eV/A
MAX_STEPS = 300
INTERNAL = False        # Sella internal coordinates instead of Cartesian
FREE_RADIUS = 4.0       # A, framework atoms this close to the adsorbate move
VIB_DELTA = 0.01        # A, finite-difference displacement
IMAG_CUTOFF = 50.0      # cm-1, smaller imaginary modes count as noise
MODE_MIN = 0.2          # |bond stretch| in the first mode that counts as moving it


def load_mace(model_path, device="cpu"):
    """One MACE calculator in float64 (Sella's curvature estimates need
    precise forces). ``model_path`` must be a file: a bare model name makes
    MACE try to download it, which fails on a compute node without internet."""
    from mace.calculators import mace_mp   # heavy import, only when needed
    return mace_mp(model=model_path, default_dtype="float64", device=device)


def free_atoms(atoms, n_framework, free_radius=FREE_RADIUS):
    """Sorted indices allowed to move: every adsorbate atom (index >=
    ``n_framework``) and every framework atom within ``free_radius`` of one."""
    adsorbate = np.arange(n_framework, len(atoms))
    local = [f for f in range(n_framework)
             if free_radius > 0
             and atoms.get_distances(f, adsorbate, mic=True).min() < free_radius]
    return sorted(local) + [int(i) for i in adsorbate]


def saddle_search(atoms, calc, free, fmax=FMAX, max_steps=MAX_STEPS,
                  internal=INTERNAL, logfile=None, trajectory=None):
    """Run Sella (order 1) on ``atoms`` in place, only ``free`` atoms moving.

    Returns a dict: ``converged`` (largest force on a free atom below
    ``fmax``, checked here rather than taken from the optimizer), ``fmax``,
    ``n_steps`` and ``energy``. ``atoms`` keeps ``calc`` and the FixAtoms
    constraint, ready for :func:`frequency_check`.
    """
    from sella import Sella   # only on the machines that run the search

    atoms.pbc = True
    free = set(free)
    atoms.set_constraint(FixAtoms(indices=[a for a in range(len(atoms)) if a not in free]))
    atoms.calc = calc
    optimizer = Sella(atoms, order=1, internal=internal,
                      logfile=logfile, trajectory=trajectory)
    optimizer.run(fmax=fmax, steps=max_steps)
    largest = float(np.linalg.norm(atoms.get_forces(), axis=1).max())  # fixed atoms read 0
    return {"converged": largest < fmax, "fmax": largest,
            "n_steps": optimizer.get_number_of_steps(),
            "energy": float(atoms.get_potential_energy())}


def bond_stretches(atoms, mode, bonds):
    """[how much ``mode`` (natoms x 3, any scale) lengthens each (i, j, change)
    bond], with the mode normalised to unit length: + stretches, - shortens."""
    mode = np.asarray(mode, dtype=float)
    mode = mode / np.linalg.norm(mode)
    out = []
    for i, j, _ in bonds:
        d = atoms.get_distance(i, j, mic=True, vector=True)
        out.append(float(np.dot(mode[j] - mode[i], d / np.linalg.norm(d))))
    return out


def frequency_check(atoms, free, bonds, name, delta=VIB_DELTA, imag_cutoff=IMAG_CUTOFF):
    """Finite-difference frequencies of the ``free`` atoms at the current
    geometry (``atoms`` must carry its calculator). ``name`` is the vibration
    cache directory; it is deleted first, because ASE reuses cached
    displacements and stale ones give wrong modes. Writes the first mode as
    ``<name>.0.traj`` for ase gui.

    Returns a dict: ``imaginary`` (cm-1, the ones above ``imag_cutoff``,
    largest first) and ``stretches`` (:func:`bond_stretches` of the first
    mode, or None when there is no imaginary mode).
    """
    from ase.vibrations import Vibrations

    shutil.rmtree(name, ignore_errors=True)
    vib = Vibrations(atoms, indices=list(free), name=name, delta=delta)
    vib.run()
    imaginary = sorted((float(f.imag) for f in vib.get_frequencies() if f.imag > imag_cutoff),
                       reverse=True)
    vib.write_mode(0)
    stretches = bond_stretches(atoms, vib.get_mode(0), bonds) if imaginary else None
    return {"imaginary": imaginary, "stretches": stretches}


def verdict(search, check, mode_min=MODE_MIN):
    """One word for what a start ended as. ``check`` is None when the
    search did not converge (no frequency check is worth running then).

        not_converged   Sella stopped before the force criterion
        minimum         no imaginary mode above the cutoff
        higher_order    two or more imaginary modes
        wrong_mode      one imaginary mode, but it moves no reacting bond
                        by at least ``mode_min``
        ts              one imaginary mode that moves a reacting bond
    """
    if not search["converged"] or check is None:
        return "not_converged"
    if not check["imaginary"]:
        return "minimum"
    if len(check["imaginary"]) > 1:
        return "higher_order"
    if max(abs(s) for s in check["stretches"]) < mode_min:
        return "wrong_mode"
    return "ts"