#!/usr/bin/env python
"""Relax one config with MACE, framework frozen. This is the worker that
job.sh runs inside every relaxation directory (step 2 writes the job).

    python relax_one.py <stem>_init.xyz

Writes relax.xyz (with energy and forces), relax.traj and relax.log next
to the input. Needs only ase, mace and pyntaz.geometry / pyntaz.relax, so
it runs on a compute node without maze or RMG; the repo must be on
PYTHONPATH (job.sh exports it).
"""

import os
import sys

import _common  # noqa: F401
import layout
import settings
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator
from pyntaz.geometry import framework_indices
from pyntaz.relax import relax


def main(xyz_path):
    if not os.path.isfile(settings.MACE_MODEL):
        sys.exit("MACE model file not found: " + settings.MACE_MODEL)
    out_dir = os.path.dirname(os.path.abspath(xyz_path))

    atoms = read(xyz_path)
    frozen = framework_indices(atoms)
    free = [i for i in range(len(atoms)) if i not in set(frozen)]
    print("frozen  %d atoms  %s" % (len(frozen), sorted({atoms[i].symbol for i in frozen})))
    print("free    %d atoms  %s" % (len(free), [atoms[i].symbol for i in free]))

    energy, forces, converged, n_steps = relax(
        atoms, settings.MACE_MODEL, frozen=frozen,
        fmax=settings.RELAX_FMAX, max_steps=settings.RELAX_MAX_STEPS,
        logfile=os.path.join(out_dir, layout.RELAX_LOG),
        trajectory=os.path.join(out_dir, layout.RELAX_TRAJECTORY))
    print("energy    %.3f eV" % energy)
    print("steps     %d" % n_steps)
    print("converged %s" % converged)

    # MACE's results are dropped with the calculator; cache them so extxyz
    # writes energy and forces into the file
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    write(os.path.join(out_dir, layout.RELAXED_STRUCTURE), atoms)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
