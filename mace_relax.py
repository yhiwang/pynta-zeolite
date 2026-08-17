#!/usr/bin/env python
"""
Relax one adsorbate config with MACE, freezing the zeolite framework.

    python mace_relax.py <init.xyz>

The framework is the largest connected cluster of Si/O/Al, found by
connectivity rather than atom index, so an adsorbate containing O forms its
own cluster and stays free.

Writes relax.xyz, relax.traj and relax.log next to the input.
"""

import os
import sys

from ase.io import read, write
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.calculators.singlepoint import SinglePointCalculator
from mace.calculators import mace_mp

import paths
from functions_v2 import find_framework_indices

FMAX = 0.05
STEPS = 100


def main(xyz_path):
    # a bare model name makes MACE try to download, which fails on a compute
    # node with no internet
    if not os.path.isfile(paths.MODEL):
        sys.exit("model file not found: " + paths.MODEL)

    atoms = read(xyz_path)
    atoms.pbc = True

    frozen = find_framework_indices(atoms)
    free = [i for i in range(len(atoms)) if i not in set(frozen)]
    print("frozen  %d atoms  %s"
          % (len(frozen), sorted(set(atoms[i].symbol for i in frozen))))
    print("free    %d atoms  %s" % (len(free), [atoms[i].symbol for i in free]))
    if frozen:
        atoms.set_constraint(FixAtoms(indices=frozen))

    atoms.calc = mace_mp(model=paths.MODEL, default_dtype="float64",
                         device="cpu")

    outdir = os.path.dirname(os.path.abspath(xyz_path))
    opt = BFGS(atoms,
               logfile=os.path.join(outdir, "relax.log"),
               trajectory=os.path.join(outdir, "relax.traj"))
    converged = opt.run(fmax=FMAX, steps=STEPS)

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    print("energy    %.3f eV" % energy)
    print("steps     %d" % opt.get_number_of_steps())
    print("converged %s" % bool(converged))

    # MACE's results are dropped with the calculator; cache them so extxyz
    # writes energy and forces into the file
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    write(os.path.join(outdir, "relax.xyz"), atoms)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])