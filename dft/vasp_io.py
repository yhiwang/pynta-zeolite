"""Reading finished VASP single points back in the config's atom order.
Pure functions on one job folder; the dft step scripts do the walking."""

import os

import numpy as np
from ase.geometry import find_mic
from ase.io import read

OUTCAR = "OUTCAR"
SORT_FILE = "ase-sort.dat"                        # written by ase in dft step 2
FINISHED_MARK = "General timing and accounting"   # last block of a finished OUTCAR
UNCONVERGED_MARK = "EDIFF was not reached"        # VASP 6: SCF hit NELM
POSITION_TOL = 1e-3                               # A, config vs OUTCAR geometry


def _tail(path, size=20000):
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell() - size))
        return handle.read()


def finished(folder):
    """True when VASP ran to the end (whether or not the SCF converged)."""
    path = os.path.join(folder, OUTCAR)
    return os.path.isfile(path) and FINISHED_MARK.encode() in _tail(path)


def scf_converged(folder):
    """False when VASP stopped the SCF at NELM without reaching EDIFF."""
    with open(os.path.join(folder, OUTCAR), "rb") as handle:
        return UNCONVERGED_MARK.encode() not in handle.read()


def read_resort(folder):
    """Index list mapping VASP's element-sorted order back to config order:
    ``original = sorted[resort]`` (the same convention as ase's Vasp)."""
    with open(os.path.join(folder, SORT_FILE)) as handle:
        return np.array([int(line.split()[1]) for line in handle if line.strip()])


def read_dft(folder, config):
    """(energy, forces in config order) of a finished single point. Raises
    if the geometry VASP saw does not match ``config`` atom for atom."""
    sorted_atoms = read(os.path.join(folder, OUTCAR))     # last ionic step
    resort = read_resort(folder)
    if len(resort) != len(config):
        raise ValueError("%d atoms in %s, %d in the config"
                         % (len(resort), SORT_FILE, len(config)))
    delta = sorted_atoms.get_positions()[resort] - config.get_positions()
    _, lengths = find_mic(delta, config.cell, True)
    if lengths.max() > POSITION_TOL:
        raise ValueError("OUTCAR geometry differs from the config by %.3f A"
                         % lengths.max())
    return float(sorted_atoms.get_potential_energy()), sorted_atoms.get_forces()[resort]
