"""Which frames of a trajectory go to DFT. Pure functions on ase Atoms, no
files: the dft step scripts do the reading and writing."""

from ase.neighborlist import neighbor_list


def min_distance(atoms, search=1.5):
    """Shortest interatomic distance, or ``search`` when no pair is closer
    than that (every X-H bond is shorter, so this is the true minimum in
    practice)."""
    distances = neighbor_list("d", atoms, search)
    return float(distances.min()) if len(distances) else float(search)