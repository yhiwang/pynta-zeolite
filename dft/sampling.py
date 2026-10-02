"""Which frames of a trajectory go to DFT. Pure functions on ase Atoms, no
files: the dft step scripts do the reading and writing."""

import numpy as np
from ase.geometry import find_mic
from ase.neighborlist import neighbor_list


def max_displacement(a, b):
    """Largest distance any atom moved between ``a`` and ``b`` (minimum
    image, so an atom crossing the cell edge is not a 20 A jump)."""
    delta = b.get_positions() - a.get_positions()
    _, lengths = find_mic(delta, a.cell, a.pbc)
    return float(lengths.max())


def min_distance(atoms, search=1.5):
    """Shortest interatomic distance, or ``search`` when no pair is closer
    than that (every X-H bond is shorter, so this is the true minimum in
    practice)."""
    distances = neighbor_list("d", atoms, search)
    return float(distances.min()) if len(distances) else float(search)


def select_frames(frames, stride=1, min_disp=0.1, max_frames=8):
    """Indices of the frames to keep, in order.

    - the last frame (where the search stopped) is always kept, the first
      (where it started) too unless it is within ``min_disp`` of the last
    - of every ``stride``-th frame in between, one is kept once some atom
      has moved ``min_disp`` since the previous kept frame -- the optimizer
      takes big steps early and tiny ones near the end, so this spreads the
      picks by geometry instead of by step number
    - a pick closer than ``min_disp`` to the last frame gives way to it
    - ``max_frames`` caps the total: the interior picks are thinned evenly
    """
    n = len(frames)
    if n == 0:
        return []
    last = n - 1
    kept = [0]
    for i in range(stride, last, stride):
        if max_displacement(frames[kept[-1]], frames[i]) >= min_disp:
            kept.append(i)
    if last > 0:
        if max_displacement(frames[kept[-1]], frames[last]) < min_disp:
            kept.pop()
        kept.append(last)
    return thin(kept, max_frames)


def thin(kept, max_frames):
    """Keep the first and last entries and ``max_frames - 2`` evenly spaced
    ones in between."""
    if max_frames is None or len(kept) <= max_frames:
        return kept
    if max_frames < 2:
        return kept[-1:]
    interior = kept[1:-1]
    picks = np.linspace(0, len(interior) - 1, max_frames - 2).round().astype(int)
    return [kept[0]] + [interior[k] for k in picks] + [kept[-1]]