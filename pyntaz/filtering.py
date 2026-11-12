"""RMSD deduplication: cluster structures by positional RMSD of the
adsorbate atoms; the first member of each cluster stands for it. Step 2 uses
it on the TS guesses.

The bond-survival test of the old adsorbate branch is in
``archive/adsorbates/pyntaz/filtering.py``.
"""

from .geometry import adsorbate_indices, positional_rmsd

RMSD_THRESHOLD = 2.0            # A, below this two structures are the same one


def cluster_by_rmsd(entries, threshold=RMSD_THRESHOLD):
    """[(representative, [members])] by greedy clustering on adsorbate RMSD.

    ``entries`` is ``[(stem, atoms, energy)]`` already energy-ordered, so
    every representative is the lowest-energy member of its own cluster.
    """
    indices = adsorbate_indices(entries[0][1])
    clusters = []
    for entry in entries:
        for representative, members in clusters:
            if positional_rmsd(representative[1], entry[1], indices) < threshold:
                members.append(entry)
                break
        else:
            clusters.append((entry, [entry]))
    return clusters
