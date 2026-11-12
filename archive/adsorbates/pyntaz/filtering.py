"""Step 3 of the workflow: keep the relaxed configs that held together, then
deduplicate them.

*Survival* compares the first and last frame of a relaxation and keeps a
config only if every bond inside the adsorbate, plus the binder-to-framework
bond, changed by less than ``BOND_CHANGE_THRESHOLD``.

*Deduplication* clusters the survivors of each (species, site) by
positional RMSD of the adsorbate atoms and keeps the lowest-energy member of
each cluster.
"""

from .geometry import (framework_indices, adsorbate_indices, neighbor_list,
                       bonded_neighbors, nearest_atom, positional_rmsd)

BOND_CHANGE_THRESHOLD = 0.5     # A, a bond that moved more than this has broken
RMSD_THRESHOLD = 2.0            # A, below this two relaxed configs are the same minimum


# --------------------------------------------------------------------------
# bond map of the adsorbate
# --------------------------------------------------------------------------

def binder_indices(info, n_framework=None):
    """Absolute indices of the binding atoms in a combined structure
    (framework atoms first, then the adsorbate), from the species record of
    :func:`pyntaz.adsorbates.species_info`. ``n_framework`` defaults to
    ``info["nslab"]``; pass the count found by connectivity when you have it."""
    if n_framework is None:
        n_framework = info["nslab"]
    return sorted(n_framework + int(k)
                  for k in info["gratom_to_molecule_surface_atom_map"])


def adsorbate_bond_map(atoms, binders, mult=1.2):
    """{adsorbate index: [bonded neighbor indices]} for every adsorbate atom.

    Adsorbate-adsorbate bonds come from the neighbor list with ``mult=1.2``
    (see :func:`geometry.neighbor_list` for why). Framework partners are
    excluded on purpose: a guess pressed into the pore wall puts a tail H
    inside covalent range of a framework O, and relaxation pushing it back
    out would read as a bond breaking. The exception is each *binder*, which
    additionally gets its single nearest framework atom -- and ``binders``
    comes from the species record, never from geometry, because a clashing
    guess can put a tail atom as close to the wall as the binder is.
    """
    nl = neighbor_list(atoms, mult=mult, skin=0.0)
    framework = framework_indices(atoms)
    adsorbate = adsorbate_indices(atoms, framework)
    adsorbate_set, binder_set = set(adsorbate), set(binders)

    bond_map = {}
    for i in adsorbate:
        neighbors = [j for j in bonded_neighbors(nl, i) if j in adsorbate_set]
        if i in binder_set and framework:
            site_atom, _ = nearest_atom(atoms, i, framework)
            neighbors = sorted(neighbors + [site_atom])
        bond_map[i] = neighbors
    return bond_map


def bonds_survived(initial, relaxed, bond_map, threshold=BOND_CHANGE_THRESHOLD):
    """(survived, broken) -- ``broken`` lists (i, j, d_initial, d_relaxed,
    change) for every bond that moved more than ``threshold``, largest
    change first.

    Each adsorbate-adsorbate bond appears twice in the map (once under each
    atom), so pairs are collapsed first. 0.5 A is a bond criterion: a C-C at
    1.53 A reaching 2.0 A has broken.
    """
    pairs = sorted({(min(i, j), max(i, j))
                    for i, neighbors in bond_map.items() for j in neighbors})
    broken = []
    for i, j in pairs:
        d_initial = initial.get_distance(i, j, mic=True)
        d_relaxed = relaxed.get_distance(i, j, mic=True)
        if abs(d_relaxed - d_initial) > threshold:
            broken.append((i, j, d_initial, d_relaxed, d_relaxed - d_initial))
    broken.sort(key=lambda row: -abs(row[4]))
    return not broken, broken


def relaxation_survived(initial, relaxed, binders, threshold=BOND_CHANGE_THRESHOLD):
    """Whether a relaxation from ``initial`` to ``relaxed`` kept every bond of
    the adsorbate (``binders`` are the absolute indices of its binding
    atoms, see :func:`adsorbates.binder_indices`)."""
    survived, _ = bonds_survived(initial, relaxed,
                                 adsorbate_bond_map(initial, binders), threshold)
    return survived


# --------------------------------------------------------------------------
# deduplication
# --------------------------------------------------------------------------

def sort_by_energy(entries):
    """Sort ``[(stem, atoms, energy)]`` energy first, missing energies last
    in stem order. Returns a new list."""
    return sorted(entries, key=lambda entry: (entry[2] is None,
                                              0.0 if entry[2] is None else entry[2],
                                              entry[0]))


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
