"""Geometry helpers shared by every stage: neighbor lists, telling the
framework apart from the adsorbate, clash and clearance tests, minimum-image
vectors, RMSD and Kabsch superposition, and rotations.

Only ASE and numpy here -- no RMG, no maze -- so the relax worker on a
compute node can import it without the heavy dependencies.
"""

import numpy as np
from ase.data import covalent_radii
from ase.geometry import find_mic, get_distances
from ase.neighborlist import NeighborList, natural_cutoffs

FRAMEWORK_ELEMENTS = ("Si", "O", "Al")
MIN_FRAMEWORK_SIZE = 10     # fewer connected Si/O/Al than this is not a framework


# --------------------------------------------------------------------------
# neighbor lists and connectivity
# --------------------------------------------------------------------------

def neighbor_list(atoms, mult=1.0, skin=0.0, periodic=None):
    """Bonded-neighbor list from covalent radii.

    ``mult`` scales the covalent radii and ``skin`` is added on top; both are
    part of every caller's chemistry, so there is no default that fits all:

    * framework detection uses ``mult=1.0, skin=0.0`` -- anything looser starts
      reporting non-bonded contacts as bonds and fuses the adsorbate into the
      framework cluster;
    * adsorbate bond maps use ``mult=1.2`` -- covalent radii sum to 1.07 A for
      C-H, under the real bond length, so a tight cutoff finds no bonds at all;
    * molecule-only questions (which atoms hang off the binder) use
      ``skin=0.3, periodic=False`` on an isolated copy.

    ``periodic=False`` builds the list on a non-periodic copy of ``atoms``.
    """
    if periodic is False:
        atoms = atoms.copy()
        atoms.pbc = False
    nl = NeighborList(natural_cutoffs(atoms, mult=mult), skin=skin,
                      self_interaction=False, bothways=True)
    nl.update(atoms)
    return nl


def bonded_neighbors(nl, index):
    """Sorted neighbor indices of one atom from a neighbor list."""
    return sorted(int(j) for j in nl.get_neighbors(index)[0])


def connected_components(node_indices, neighbors_of):
    """Connected components of a graph given by a neighbor function.

    ``node_indices`` is the iterable of nodes to visit (in that order) and
    ``neighbors_of(node)`` returns the nodes bonded to it. Neighbors that are
    not in ``node_indices`` are ignored, which is how callers restrict the
    graph to a subset (framework elements only, adsorbate atoms only).
    Returns a list of sorted index lists, one per component, in first-seen
    order.
    """
    allowed = set(node_indices)
    seen, components = set(), []
    for start in node_indices:
        if start in seen:
            continue
        component, stack = [], [start]
        seen.add(start)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in neighbors_of(current):
                if neighbor in allowed and neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


# --------------------------------------------------------------------------
# framework vs adsorbate
# --------------------------------------------------------------------------

def framework_indices(atoms, elements=FRAMEWORK_ELEMENTS,
                      min_size=MIN_FRAMEWORK_SIZE):
    """Indices of the largest connected cluster of framework elements.

    Found by connectivity rather than by index, so an adsorbate that contains
    oxygen forms its own small cluster and is not mistaken for framework.
    Returns [] when nothing reaches ``min_size`` (a gas-phase molecule).
    """
    nl = neighbor_list(atoms, mult=1.0, skin=0.0)
    candidates = sorted(i for i, atom in enumerate(atoms)
                        if atom.symbol in elements)
    clusters = connected_components(candidates,
                                    lambda i: bonded_neighbors(nl, i))
    if not clusters:
        return []
    biggest = max(clusters, key=len)
    return biggest if len(biggest) >= min_size else []


def adsorbate_indices(atoms, framework=None):
    """Every index that is not framework."""
    if framework is None:
        framework = framework_indices(atoms)
    framework = set(framework)
    return [i for i in range(len(atoms)) if i not in framework]


def nearest_atom(atoms, index, candidates):
    """(candidate index, distance) of the closest of ``candidates`` to
    ``index``, minimum image."""
    candidates = list(candidates)
    distances = atoms.get_distances(index, candidates, mic=True)
    closest = int(np.argmin(distances))
    return candidates[closest], float(distances[closest])


def adsorbate_framework_distances(adsorbate, framework):
    """(n_adsorbate, n_framework) minimum-image distance block between a
    placed adsorbate and the framework, using the framework cell."""
    combined = framework + adsorbate
    n_framework = len(framework)
    return combined.get_all_distances(mic=True)[n_framework:, :n_framework]


def closest_framework_per_atom(adsorbate, framework):
    """Per adsorbate atom, (framework index, distance) of its nearest
    framework atom."""
    distances = adsorbate_framework_distances(adsorbate, framework)
    return [(int(np.argmin(row)), float(row.min())) for row in distances]


def center_of_mass_pbc(atoms):
    """Mass-weighted centre of mass robust to wrap-around, via the circular
    mean of the fractional coordinates. Use it instead of
    ``get_center_of_mass()`` when the group might straddle a cell boundary."""
    masses = atoms.get_masses()
    total_mass = masses.sum()
    cell = atoms.get_cell()
    fractional = np.linalg.solve(cell.T, atoms.get_positions().T).T % 1.0
    theta = 2 * np.pi * fractional
    sin_mean = np.sum(np.sin(theta.T) * masses, axis=1) / total_mass
    cos_mean = np.sum(np.cos(theta.T) * masses, axis=1) / total_mass
    mean_fractional = np.arctan2(sin_mean, cos_mean) / (2 * np.pi) % 1.0
    return np.dot(mean_fractional, cell)


# --------------------------------------------------------------------------
# clashes and clearance
# --------------------------------------------------------------------------

def find_clashes(atoms, group_a, group_b, tolerance=0.7, ignore_pairs=None,
                 show=False):
    """[(i, j)] pairs from ``group_a`` x ``group_b`` closer than
    ``tolerance`` times the sum of their covalent radii, minimum image.

    A hard boolean screen, unlike the clearance score the placement sweep
    ranks by. ``atoms`` must carry the cell and pbc for mic to work.
    """
    ignore = set(ignore_pairs) if ignore_pairs else set()
    distances = atoms.get_all_distances(mic=True)
    radii = covalent_radii[atoms.get_atomic_numbers()]
    clashes = []
    for i in group_a:
        for j in group_b:
            if (i, j) in ignore:
                continue
            if distances[i, j] < tolerance * (radii[i] + radii[j]):
                clashes.append((i, j))
                if show:
                    print("  clash: %d <-> %d  d=%.2f A" % (i, j, distances[i, j]))
    return clashes


def min_clearance(points, obstacles, cell, pbc):
    """Smallest minimum-image distance from any of ``points`` (n,3) to any of
    ``obstacles`` (m,3). The score every orientation sweep is ranked by:
    larger means the tail sits further from the wall."""
    if len(points) == 0:
        return 0.0
    _, distances = get_distances(points, obstacles, cell=cell, pbc=pbc)
    return float(distances.min())


# --------------------------------------------------------------------------
# vectors, rotations, superposition
# --------------------------------------------------------------------------

def mic_unit_vector(atoms, index_from, index_to):
    """(unit vector, distance) from ``index_from`` to ``index_to`` under the
    minimum-image convention."""
    delta = np.array([atoms.positions[index_to] - atoms.positions[index_from]])
    vectors, lengths = find_mic(delta, atoms.cell, atoms.pbc)
    return vectors[0] / lengths[0], float(lengths[0])


def unit(vector):
    """``vector`` scaled to length one."""
    return vector / np.linalg.norm(vector)


def rotate_vector(vector, axis, angle_deg):
    """Rodrigues rotation of ``vector`` about the unit ``axis``."""
    theta = np.radians(angle_deg)
    return (vector * np.cos(theta)
            + np.cross(axis, vector) * np.sin(theta)
            + axis * np.dot(axis, vector) * (1.0 - np.cos(theta)))


def positional_rmsd(atoms_a, atoms_b, indices):
    """RMSD over ``indices`` without superposition, minimum image.

    Meaningful only because the framework is frozen: both structures sit in
    the same frame, so every displacement is real and not a rigid-body
    offset."""
    delta = atoms_b.get_positions()[indices] - atoms_a.get_positions()[indices]
    _, lengths = find_mic(delta, atoms_a.cell, atoms_a.pbc)
    return float(np.sqrt((lengths ** 2).mean()))


def kabsch_transform(source, target):
    """(rotation, translation) with ``rotation @ x + translation`` taking the
    ``source`` point cloud onto ``target``, both (n, 3)."""
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    u_matrix, _, vt_matrix = np.linalg.svd(
        (source - source_center).T @ (target - target_center))
    sign_fix = np.diag([1.0, 1.0,
                        np.sign(np.linalg.det(vt_matrix.T @ u_matrix.T))])
    rotation = vt_matrix.T @ sign_fix @ u_matrix.T
    return rotation, target_center - rotation @ source_center


def superposed_rmsd(source, target):
    """RMSD between two (n, 3) clouds after optimal superposition."""
    rotation, translation = kabsch_transform(source, target)
    moved = source @ rotation.T + translation
    return float(np.sqrt(((moved - target) ** 2).sum(axis=1).mean()))


def cloud_rmsd(source, target):
    """RMSD between two (n, 3) clouds as they stand (no superposition)."""
    return float(np.sqrt(((source - target) ** 2).sum(axis=1).mean()))
