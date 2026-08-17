import numpy as np
from ase.data import covalent_radii
from ase.neighborlist import NeighborList, natural_cutoffs


def find_overlaps(combined, framework_indices, adsorbate_indices,
                  tol=0.7, ignore_pairs=None, show=False):
    """Clashing (adsorbate_idx, framework_idx) pairs in COMBINED index space.

    Empty list means no overlap. A hard boolean screen, unlike the maximin
    score the placement sweep ranks by. `combined` must carry the framework
    cell and pbc for mic to work."""
    ignore = set(ignore_pairs) if ignore_pairs else set()
    D = combined.get_all_distances(mic=True)
    radii = covalent_radii[combined.get_atomic_numbers()]
    overlaps = []
    for a in adsorbate_indices:
        for f in framework_indices:
            if (a, f) in ignore:
                continue
            if D[a, f] < tol * (radii[a] + radii[f]):
                overlaps.append((a, f))
                if show:
                    print("  overlap: ads %d <-> fw %d  d=%.2f A"
                          % (a, f, D[a, f]))
    return overlaps


def check_for_overlaps(combined, framework_indices, adsorbate_indices,
                       tol=0.7, ignore_pairs=None, show=False):
    return len(find_overlaps(combined, framework_indices, adsorbate_indices,
                             tol=tol, ignore_pairs=ignore_pairs,
                             show=show)) > 0


def adsorbate_framework_distances(placed_ads, framework):
    """(n_ads, n_fw) minimum-image distance block. Framework first so the
    combined keeps the framework cell and pbc."""
    combined = framework + placed_ads
    n_fw = len(framework)
    return combined.get_all_distances(mic=True)[n_fw:, :n_fw]


def closest_framework_per_atom(placed_ads, framework):
    """Per adsorbate atom, its nearest framework atom as (index, distance)."""
    D = adsorbate_framework_distances(placed_ads, framework)
    return [(int(np.argmin(D[a])), D[a, int(np.argmin(D[a]))])
            for a in range(D.shape[0])]


def get_center_of_mass_pbc(atoms):
    """Mass-weighted COM robust to wrap-around, via the circular mean of the
    fractional coordinates. Use this instead of get_center_of_mass() whenever
    the group might straddle a cell boundary."""
    masses = atoms.get_masses()
    total_mass = masses.sum()
    cell = atoms.get_cell()
    frac = np.linalg.solve(cell.T, atoms.get_positions().T).T % 1.0
    theta = 2 * np.pi * frac
    sin_sum = np.sum(np.sin(theta.T) * masses, axis=1) / total_mass
    cos_sum = np.sum(np.cos(theta.T) * masses, axis=1) / total_mass
    avg_theta = np.arctan2(sin_sum, cos_sum) / (2 * np.pi) % 1.0
    return np.dot(avg_theta, cell)


def find_framework_indices(atoms, elements=("Si", "O", "Al"), min_size=10):
    """Indices of the largest connected cluster of `elements`.

    mult=1.0 is required, not stylistic: a looser cutoff starts reporting
    non-bonded contacts as bonds and fuses the adsorbate into the framework
    cluster. Returns [] when nothing reaches min_size (gas phase)."""
    nl = NeighborList(natural_cutoffs(atoms, mult=1.0), skin=0.0,
                      self_interaction=False, bothways=True)
    nl.update(atoms)
    allowed = {i for i, a in enumerate(atoms) if a.symbol in elements}

    seen, clusters = set(), []
    for start in sorted(allowed):
        if start in seen:
            continue
        cluster, stack = set(), [start]
        seen.add(start)
        while stack:
            current = stack.pop()
            cluster.add(current)
            for neighbor in nl.get_neighbors(current)[0]:
                neighbor = int(neighbor)
                if neighbor in allowed and neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        clusters.append(cluster)

    if not clusters:
        return []
    biggest = max(clusters, key=len)
    return sorted(biggest) if len(biggest) >= min_size else []


def extra_framework_indices(atoms):
    framework = set(find_framework_indices(atoms))
    return [i for i in range(len(atoms)) if i not in framework]


def nearest_framework(atoms, index, framework):
    distances = atoms.get_distances(index, framework, mic=True)
    k = int(np.argmin(distances))
    return framework[k], float(distances[k])


def adsorbate_neighbors(nl, index, adsorbate_set):
    """Bonded neighbors of an adsorbate atom, adsorbate atoms only.

    Framework partners are excluded on purpose: a guess pressed into the pore
    wall puts a tail H inside covalent range of a framework O, and relaxation
    pushing it back out would read as a bond breaking."""
    return sorted(j for j in map(int, nl.get_neighbors(index)[0])
                  if j in adsorbate_set)


def binder_neighbors(atoms, nl, binder, adsorbate_set, framework):
    """Adsorbate neighbors of a binder, plus the single nearest framework
    atom rather than everything inside the cutoff."""
    site, _ = nearest_framework(atoms, binder, framework)
    return sorted(adsorbate_neighbors(nl, binder, adsorbate_set) + [site])


def neighbor_map(atoms, binders, mult=1.2):
    """{adsorbate index: [bonded neighbors]} for every adsorbate atom.

    mult=1.2, not the 1.0 used by find_framework_indices: covalent radii sum
    to 1.07 A for C-H, under the real bond length, so a tight cutoff returns
    no molecular bonds at all. The two values are opposed on purpose.

    `binders` comes from info.json, never from geometry -- a clashing guess
    can put a tail atom as close to the wall as the binder is."""
    nl = NeighborList(natural_cutoffs(atoms, mult=mult), skin=0.0,
                      self_interaction=False, bothways=True)
    nl.update(atoms)

    framework = find_framework_indices(atoms)
    adsorbate = extra_framework_indices(atoms)
    adsorbate_set, binder_set = set(adsorbate), set(binders)

    neighbors = {}
    for i in adsorbate:
        if i in binder_set and framework:
            neighbors[i] = binder_neighbors(atoms, nl, i, adsorbate_set,
                                            framework)
        else:
            neighbors[i] = adsorbate_neighbors(nl, i, adsorbate_set)
    return neighbors


def check_config_survived(initial, relaxed, neighbors, threshold=0.5):
    """(survived, rows) for the bonds in `neighbors`, rows being
    (i, j, d0, d1, delta) for every bond that moved more than threshold,
    largest change first.

    Each adsorbate-adsorbate bond appears twice in the map, once under each
    atom, so pairs are collapsed first. threshold=0.5 because these are
    bonds: a C-C at 1.53 A reaching 2.0 A has broken."""
    pairs = sorted({(min(i, j), max(i, j))
                    for i, js in neighbors.items() for j in js})
    rows = []
    for i, j in pairs:
        d0 = initial.get_distance(i, j, mic=True)
        d1 = relaxed.get_distance(i, j, mic=True)
        if abs(d1 - d0) > threshold:
            rows.append((i, j, d0, d1, d1 - d0))
    rows.sort(key=lambda row: -abs(row[4]))
    return not rows, rows