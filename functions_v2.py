import numpy as np
import random
from ase.data import covalent_radii

# ============================================================
# Overlap detection  (used by add_adsorbate_to_site_z)
# ============================================================

def find_overlaps(combined, framework_indices, adsorbate_indices,
                  tol=0.7, ignore_pairs=None, show=False):
    """List of clashing (adsorbate_idx, framework_idx) pairs in COMBINED index
    space, excluding ignore_pairs. Empty list => no overlap. Reads only.
    `combined` must carry the framework cell/pbc for mic to work."""
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
                    print(f"  overlap: ads {a} <-> fw {f}  d={D[a, f]:.2f} A")
    return overlaps


def check_for_overlaps(combined, framework_indices, adsorbate_indices,
                       tol=0.7, ignore_pairs=None, show=False):
    """True if any adsorbate-framework clash remains after ignoring ignore_pairs."""
    return len(find_overlaps(combined, framework_indices, adsorbate_indices,
                             tol=tol, ignore_pairs=ignore_pairs, show=show)) > 0


# ============================================================
# Adsorbate <-> framework distances  (clearance diagnostics)
# ============================================================

def adsorbate_framework_distances(placed_ads, framework):
    """(n_ads, n_fw) minimum-image distance block. D[a, f] = distance from
    adsorbate atom a to framework atom f. Framework-first so the combined keeps
    the framework cell/pbc (needed for mic)."""
    combined = framework + placed_ads
    n_fw = len(framework)
    return combined.get_all_distances(mic=True)[n_fw:, :n_fw]


def closest_framework_per_atom(placed_ads, framework):
    """Per adsorbate atom, its nearest framework atom as (framework_idx, distance).
    Used to check that the binding atom is the closest atom to the wall."""
    D = adsorbate_framework_distances(placed_ads, framework)
    result = []
    for a in range(D.shape[0]):
        f_idx = int(np.argmin(D[a]))     # nearest framework atom to adsorbate atom a
        result.append((f_idx, D[a, f_idx]))
    return result


# ============================================================
# Rotation utilities
# ============================================================

def get_center_of_mass_pbc(atoms):
    """Mass-weighted COM robust to periodic wrap-around (circular mean of
    fractional coordinates)."""
    masses = atoms.get_masses()
    total_mass = masses.sum()
    cell = atoms.get_cell()
    frac = np.linalg.solve(cell.T, atoms.get_positions().T).T % 1.0
    theta = 2 * np.pi * frac
    sin_sum = np.sum(np.sin(theta.T) * masses, axis=1) / total_mass
    cos_sum = np.sum(np.cos(theta.T) * masses, axis=1) / total_mass
    avg_theta = np.arctan2(sin_sum, cos_sum) / (2 * np.pi) % 1.0
    return np.dot(avg_theta, cell)


def rotate_free(ads):
    """Rigid RANDOM rotation of an isolated molecule about its own COM, no wrap.
    NOTE: not used by the deterministic spin-about-normal placement; keep only if
    you need random orientation sampling elsewhere."""
    m = ads.copy()
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    m.rotate(random.uniform(-180, 180), axis, center=m.get_center_of_mass())
    return m


# DEPRECATED: random_rotate wraps atoms (atoms.wrap()) and can split a molecule
# across the cell boundary -- the failure you moved away from. Left out of the
# active file on purpose. Re-add only if you truly need in-cell rotation, and
# unwrap the molecule afterward.