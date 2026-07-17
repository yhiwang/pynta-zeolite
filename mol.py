import numpy as np
from ase.neighborlist import natural_cutoffs, NeighborList
from functions_v2 import adsorbate_framework_distances


def bonded_neighbors(ads, binder_idx, mult=1.0):
    """Indices bonded to ads[binder_idx], via an ASE neighbor list with
    covalent-radii cutoffs. Evaluated non-periodically so only true
    intramolecular bonds are returned (no periodic-image bonds)."""
    mol = ads.copy()
    mol.pbc = False                                   # ignore any adsorbate cell
    cutoffs = natural_cutoffs(mol, mult=mult)         # per-atom covalent radii * mult
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(mol)
    neighbor_idxs, _ = nl.get_neighbors(binder_idx)
    return list(neighbor_idxs)


def open_direction(ads, binder_idx, neighbor_idxs, iso_tol=0.15):
    """Unit vector from the binder toward its least-crowded direction.
    None when undefined (single atom, or a fully-surrounded/isotropic center)."""
    # (1) single-atom adsorbate: no neighbors, orientation is meaningless
    if not neighbor_idxs:
        return None

    P = ads.get_positions()
    b = P[binder_idx]
    bond_dirs = np.array([(P[j] - b) / np.linalg.norm(P[j] - b) for j in neighbor_idxs])

    # (2) clear gap: opposite the sum of the existing bonds
    open_dir = -bond_dirs.sum(axis=0)
    if np.linalg.norm(open_dir) >= 1e-3:
        return open_dir / np.linalg.norm(open_dir)

    # (3) bonds balanced -> coverage tensor M = sum d d^T; least-covered eigenvector
    M = bond_dirs.T @ bond_dirs
    evals, evecs = np.linalg.eigh(M)                  # ascending eigenvalues
    evals = evals / len(bond_dirs)                    # normalize by #bonds
    if evals[-1] - evals[0] < iso_tol:                # isotropic -> fully surrounded
        return None                                   # tetrahedral / octahedral
    axis = evecs[:, 0]                                # least-covered direction
    com_offset = b - ads.get_center_of_mass()         # sign-breaker only
    if np.dot(axis, com_offset) < 0:
        axis = -axis
    return axis / np.linalg.norm(axis)


# def add_adsorbate_to_site_z(atoms, adsorbate, surf_ind, site, height,
#                             neighbor_idxs=None, orientation=None, tilt_angle=0.):
#     """Place `adsorbate` so atom `surf_ind` sits at the site with its open
#     coordination slot pointing at the site (down the -normal).

#     atoms       : ase.Atoms  framework; MUTATED in place
#     adsorbate   : ase.Atoms  molecule to place
#     surf_ind    : int        binding atom index within adsorbate
#     site        : dict       needs 'position' and 'normal'
#     height      : float      bond distance along the normal
#     neighbor_idxs : list[int] atoms bonded to the binder; if None, inferred by geometry
#     """
#     ads = adsorbate.copy()

#     normal = np.array(site['normal'], dtype=float)
#     if np.isnan(np.sum(normal)):
#         normal = np.array([0., 0., 1.])
#     pos = np.array(site['position'], dtype=float) + normal * height

#     # deterministic orientation: aim the binder's open slot at the site
#     if neighbor_idxs is None:
#         neighbor_idxs = bonded_neighbors(ads, surf_ind)
#     d = open_direction(ads, surf_ind, neighbor_idxs)
#     if d is not None:
#         ads.rotate(d, -normal, center=ads.get_positions()[surf_ind])  # BEFORE the translate

#     # pin the binder exactly at the site
#     ads.translate(pos - ads[surf_ind].position)
#     atoms += ads


def add_adsorbate_to_site_z(atoms, adsorbate, surf_ind, site, height,
                            neighbor_idxs=None, orientation=None, tilt_angle=0.,
                            n_angles=24):
    """Place `adsorbate` so atom `surf_ind` sits at the site with its open slot
    pointing at the site, then spin about the normal and keep the orientation with
    the most tail clearance from the framework. Mutates `atoms` in place."""
    ads = adsorbate.copy()

    normal = np.array(site['normal'], dtype=float)
    if np.isnan(np.sum(normal)):
        normal = np.array([0., 0., 1.])
    pos = np.array(site['position'], dtype=float) + normal * height

    # orient the open slot at the site, once
    if neighbor_idxs is None:
        neighbor_idxs = bonded_neighbors(ads, surf_ind)
    d = open_direction(ads, surf_ind, neighbor_idxs)
    if d is not None:
        ads.rotate(d, -normal, center=ads.get_positions()[surf_ind])

    # sweep the spin about the normal; keep the max-clearance orientation
    tail = [i for i in range(len(ads)) if i != surf_ind]      # exclude the pinned binder
    best_trial, best_score = None, -np.inf
    for k in range(n_angles):
        angle = k * (360.0 / n_angles)
        trial = ads.copy()
        trial.rotate(angle, normal, center=trial.get_positions()[surf_ind])   # spin
        trial.translate(pos - trial[surf_ind].position)                        # pin binder
        D = adsorbate_framework_distances(trial, atoms)       # atoms == framework here
        score = D[tail].min(axis=1).sum()                     # sum of per-tail-atom nearest-wall dist
        if score > best_score:
            best_score, best_trial = score, trial

    atoms += best_trial                                       # merge the winner, in place