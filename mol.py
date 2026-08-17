import numpy as np
from ase.geometry import get_distances
from ase.neighborlist import natural_cutoffs, NeighborList


def bonded_neighbors(ads, binder_idx, mult=1.0):
    mol = ads.copy()
    mol.pbc = False
    nl = NeighborList(natural_cutoffs(mol, mult=mult),
                      self_interaction=False, bothways=True)
    nl.update(mol)
    return list(nl.get_neighbors(binder_idx)[0])


def open_direction(ads, binder_idx, neighbor_idxs, iso_tol=0.15):
    """Unit vector from the binder toward its least-crowded direction, or
    None when undefined: a lone atom, or a fully surrounded center."""
    if not neighbor_idxs:
        return None

    P = ads.get_positions()
    b = P[binder_idx]
    bond_dirs = np.array([(P[j] - b) / np.linalg.norm(P[j] - b)
                          for j in neighbor_idxs])

    open_dir = -bond_dirs.sum(axis=0)
    if np.linalg.norm(open_dir) >= 1e-3:
        return open_dir / np.linalg.norm(open_dir)

    M = bond_dirs.T @ bond_dirs
    evals, evecs = np.linalg.eigh(M)
    evals = evals / len(bond_dirs)
    if evals[-1] - evals[0] < iso_tol:
        return None
    axis = evecs[:, 0]
    if np.dot(axis, b - ads.get_center_of_mass()) < 0:
        axis = -axis
    return axis / np.linalg.norm(axis)


def reference_neighbor(ads, binder_idx, mult=1.0):
    """The binder neighbor defining the molecule's zero degree: the one with
    the most heavy atoms hanging off it, never crossing back through the
    binder. Heavy neighbors before hydrogens, lowest index breaking ties."""
    neighbors = bonded_neighbors(ads, binder_idx, mult)
    if not neighbors:
        return None

    symbols = ads.get_chemical_symbols()
    non_H_neighbors = [j for j in neighbors if symbols[j] != "H"]
    search_list = non_H_neighbors or neighbors

    best, best_key = None, None
    for j in search_list:
        seen, stack, n_heavy = {binder_idx, j}, [j], 0
        while stack:
            i = stack.pop()
            if symbols[i] != "H":
                n_heavy += 1
            for k in bonded_neighbors(ads, i, mult):
                if k not in seen:
                    seen.add(k)
                    stack.append(k)
        key = (-n_heavy, int(j))
        if best_key is None or key < best_key:
            best_key, best = key, int(j)
    return best


def _rot_about(v, axis, angle_deg):
    th = np.radians(angle_deg)
    return (v * np.cos(th)
            + np.cross(axis, v) * np.sin(th)
            + axis * np.dot(axis, v) * (1.0 - np.cos(th)))


def _tail_clearance(trial, fw_pos, cell, pbc, tail):
    if not tail:
        return 0.0
    _, D = get_distances(trial.get_positions(), fw_pos, cell=cell, pbc=pbc)
    return float(D[tail].min())


def _combine(framework, trials):
    structures = []
    for trial in trials:
        combined = framework.copy()
        combined += trial
        structures.append(combined)
    return structures


def add_adsorbate_to_site_z(atoms, adsorbate, surf_ind, site, height,
                            neighbor_idxs=None, n_angles=24, best_only=True):
    """Pin `surf_ind` at the site with its open slot facing in, zero the
    azimuth, then sweep about the normal.

    Zero degrees puts the binder's heaviest branch along the site's
    tangent_ref, measured right-handed about the normal, so angle k means
    the same thing across sites and species.

    Returns (structures, tags, scores) with one entry each, or the whole
    sweep when best_only is False. Does not mutate `atoms`."""
    ads = adsorbate.copy()

    normal = np.array(site["normal"], dtype=float)
    pos = np.array(site["position"], dtype=float) + normal * height

    if neighbor_idxs is None:
        neighbor_idxs = bonded_neighbors(ads, surf_ind)
    d = open_direction(ads, surf_ind, neighbor_idxs)
    if d is not None:
        ads.rotate(d, -normal, center=ads.get_positions()[surf_ind])

    ref = reference_neighbor(ads, surf_ind)
    site_ref = site.get("tangent_ref")
    if ref is not None and site_ref is not None:
        P = ads.get_positions()
        v = P[ref] - P[surf_ind]
        mol_ref = v - np.dot(v, normal) * normal
        if np.linalg.norm(mol_ref) > 0.1:
            mol_ref = mol_ref / np.linalg.norm(mol_ref)
            azimuth = np.degrees(np.arctan2(
                np.dot(np.cross(site_ref, mol_ref), normal),
                np.dot(site_ref, mol_ref)))
            ads.rotate(-azimuth, normal, center=P[surf_ind])

    tail = [i for i in range(len(ads)) if i != surf_ind]
    if not tail:
        n_angles = 1

    fw_pos = atoms.get_positions()
    cell, pbc = atoms.cell, atoms.pbc

    trials, tags, scores = [], [], []
    for k in range(n_angles):
        angle = k * (360.0 / n_angles)
        trial = ads.copy()
        trial.rotate(angle, normal, center=trial.get_positions()[surf_ind])
        trial.translate(pos - trial[surf_ind].position)

        trials.append(trial)
        tags.append({"angle": angle})
        scores.append(_tail_clearance(trial, fw_pos, cell, pbc, tail))

    if best_only:
        k = int(np.argmax(scores))
        trials, tags, scores = [trials[k]], [tags[k]], [scores[k]]

    return _combine(atoms, trials), tags, scores


def add_adsorbate_to_double_site_z(atoms, adsorbate, surf_inds, sites, h1, h2,
                                   n_big=24, n_small=12, try_flip=True,
                                   best_only=True):
    """Bridge `adsorbate` across two framework oxygens.

    phi sweeps the binder pair about the O1-O2 axis; psi rolls the molecule
    about its own binder-binder axis, moving only the tail. The perpendicular
    depth t is forced by the two bond lengths rather than guessed, so every
    phi gives exact bonds. flip swaps which binder sits over which oxygen.

    Returns (structures, tags, scores) with one entry each, or the whole
    sweep when best_only is False. Does not mutate `atoms`."""
    ads0 = adsorbate.copy()
    b1, b2 = surf_inds[0], surf_inds[1]
    tail = [i for i in range(len(ads0)) if i not in (b1, b2)]

    o1 = np.array(sites[0]["position"], dtype=float)
    o2 = np.array(sites[1]["position"], dtype=float)
    d_OO = np.linalg.norm(o2 - o1)
    u = (o2 - o1) / d_OO

    P = ads0.get_positions()
    d_mol = np.linalg.norm(P[b2] - P[b1])
    delta = 0.5 * (d_OO - d_mol)

    ref = np.array([0., 0., 1.])
    if abs(np.dot(ref, u)) > 0.9:
        ref = np.array([1., 0., 0.])
    n0 = ref - np.dot(ref, u) * u
    n0 /= np.linalg.norm(n0)

    fw_pos = atoms.get_positions()
    cell, pbc = atoms.cell, atoms.pbc

    trials, tags, scores = [], [], []

    for flip in ((False, True) if try_flip else (False,)):
        uu = -u if flip else u
        hA, hB = (h2, h1) if flip else (h1, h2)

        # hA^2 = a^2 + t^2 and hB^2 = (2*delta - a)^2 + t^2, solved for the
        # along-axis offset a from O1 to the binder above it
        a = 0.0 if abs(delta) < 1e-8 else \
            (hA * hA - hB * hB + 4.0 * delta * delta) / (4.0 * delta)
        t2 = hA * hA - a * a
        if t2 <= 0.0:
            raise ValueError("molecule span %.2f A incompatible with d(O-O) "
                             "%.2f A at bond lengths %.2f/%.2f A"
                             % (d_mol, d_OO, hA, hB))
        t = np.sqrt(t2)
        mid_off = a + 0.5 * d_mol

        base = ads0.copy()
        Pb = base.get_positions()
        base.rotate(Pb[b2] - Pb[b1], uu, center=0.5 * (Pb[b1] + Pb[b2]))

        for i in range(n_big):
            phi = i * (360.0 / n_big)
            target_mid = o1 + mid_off * u + t * _rot_about(n0, u, phi)

            for j in range(n_small):
                psi = j * (360.0 / n_small)
                trial = base.copy()
                Pt = trial.get_positions()
                bm = 0.5 * (Pt[b1] + Pt[b2])
                trial.rotate(psi, uu, center=bm)
                trial.translate(target_mid - bm)

                trials.append(trial)
                tags.append({"flip": flip, "phi": phi, "psi": psi, "t": t})
                scores.append(_tail_clearance(trial, fw_pos, cell, pbc, tail))

    if best_only:
        k = int(np.argmax(scores))
        trials, tags, scores = [trials[k]], [tags[k]], [scores[k]]

    return _combine(atoms, trials), tags, scores