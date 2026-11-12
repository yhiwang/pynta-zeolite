"""Put one adsorbate onto one site (or one pair of sites) of the framework
and sweep its orientations.

Monodentate: the binding atom is pinned above the oxygen along the site
normal, the molecule's open slot is turned to face the framework, its
heaviest branch is aligned with the site's ``tangent_ref`` (zero degrees),
and the whole molecule is spun about the normal in ``n_angles`` steps.

Bidentate: the two binding atoms bridge two oxygens; ``phi`` swings the
binder pair around the O-O axis, ``psi`` rolls the molecule about its own
binder-binder axis, and ``flip`` swaps which binder sits on which oxygen.

Every orientation is scored by its tail clearance (smallest distance from
any non-binding atom to the framework); callers may keep the best or the
whole sweep. Each orientation is identified by a *tag* dict, and
:func:`orientation_stem` / :func:`parse_orientation_stem` convert a tag to
the directory name used on disk and back.
"""

import re

import numpy as np

from .geometry import (neighbor_list, min_clearance, rotate_vector, unit)

# sweep sizes
MONODENTATE_ANGLES = 24     # spin steps about the site normal (15 deg)
BIDENTATE_PHI_STEPS = 24    # swing about the O-O axis (15 deg)
BIDENTATE_PSI_STEPS = 12    # roll about the binder-binder axis (30 deg)

GAS_STEM = "gas"            # the one "orientation" of a gas-phase molecule

_MONO_STEM = re.compile(r"^degrees_(\d+)$")
_BI_STEM = re.compile(r"^flip(\d)_phi(\d+)_psi(\d+)$")


# --------------------------------------------------------------------------
# orientation tags <-> directory stems
# --------------------------------------------------------------------------

def orientation_stem(tag):
    """Directory / file stem for one orientation, from its tag.

    Zero-padded so ``ls`` sorts numerically. The stem is the only record of
    which orientation a config is, so it must round-trip: monodentate angles
    are multiples of 15 deg and bidentate phi/psi of 15/30 deg, so rounding
    to whole degrees never collides. A gas-phase (empty) tag gives ``gas``.
    """
    if not tag:
        return GAS_STEM
    if "angle" in tag:
        return "degrees_%03d" % round(tag["angle"])
    return "flip%d_phi%03d_psi%03d" % (int(tag["flip"]),
                                       round(tag["phi"]), round(tag["psi"]))


def parse_orientation_stem(stem):
    """Tag dict from a stem: {"angle": 45} / {"flip": 0, "phi": 105,
    "psi": 240} / {} for gas, or None when the stem is not recognised."""
    if stem == GAS_STEM:
        return {}
    match = _MONO_STEM.match(stem)
    if match:
        return {"angle": int(match.group(1))}
    match = _BI_STEM.match(stem)
    if match:
        return {"flip": int(match.group(1)), "phi": int(match.group(2)),
                "psi": int(match.group(3))}
    return None


def is_monodentate_stem(stem):
    return bool(_MONO_STEM.match(stem))


def is_bidentate_stem(stem):
    return bool(_BI_STEM.match(stem))


# --------------------------------------------------------------------------
# molecule-only questions (no framework involved)
# --------------------------------------------------------------------------

def molecule_neighbor_list(adsorbate):
    """Bond list of the isolated molecule (non-periodic, default ASE skin)."""
    return neighbor_list(adsorbate, mult=1.0, skin=0.3, periodic=False)


def molecule_bonded_neighbors(adsorbate, index, nl=None):
    """Indices bonded to ``index`` in the isolated molecule, in neighbor-list
    order."""
    nl = nl or molecule_neighbor_list(adsorbate)
    return [int(j) for j in nl.get_neighbors(index)[0]]


def open_direction(adsorbate, binder, neighbor_indices, isotropy_tol=0.15):
    """Unit vector from the binder toward its least crowded direction, or
    None when undefined (a lone atom, or a fully surrounded centre).

    Normally the negative sum of the bond directions. When that vanishes
    (planar sp2 centre) the axis of least bond density is used, pointed away
    from the molecule's centre of mass; when the bond directions are
    isotropic there is no open direction at all.
    """
    if not neighbor_indices:
        return None

    positions = adsorbate.get_positions()
    binder_position = positions[binder]
    bond_directions = np.array([unit(positions[j] - binder_position)
                                for j in neighbor_indices])

    open_dir = -bond_directions.sum(axis=0)
    if np.linalg.norm(open_dir) >= 1e-3:
        return unit(open_dir)

    inertia_like = bond_directions.T @ bond_directions
    eigenvalues, eigenvectors = np.linalg.eigh(inertia_like)
    eigenvalues = eigenvalues / len(bond_directions)
    if eigenvalues[-1] - eigenvalues[0] < isotropy_tol:
        return None
    axis = eigenvectors[:, 0]
    if np.dot(axis, binder_position - adsorbate.get_center_of_mass()) < 0:
        axis = -axis
    return unit(axis)


def azimuth_reference_neighbor(adsorbate, binder, nl=None):
    """The binder neighbor that defines the molecule's zero degrees: the one
    with the most heavy atoms hanging off it (never crossing back through
    the binder). Heavy neighbors beat hydrogens; lowest index breaks ties."""
    nl = nl or molecule_neighbor_list(adsorbate)
    neighbors = molecule_bonded_neighbors(adsorbate, binder, nl)
    if not neighbors:
        return None

    symbols = adsorbate.get_chemical_symbols()
    heavy_neighbors = [j for j in neighbors if symbols[j] != "H"]
    candidates = heavy_neighbors or neighbors

    best, best_key = None, None
    for candidate in candidates:
        seen, stack, n_heavy = {binder, candidate}, [candidate], 0
        while stack:
            current = stack.pop()
            if symbols[current] != "H":
                n_heavy += 1
            for neighbor in molecule_bonded_neighbors(adsorbate, current, nl):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        key = (-n_heavy, int(candidate))
        if best_key is None or key < best_key:
            best_key, best = key, int(candidate)
    return best


# --------------------------------------------------------------------------
# placement sweeps
# --------------------------------------------------------------------------

def _tail_clearance(trial, framework_positions, cell, pbc, tail):
    if not tail:
        return 0.0
    return min_clearance(trial.get_positions()[tail], framework_positions,
                         cell, pbc)


def _combine(framework, trials):
    """framework + adsorbate for every trial, framework first so the
    combined structure keeps the framework cell and pbc."""
    structures = []
    for trial in trials:
        combined = framework.copy()
        combined += trial
        structures.append(combined)
    return structures


def _keep_best(trials, tags, scores, best_only):
    if best_only:
        best = int(np.argmax(scores))
        return [trials[best]], [tags[best]], [scores[best]]
    return trials, tags, scores


def place_monodentate(framework, adsorbate, binder, site, bond_length,
                      neighbor_indices=None,
                      n_angles=MONODENTATE_ANGLES, best_only=True):
    """Bind ``adsorbate`` atom ``binder`` on ``site`` and spin it about the
    site normal.

    Zero degrees puts the binder's heaviest branch along the site's
    ``tangent_ref``, measured right-handed about the normal, so angle k means
    the same thing across sites and species.

    Returns ``(structures, tags, scores)`` -- one entry each with
    ``best_only``, otherwise the whole sweep. ``framework`` is not modified.
    """
    molecule = adsorbate.copy()

    normal = np.array(site["normal"], dtype=float)
    binder_target = np.array(site["position"], dtype=float) + normal * bond_length

    if neighbor_indices is None:
        neighbor_indices = molecule_bonded_neighbors(molecule, binder)
    open_dir = open_direction(molecule, binder, neighbor_indices)
    if open_dir is not None:
        molecule.rotate(open_dir, -normal, center=molecule.get_positions()[binder])

    reference = azimuth_reference_neighbor(molecule, binder)
    site_reference = site.get("tangent_ref")
    if reference is not None and site_reference is not None:
        positions = molecule.get_positions()
        branch = positions[reference] - positions[binder]
        branch_in_plane = branch - np.dot(branch, normal) * normal
        if np.linalg.norm(branch_in_plane) > 0.1:
            branch_in_plane = unit(branch_in_plane)
            azimuth = np.degrees(np.arctan2(
                np.dot(np.cross(site_reference, branch_in_plane), normal),
                np.dot(site_reference, branch_in_plane)))
            molecule.rotate(-azimuth, normal, center=positions[binder])

    tail = [i for i in range(len(molecule)) if i != binder]
    if not tail:
        n_angles = 1

    framework_positions = framework.get_positions()
    cell, pbc = framework.cell, framework.pbc

    trials, tags, scores = [], [], []
    for k in range(n_angles):
        angle = k * (360.0 / n_angles)
        trial = molecule.copy()
        trial.rotate(angle, normal, center=trial.get_positions()[binder])
        trial.translate(binder_target - trial[binder].position)

        trials.append(trial)
        tags.append({"angle": angle})
        scores.append(_tail_clearance(trial, framework_positions, cell, pbc, tail))

    trials, tags, scores = _keep_best(trials, tags, scores, best_only)
    return _combine(framework, trials), tags, scores


def place_bidentate(framework, adsorbate, binders, sites, bond_lengths,
                    n_phi=BIDENTATE_PHI_STEPS, n_psi=BIDENTATE_PSI_STEPS,
                    try_flip=True, best_only=True):
    """Bridge ``adsorbate`` across two framework oxygens.

    ``binders`` = (binder_1, binder_2) adsorbate indices, ``sites`` the two
    site dicts, ``bond_lengths`` = (binder_1-O_1, binder_2-O_2) in A.

    ``phi`` sweeps the binder pair about the O1-O2 axis; ``psi`` rolls the
    molecule about its own binder-binder axis, moving only the tail. The
    perpendicular depth is forced by the two bond lengths rather than
    guessed, so every phi gives exact bonds. ``flip`` swaps which binder
    sits over which oxygen.

    Returns ``(structures, tags, scores)`` like :func:`place_monodentate`.
    """
    molecule = adsorbate.copy()
    binder_1, binder_2 = binders
    tail = [i for i in range(len(molecule)) if i not in (binder_1, binder_2)]

    o1_position = np.array(sites[0]["position"], dtype=float)
    o2_position = np.array(sites[1]["position"], dtype=float)
    d_oo = np.linalg.norm(o2_position - o1_position)
    bridge_axis = (o2_position - o1_position) / d_oo

    positions = molecule.get_positions()
    d_binders = np.linalg.norm(positions[binder_2] - positions[binder_1])
    half_excess = 0.5 * (d_oo - d_binders)

    # any direction perpendicular to the bridge axis: phi = 0 reference
    reference = np.array([0., 0., 1.])
    if abs(np.dot(reference, bridge_axis)) > 0.9:
        reference = np.array([1., 0., 0.])
    perp_reference = unit(reference - np.dot(reference, bridge_axis) * bridge_axis)

    framework_positions = framework.get_positions()
    cell, pbc = framework.cell, framework.pbc

    trials, tags, scores = [], [], []

    for flip in ((False, True) if try_flip else (False,)):
        binder_axis = -bridge_axis if flip else bridge_axis
        bond_o1, bond_o2 = ((bond_lengths[1], bond_lengths[0]) if flip
                            else (bond_lengths[0], bond_lengths[1]))

        # bond_o1^2 = along^2 + depth^2 and
        # bond_o2^2 = (2*half_excess - along)^2 + depth^2,
        # solved for the along-axis offset from O1 to the binder above it
        along = 0.0 if abs(half_excess) < 1e-8 else \
            (bond_o1 ** 2 - bond_o2 ** 2 + 4.0 * half_excess ** 2) / (4.0 * half_excess)
        depth_squared = bond_o1 ** 2 - along ** 2
        if depth_squared <= 0.0:
            raise ValueError("molecule span %.2f A incompatible with d(O-O) "
                             "%.2f A at bond lengths %.2f/%.2f A"
                             % (d_binders, d_oo, bond_o1, bond_o2))
        depth = np.sqrt(depth_squared)
        midpoint_offset = along + 0.5 * d_binders

        aligned = molecule.copy()
        aligned_positions = aligned.get_positions()
        aligned.rotate(aligned_positions[binder_2] - aligned_positions[binder_1],
                       binder_axis,
                       center=0.5 * (aligned_positions[binder_1]
                                     + aligned_positions[binder_2]))

        for i in range(n_phi):
            phi = i * (360.0 / n_phi)
            target_midpoint = (o1_position + midpoint_offset * bridge_axis
                               + depth * rotate_vector(perp_reference, bridge_axis, phi))

            for j in range(n_psi):
                psi = j * (360.0 / n_psi)
                trial = aligned.copy()
                trial_positions = trial.get_positions()
                binder_midpoint = 0.5 * (trial_positions[binder_1]
                                         + trial_positions[binder_2])
                trial.rotate(psi, binder_axis, center=binder_midpoint)
                trial.translate(target_midpoint - binder_midpoint)

                trials.append(trial)
                tags.append({"flip": int(flip), "phi": phi, "psi": psi,
                             "depth": depth})
                scores.append(_tail_clearance(trial, framework_positions,
                                              cell, pbc, tail))

    trials, tags, scores = _keep_best(trials, tags, scores, best_only)
    return _combine(framework, trials), tags, scores
