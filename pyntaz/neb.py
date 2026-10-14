import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms
from ase.geometry import find_mic
from ase.mep import NEB
from ase.optimize import BFGS, FIRE

IMAGES_BEFORE = 3       # images between the initial state and the TS
IMAGES_AFTER = 3        # images between the TS and the final state
INTERPOLATION = "idpp"  # "idpp" or "linear"
FREE_RADIUS = 4.0       # A, framework atoms this close to the adsorbate move
SAME_POSITION = 1e-3    # A, atoms that moved more than this between states are free
SPRING = 0.1            # eV/A^2, NEB spring constant
METHOD = "improvedtangent"
FMAX_ROUGH = 0.5        # eV/A, band without climbing
STEPS_ROUGH = 200
FMAX = 0.05             # eV/A, band with the climbing image
STEPS = 500
TS_RMSD = 0.3           # A, adsorbate RMSD below which the top image is the Sella TS
TS_BOND = 0.25
MIN_DEPTH = 0.05        # eV, an interior image this far below both neighbours is a minimum


def displacements(a, b, indices=None):
    """Minimum-image distance each atom moved between ``a`` and ``b``."""
    delta = b.get_positions() - a.get_positions()
    if indices is not None:
        delta = delta[indices]
    _, lengths = find_mic(delta, a.cell, a.pbc)
    return lengths


def free_region(states, n_framework, free_radius=FREE_RADIUS, same=SAME_POSITION):
    """Sorted indices that move in the NEB, the same set for every image.

    Every adsorbate atom (index >= ``n_framework``), every framework atom
    within ``free_radius`` of the adsorbate in any of ``states``, and every
    atom whose position differs by more than ``same`` between any two of
    them. Each state was relaxed with its own free region, so the union is
    what lets all of them be reached; an atom that moved in one state but
    is not near the adsorbate in another must still be free, or the band
    would pin it in two places at once.
    """
    free = set(range(n_framework, len(states[0])))
    adsorbate = np.arange(n_framework, len(states[0]))
    for atoms in states:
        for f in range(n_framework):
            if f not in free and atoms.get_distances(f, adsorbate, mic=True).min() < free_radius:
                free.add(f)
    for k, a in enumerate(states):
        for b in states[k + 1:]:
            free.update(int(i) for i in np.flatnonzero(displacements(a, b) > same))
    return sorted(free)


def half_band(start, end, n_between, method=INTERPOLATION):
    """[start, n_between interpolated images, end]; the images are new
    copies, ``start`` and ``end`` are the objects passed in."""
    images = [start] + [start.copy() for _ in range(n_between)] + [end]
    if n_between:
        neb = NEB(images)
        # every atom interpolated; the frozen ones sit at the same place in
        # start and end, so they stay put anyway
        neb.interpolate(method=method, mic=True, apply_constraint=False)
    return images


def build_band(initial, ts, final, n_framework, n_before=IMAGES_BEFORE,
               n_after=IMAGES_AFTER, method=INTERPOLATION, free_radius=FREE_RADIUS):
    """The starting band and its free atoms.

    Returns ``(images, free)``: ``images`` is initial, ``n_before`` images,
    the TS, ``n_after`` images, final -- ``n_before + n_after + 3`` in all,
    the TS at index ``n_before + 1`` -- each with the same FixAtoms on
    everything outside ``free``. The inputs are not changed.
    """
    if n_before < 0 or n_after < 0:
        raise ValueError("image counts must be >= 0")
    if method not in ("idpp", "linear"):
        raise ValueError('method must be "idpp" or "linear"')
    states = [a.copy() for a in (initial, ts, final)]
    for atoms in states:
        atoms.calc = None
        atoms.pbc = True
        atoms.set_constraint()
    if len({len(a) for a in states}) != 1:
        raise ValueError("initial, TS and final have different atom counts")
    if len({a.get_chemical_formula(mode="all") for a in states}) != 1:
        raise ValueError("initial, TS and final have a different atom order")

    free = free_region(states, n_framework, free_radius)
    frozen = [i for i in range(len(states[0])) if i not in set(free)]
    for atoms in states:
        atoms.set_constraint(FixAtoms(indices=frozen))
    start, saddle, end = states
    left = half_band(start, saddle, n_before, method)
    right = half_band(saddle, end, n_after, method)
    images = left + right[1:]
    for atoms in images:
        atoms.set_constraint(FixAtoms(indices=frozen))
    return images, free


# --------------------------------------------------------------------------
# the NEB
# --------------------------------------------------------------------------

class ImageCalculator(Calculator):
    """One per NEB image, all sharing one loaded model.

    ASE's ``allow_shared_calculator`` would work for the forces, but a
    shared calculator only remembers the image it computed last, so the
    trajectory would store an energy for one image per step. This keeps
    each image's own results and calls the model for the numbers.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, model):
        super().__init__()
        self.model = model

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        energy = self.model.get_property("energy", self.atoms)
        forces = self.model.get_property("forces", self.atoms)
        self.results = {"energy": float(energy), "forces": np.array(forces, copy=True)}


def relax_endpoint(atoms, calc, fmax=FMAX, steps=STEPS_ROUGH, logfile=None):
    """BFGS on an end point in place, with whatever FixAtoms it carries.

    Step 7 relaxed each end point with its own free region; the band frees
    the union of all three, and the model may have changed since. Without
    this, the images next to an end point relax framework atoms the end
    point never could and drop below it, which reads as an intermediate.
    ``atoms`` keeps ``calc``. Returns energies before and after, steps,
    convergence and the largest displacement.
    """
    atoms.calc = calc
    before = float(atoms.get_potential_energy())
    start = atoms.copy()
    with BFGS(atoms, logfile=logfile) as optimizer:
        optimizer.run(fmax=fmax, steps=steps)
        n_steps = optimizer.nsteps
    largest = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    return {"energy_before": before, "energy": float(atoms.get_potential_energy()),
            "n_steps": n_steps, "converged": largest < fmax, "fmax": largest,
            "max_displacement": float(displacements(start, atoms).max())}


def band_fmax(neb):
    """Largest NEB force (spring + perpendicular true force, climbing
    included) on any atom of any interior image."""
    return float(np.linalg.norm(neb.get_forces(), axis=1).max())


def run_neb(images, spring=SPRING, method=METHOD, fmax_rough=FMAX_ROUGH,
            steps_rough=STEPS_ROUGH, climb=True, fmax=FMAX, steps=STEPS,
            trajectory=None, logfile=None):
    """NEB on ``images`` in place: FIRE without climbing to ``fmax_rough``,
    then (``climb``) with the climbing image to ``fmax``.

    Every image must already have its calculator: the end points a fixed
    one (they never move), the interior an :class:`ImageCalculator`.
    ``trajectory`` collects every image of every step, both stages.

    Returns a dict: ``converged`` (of the last stage run), ``fmax``,
    ``n_steps`` per stage and ``climbing_image`` (index of the highest
    interior image at the end, which climbs when ``climb``).
    """
    neb = NEB(images, k=spring, climb=False, method=method)
    with FIRE(neb, trajectory=trajectory, logfile=logfile) as optimizer:
        optimizer.run(fmax=fmax_rough, steps=steps_rough)
        n_rough = optimizer.nsteps
    rough = band_fmax(neb)
    result = {"rough": {"converged": rough < fmax_rough, "fmax": rough, "n_steps": n_rough}}

    if climb:
        neb.climb = True
        with FIRE(neb, trajectory=trajectory, logfile=logfile,
                  append_trajectory=True) as optimizer:
            optimizer.run(fmax=fmax, steps=steps)
            n_climb = optimizer.nsteps
        final = band_fmax(neb)
        result["climb"] = {"converged": final < fmax, "fmax": final, "n_steps": n_climb}
        last = result["climb"]
    else:
        last = result["rough"]
    energies = [float(a.get_potential_energy()) for a in images]
    result.update(converged=last["converged"], fmax=last["fmax"], energies=energies,
                  climbing_image=int(np.argmax(energies[1:-1]) + 1))
    return result


# --------------------------------------------------------------------------
# what the band says
# --------------------------------------------------------------------------

def path_length(images):
    """Cumulative minimum-image distance along the band, A, one per image."""
    steps = [float(np.sqrt((displacements(a, b) ** 2).sum()))
             for a, b in zip(images[:-1], images[1:])]
    return [0.0] + list(np.cumsum(steps))


def interior_minima(energies, min_depth=MIN_DEPTH):
    """Interior image indices at least ``min_depth`` below both neighbours."""
    return [i for i in range(1, len(energies) - 1)
            if energies[i] < energies[i - 1] - min_depth
            and energies[i] < energies[i + 1] - min_depth]


def analyze_band(images, energies, ts, n_framework, ts_energy=None, min_depth=MIN_DEPTH, bonds=None):
    """Numbers for the verdict, from the converged band.

    ``ts`` is the Sella TS (same atom order), ``ts_energy`` its energy with
    the model the NEB used. The highest interior image is compared with
    it by the RMSD of the adsorbate atoms (minimum image, no superposition:
    the framework holds both in place).
    """
    energies = list(energies)
    top = int(np.argmax(energies[1:-1]) + 1)
    adsorbate = np.arange(n_framework, len(ts))
    moved = displacements(ts, images[top], adsorbate)
    out = {"top_image": top,
           "Ea_fwd": energies[top] - energies[0],
           "Ea_rev": energies[top] - energies[-1],
           "dE_rxn": energies[-1] - energies[0],
           "top_vs_ts_rmsd": float(np.sqrt((moved ** 2).mean())),
           "top_vs_ts_max": float(moved.max()),
           "minima": interior_minima(energies, min_depth),
           "path_length": path_length(images)}
    if ts_energy is not None:
        out["top_minus_ts_energy"] = energies[top] - ts_energy

    if bonds:
        ts_lengths = [float(ts.get_distance(*b[:2], mic=True)) for b in bonds]
        top_lengths = [float(images[top].get_distance(*b[:2], mic=True)) for b in bonds]
        out["bonds_ts"] = ts_lengths
        out["bonds_top"] = top_lengths
        out["top_vs_ts_bond_max"] = max(abs(a - b) for a, b in zip(ts_lengths, top_lengths))
    return out


def verdict(search, analysis, ts_rmsd=TS_RMSD, ts_bond=TS_BOND):
    """One word for the band.

    not_converged   the last NEB stage did not reach its fmax
    intermediate    the band dips into a minimum between the end points:
                    the reaction runs in two steps (or the band found a
                    detour), so one TS does not connect IS and FS
    different_ts    a single barrier, but its top is not the Sella TS
    connected       a single barrier whose top is the Sella TS: that TS
                    connects this initial and final state
    """
    if not search["converged"]:
        return "not_converged"
    if analysis["minima"]:
        return "intermediate"
    if (analysis["top_vs_ts_rmsd"] > ts_rmsd
            or analysis.get("top_vs_ts_bond_max", 0.0) > ts_bond):
        return "different_ts"
    return "connected"


def plot_band(path, energies, analysis, ts_index, title=""):
    """Energy along the band, relative to the initial state; the highest
    image and the image that started at the Sella TS are marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = analysis["path_length"]
    y = [e - energies[0] for e in energies]
    top = analysis["top_image"]
    fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=150)
    ax.plot(x, y, "-o", color="#1b9e77", ms=5, lw=1.4)
    ax.plot(x[top], y[top], "o", ms=9, mfc="none", mec="#d95f02", mew=1.6,
            label="highest image  %.2f eV" % y[top])
    ax.axvline(x[ts_index], color="#7570b3", lw=.8, ls="--",
               label="image %d (started at the Sella TS)" % ts_index)
    for i in analysis["minima"]:
        ax.plot(x[i], y[i], "v", color="#e7298a", ms=7)
    ax.axhline(0, color="#6b6b6b", lw=.6)
    ax.set_xlabel("path length (A)")
    ax.set_ylabel("E - E$_{IS}$ (eV)")
    ax.set_title(title, loc="left", fontsize=9)
    ax.grid(color="#e9e9e9")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)