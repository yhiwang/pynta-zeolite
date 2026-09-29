"""Step 7 of the workflow: harmonic pre-relax of one TS guess with MACE.

A TS guess from step 5 has every changing bond at a rough length and sits
on a framework that was never relaxed around the Al. Before a saddle
search, this relaxes it in alternating stages, each moving one group of
atoms with everything else fixed:

    framework    framework atoms within ``framework_radius`` of the
                 adsorbate (springs on, so the seated oxygens hold their
                 TS distances)
    spectators   adsorbate atoms in no forming or breaking bond (no springs)
    reacting     adsorbate atoms in a forming or breaking bond, pulled by
                 harmonic springs toward (r_i + r_j) x ``mult[change]``,
                 r = ASE covalent radius

One cycle is framework, spectators, reacting. Cycles repeat until the MACE
energy changes by less than ``e_tol`` over a cycle, or ``max_cycles`` is
reached; a last spectator stage then relaxes the spectators around the
final reacting geometry. The result is a spring-biased starting point for a
saddle search, not a transition state.

Only ASE, MACE and numpy are needed, so this runs on a compute node without
maze or RMG. :func:`guess_bond_indices` takes a :class:`pyntaz.ts_graph.TSGraph`
but never imports RMG itself; call it where the graph was built.
"""

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.mixing import SumCalculator
from ase.constraints import FixAtoms
from ase.data import covalent_radii
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS

MULT = {"form": 1.35, "break": 1.35}   # TS length = (r_i + r_j) x this
K_RESTRAINT = 30.0                     # eV/A^2
FRAMEWORK_RADIUS = 4.0                 # A
STEPS = {"framework": 100, "spectators": 20, "reacting": 100}
MAX_CYCLES = 3
E_TOL = 0.01                           # eV
FMAX = 0.05                            # eV/A, per stage

REACTING_CHANGES = ("form", "break")   # "order" bonds (C=C -> C-C) get no spring


# --------------------------------------------------------------------------
# graph atoms -> indices in a guess structure
# --------------------------------------------------------------------------

def guess_atom_map(ts, oxygens, n_framework):
    """{graph index: structure index} for a guess built by
    :class:`pyntaz.ts_graph.PairSweep` on the ordered ``oxygens``.

    Mirrors how ``PairSweep`` lays a guess out: the framework first
    (``n_framework`` atoms), then every non-X graph atom in graph order
    (``PairSweep.keep``); the two X atoms are the seated oxygens, the first
    X on ``oxygens[0]``. Change it together with ``PairSweep`` or not at all.
    """
    sites = ts.sites()
    if len(sites) != 2:
        raise NotImplementedError("two-site reactions only, this one has %d X" % len(sites))
    keep = [i for i, element in enumerate(ts.elements) if element != "X"]
    where = {g: n_framework + k for k, g in enumerate(keep)}
    where[sites[0]], where[sites[1]] = int(oxygens[0]), int(oxygens[1])
    return where


def guess_bond_indices(ts, oxygens, n_framework):
    """[{"change", "tags", "indices"}] for every forming or breaking bond of
    ``ts``, in structure indices of a guess seated on ``oxygens``. Ready to
    json-dump; :func:`harmonic_relax` takes the (i, j, change) triples."""
    where = guess_atom_map(ts, oxygens, n_framework)
    return [{"change": change, "tags": [ts.tag(a), ts.tag(b)],
             "indices": [where[a], where[b]]}
            for change, a, b, _, _, _ in ts.changes() if change in REACTING_CHANGES]


def check_atom_order(atoms, ts, oxygens, n_framework):
    """Raise ValueError unless ``atoms`` is laid out the way
    :func:`guess_atom_map` assumes: the right atom count, and every mapped
    index holding the element the graph expects (O for the X sites)."""
    where = guess_atom_map(ts, oxygens, n_framework)
    n_real = sum(1 for element in ts.elements if element != "X")
    if len(atoms) != n_framework + n_real:
        raise ValueError("%d atoms, expected %d framework + %d adsorbate"
                         % (len(atoms), n_framework, n_real))
    for g, i in where.items():
        expected = "O" if ts.elements[g] == "X" else ts.elements[g]
        if atoms[i].symbol != expected:
            raise ValueError("atom %d is %s, graph atom %s expects %s"
                             % (i, atoms[i].symbol, ts.tag(g), expected))


# --------------------------------------------------------------------------
# the springs
# --------------------------------------------------------------------------

class BondRestraints(Calculator):
    """Harmonic springs between atom pairs: E = 1/2 k (r - r0)^2, r by the
    minimum-image convention."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, pairs, k, **kwargs):
        super().__init__(**kwargs)
        self.pairs = pairs          # [(i, j, r0)], r0 in A
        self.k = k

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        energy = 0.0
        forces = np.zeros((len(self.atoms), 3))
        for i, j, r0 in self.pairs:
            d = self.atoms.get_distance(i, j, mic=True, vector=True)
            r = np.linalg.norm(d)
            energy += 0.5 * self.k * (r - r0) ** 2
            f = self.k * (r - r0) * d / r
            forces[i] += f
            forces[j] -= f
        self.results["energy"] = energy
        self.results["forces"] = forces


def spring_targets(atoms, bonds, mult=None):
    """[(i, j, r0)] with r0 = (covalent r_i + r_j) x ``mult[change]`` for
    every (i, j, change) in ``bonds``."""
    mult = MULT if mult is None else mult
    radius = covalent_radii[atoms.get_atomic_numbers()]
    return [(i, j, float((radius[i] + radius[j]) * mult[change])) for i, j, change in bonds]


# --------------------------------------------------------------------------
# the staged relax
# --------------------------------------------------------------------------

def harmonic_relax(atoms, bonds, n_framework, model_path, mult=None, k=K_RESTRAINT,
                   framework_radius=FRAMEWORK_RADIUS, steps=None, max_cycles=MAX_CYCLES,
                   e_tol=E_TOL, fmax=FMAX, device="cpu", logfile=None,
                   trajectory=None, log=None):
    """Relax a TS guess in place in alternating stages (module docstring).

    ``bonds`` are the forming / breaking bonds as (i, j, change) in
    ``atoms`` indices; the first ``n_framework`` atoms are the framework.
    ``logfile`` / ``trajectory`` (paths or open objects) collect every
    stage's BFGS in order; a path is opened once here and overwritten. ``log`` is called with one line per cycle
    when given (``log=print`` for a job's stdout).

    ``model_path`` must be a file: a bare model name makes MACE try to
    download it, which fails on a compute node without internet.

    Returns a dict: ``energy`` and ``forces`` (MACE only, no springs, no
    constraints), ``cycles`` [(framework, spectator, reacting steps,
    energy)], ``final_spectator_steps``, ``settled`` (the energy criterion
    was met before ``max_cycles``), ``n_framework_free``, ``fmax_reacting`` /
    ``fmax_framework`` (largest spring-free MACE force on those atoms) and
    ``lengths`` [(i, j, change, r, r0)]. ``atoms`` is left without a
    calculator or constraint; attach the result yourself before writing.
    """
    from mace.calculators import mace_mp   # heavy import, only when relaxing

    log = log or (lambda line: None)
    steps = dict(STEPS, **(steps or {}))
    atoms.pbc = True

    adsorbate = np.arange(n_framework, len(atoms))
    reacting = sorted({i for i, j, _ in bonds} | {j for i, j, _ in bonds})
    targets = spring_targets(atoms, bonds, mult)
    local = [f for f in range(n_framework)
             if atoms.get_distances(f, adsorbate, mic=True).min() < framework_radius]
    movers = {"framework": local,
              "spectators": [i for i in adsorbate if i not in reacting],
              "reacting": [i for i in reacting if i >= n_framework]}

    mace = mace_mp(model=model_path, default_dtype="float64", device=device)
    springs = BondRestraints(targets, k)
    # one handle for all stages: a BFGS given a path would reopen it per stage
    # and a trajectory path would be overwritten each time
    own_log = isinstance(logfile, str)
    own_traj = isinstance(trajectory, str)
    logfile = open(logfile, "w") if own_log else logfile
    trajectory = Trajectory(trajectory, "w") if own_traj else trajectory

    def stage(which):
        free = set(movers[which])
        atoms.set_constraint(FixAtoms(indices=[a for a in range(len(atoms)) if a not in free]))
        atoms.calc = mace if which == "spectators" else SumCalculator([mace, springs])
        optimizer = BFGS(atoms, logfile=logfile, trajectory=trajectory)
        optimizer.run(fmax=fmax, steps=steps[which])
        atoms.calc = mace
        return optimizer.get_number_of_steps()

    atoms.calc = mace
    energy = atoms.get_potential_energy()
    log("  %d framework atoms within %.1f A of the adsorbate relax" % (len(local), framework_radius))
    log("  start                              E = %.4f eV" % energy)
    cycles, settled = [], False
    for cycle in range(1, max_cycles + 1):
        counts = [stage(which) for which in ("framework", "spectators", "reacting")]
        previous, energy = energy, atoms.get_potential_energy()
        cycles.append(tuple(counts) + (energy,))
        log("  cycle %d  %3d + %3d + %3d steps   E = %.4f eV   change %+.4f"
            % ((cycle,) + tuple(counts) + (energy, energy - previous)))
        if abs(energy - previous) < e_tol:
            settled = True
            break
    final_steps = stage("spectators")
    energy = atoms.get_potential_energy()
    log("  final spectators  %3d steps         E = %.4f eV" % (final_steps, energy))

    if own_log:
        logfile.close()
    if own_traj:
        trajectory.close()

    atoms.set_constraint()                 # FixAtoms would zero the forces reported
    forces = atoms.get_forces()
    atoms.calc = None
    return {"energy": energy, "forces": forces, "cycles": cycles,
            "final_spectator_steps": final_steps, "settled": settled,
            "n_framework_free": len(local),
            "fmax_reacting": float(np.linalg.norm(forces[reacting], axis=1).max()),
            "fmax_framework": float(np.linalg.norm(forces[local], axis=1).max()) if local else 0.0,
            "lengths": [(i, j, change, float(atoms.get_distance(i, j, mic=True)), r0)
                        for (i, j, change), (_, _, r0) in zip(bonds, targets)]}