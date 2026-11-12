"""Step 6 of the workflow: construct a transition-state guess for one
endpoint pair written by :mod:`pyntaz.ts_pairs`.

How it works
------------
The reaction side without gas species is already one assembled structure,
so its atom ordering is *canonical* and everything else is built to match
it. Correspondence between the two sides runs through the ``*N`` labels in
``reaction.yaml``: both sides number the same atoms the same way, so
"side index -> ase index" on each side gives the permutation between them.
RMG re-canonicalises atom order when a side is split into species, so the
link from a side to a stored config is recovered by graph isomorphism
(:func:`map_side_atoms`) rather than assumed; where several isomorphisms
exist (equivalent hydrogens, the two ethylene carbons) the one that agrees
best with the geometry is taken.

The guess itself follows the reaction (:class:`InsertionPlan`): the
breaking bond is stretched along its own axis, and the gas fragment is
aligned so its forming-bond vector lies on the *free site -> migrating
group* axis, then rolled about that axis for clearance
(:class:`TSGuessBuilder`). The stretch escalates until nothing collides.

Per pair, ``endpoint_<role>.xyz`` (the relaxed minimum, reordered) and
``ts_guess.xyz`` (the construction) are written, both in the canonical
ordering.
"""

import json
import os
import re
import traceback

import numpy as np
from ase.io import read, write
from molecule.molecule import Molecule

from . import config, runtree
from .geometry import (framework_indices, find_clashes, kabsch_transform,
                       cloud_rmsd, mic_unit_vector, min_clearance, nearest_atom,
                       unit)
from .ts_pairs import INITIAL_XYZ, FINAL_XYZ, REACTION_INFO

_REACTION_DIR = re.compile(r"^\d+_rxn$")
_PAIR_DIR = re.compile(r"^pair_\d+$")


# ==========================================================================
# RMG molecule helpers
# ==========================================================================

def index_by_id(mol):
    """{id(atom): index}. ``list.index()`` is unsafe on RMG atoms: Atom
    defines comparison operators, so ``==`` can be true for two distinct
    atoms of the same element and ``.index()`` returns the wrong one."""
    return {id(atom): i for i, atom in enumerate(mol.atoms)}


def bond_string(mol, index):
    """"1S,4D" style summary of the bonds of one atom."""
    where = index_by_id(mol)
    bonds = sorted(mol.atoms[index].bonds.items(),
                   key=lambda item: where[id(item[0])])
    return ",".join("%d%s" % (where[id(neighbor)], bond.get_order_str())
                    for neighbor, bond in bonds) or "-"


def print_graph(title, mol, ase_index_of=None):
    print("\n%s   %d atoms" % (title, len(mol.atoms)))
    for i, atom in enumerate(mol.atoms):
        ase_column = "" if ase_index_of is None else \
            "xyz %-5s" % ase_index_of.get(i, "--")
        print("  %-3d %-3s %-2s  %s bonds %s"
              % (i, atom.label or "", atom.symbol, ase_column, bond_string(mol, i)))


def bond_pairs(mol):
    """{(i, j)} with i < j for every bond."""
    where = index_by_id(mol)
    return {(min(i, where[id(neighbor)]), max(i, where[id(neighbor)]))
            for i, atom in enumerate(mol.atoms) for neighbor in atom.bonds}


def molecule_components(mol):
    """[[side index, ...]] per connected component, yaml order kept."""
    where = index_by_id(mol)
    seen, components = set(), []
    for i in range(len(mol.atoms)):
        if i in seen:
            continue
        stack, component = [i], []
        seen.add(i)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in mol.atoms[current].bonds:
                neighbor_index = where[id(neighbor)]
                if neighbor_index not in seen:
                    seen.add(neighbor_index)
                    stack.append(neighbor_index)
        components.append(sorted(component))
    return components


def submolecule(mol, indices):
    """Standalone Molecule over ``indices``; its atom j is side atom
    ``indices[j]``. Deep copy first: the labels have to go before matching,
    and clearing them on the original would destroy what the side table is
    keyed on."""
    copy = mol.copy(deep=True)
    sub = Molecule(atoms=[copy.atoms[i] for i in indices])
    sub.clear_labeled_atoms()
    sub.multiplicity = sub.get_radical_count() + 1
    sub.update_connectivity_values()
    return sub


def all_isomorphisms(sub, stored):
    """[{sub index: stored index}] over every isomorphism, [] when none.

    ``find_isomorphism`` sorts both molecules in place, so the index tables
    are taken first and the returned atom objects looked up in them. The
    direction of the returned dicts is checked rather than assumed."""
    where_sub = index_by_id(sub)
    where_stored = index_by_id(stored)
    mappings = []
    for mapping in sub.find_isomorphism(stored):
        pairs = list(mapping.items())
        if not all(id(atom) in where_sub for atom, _ in pairs):
            pairs = [(stored_atom, sub_atom) for sub_atom, stored_atom in pairs]
        mappings.append({where_sub[id(sub_atom)]: where_stored[id(stored_atom)]
                         for sub_atom, stored_atom in pairs})
    return mappings


def check_side_alignment(reactant, product):
    """``reaction.yaml`` numbers both sides for the same atoms in the same
    order. Nothing in RMG enforces that -- it is a convention of the file --
    and the whole side table rests on it, so it is checked."""
    if len(reactant.atoms) != len(product.atoms):
        raise ValueError("sides have %d and %d atoms"
                         % (len(reactant.atoms), len(product.atoms)))
    for i, (r_atom, p_atom) in enumerate(zip(reactant.atoms, product.atoms)):
        if r_atom.symbol != p_atom.symbol:
            raise ValueError("index %d is %s on the reactant side and %s on "
                             "the product side" % (i, r_atom.symbol, p_atom.symbol))
        if r_atom.label and p_atom.label and r_atom.label != p_atom.label:
            raise ValueError("index %d is %s and %s" % (i, r_atom.label, p_atom.label))


# ==========================================================================
# stored configurations
# ==========================================================================

class StoredConfig:
    """One relaxed structure from the unique tree together with the
    molecule-index bookkeeping from its species ``info.json``.

    ``role`` is "initial", "final" or "gas". Surface configs count the
    framework atoms first (``n_framework``), gas configs start at 0.
    ``mol_to_ase`` maps molecule atom index -> index in ``atoms``.
    """

    def __init__(self, path, species, role, unique_tree):
        self.path = path
        self.species = species
        self.role = role
        self.atoms = read(path)
        info = runtree.load_species_info(unique_tree, species)
        self.adjlist = info["adjlist"]
        on_surface = role != "gas"
        self.n_framework = info["nslab"] if on_surface else 0
        self.mol_to_ase = {mol_index: self.n_framework + int(ase_index)
                           for ase_index, mol_index
                           in info["atom_to_molecule_atom_map"].items()}
        if on_surface:
            self.framework = framework_indices(self.atoms)
            if len(self.framework) != self.n_framework:
                raise ValueError("%s: %d framework atoms, nslab=%d"
                                 % (path, len(self.framework), self.n_framework))

    @classmethod
    def from_path(cls, path):
        """StoredConfig from just the path of a file inside a pair folder,
        ``<run_dir>/ts_guesses/<i>_rxn/pair_NNNN/<name>.xyz``.

        ``initial.xyz`` / ``final.xyz`` take their species from the pair
        manifest; any other file there (``ts_guess.xyz``, ``endpoint_*.xyz``)
        is in the canonical ordering, i.e. the species of the side without
        gas. The unique tree is found from the run directory in the path.
        """
        path = os.path.abspath(path)
        pair_dir = os.path.dirname(path)
        reaction_dir = os.path.dirname(pair_dir)
        run_dir = os.path.dirname(os.path.dirname(reaction_dir))
        with open(os.path.join(reaction_dir, REACTION_INFO)) as handle:
            reaction = json.load(handle)
        endpoints = reaction["pairs"][os.path.basename(pair_dir)]

        role = os.path.splitext(os.path.basename(path))[0]
        if role not in ("initial", "final"):
            role = "initial" if reaction["gas"]["final"] else "final"
        return cls(path, endpoints[role]["species"], role,
                   config.RunLayout(run_dir).unique)

    def molecule(self):
        """The RMG Molecule of the species (from the stored adjacency list)."""
        mol = Molecule().from_adjacency_list(self.adjlist)
        mol.multiplicity = mol.get_radical_count() + 1
        return mol

    def adsorbate_indices(self):
        """xyz indices of the adsorbate atoms (everything after the framework)."""
        return list(range(self.n_framework, len(self.atoms)))

    def nearest_framework_atom(self, xyz_index):
        """(framework index, distance) of the framework atom closest to an
        adsorbate atom; raises for gas-phase configs."""
        if not self.n_framework:
            raise ValueError("%s is a gas-phase config" % self)
        return nearest_atom(self.atoms, xyz_index, self.framework)

    def __repr__(self):
        return "StoredConfig(%s, %s)" % (self.species, self.role)


def load_gas_configs(names, unique_tree):
    """Gas-phase StoredConfigs for the given species names."""
    return [StoredConfig(os.path.join(unique_tree, name, "0",
                                      config.GAS_STEM + ".xyz"),
                         name, "gas", unique_tree)
            for name in names]


def load_pair(pair_dir, unique_tree=None):
    """Everything step 6 loads for one pair folder, as
    ``(reaction, initial, final, gas_initial, gas_final)``.

    ``reaction`` is the reaction's info.json dict; the other four are
    StoredConfigs. ``unique_tree`` defaults to the run directory's
    ``Adsorbates_relax_unique`` found from the path.
    """
    pair_dir = os.path.abspath(pair_dir)
    reaction_dir = os.path.dirname(pair_dir)
    if unique_tree is None:
        run_dir = os.path.dirname(os.path.dirname(reaction_dir))
        unique_tree = config.RunLayout(run_dir).unique
    with open(os.path.join(reaction_dir, REACTION_INFO)) as handle:
        reaction = json.load(handle)
    endpoints = reaction["pairs"][os.path.basename(pair_dir)]
    initial = StoredConfig(os.path.join(pair_dir, INITIAL_XYZ),
                           endpoints["initial"]["species"], "initial", unique_tree)
    final = StoredConfig(os.path.join(pair_dir, FINAL_XYZ),
                         endpoints["final"]["species"], "final", unique_tree)
    gas_initial = load_gas_configs(reaction["gas"]["initial"], unique_tree)
    gas_final = load_gas_configs(reaction["gas"]["final"], unique_tree)
    return reaction, initial, final, gas_initial, gas_final


# ==========================================================================
# side atoms -> stored configs
# ==========================================================================

def map_side_atoms(adjlist, configs, targets=None, verbose=False):
    """{side index: (StoredConfig, molecule index, ase index or None)} for
    one whole reaction side.

    Each connected component of the side (bare X sites skipped) is matched
    by isomorphism against the ``configs`` in order; the first config that
    matches takes the component. With ``targets`` ({side index: position})
    the isomorphism is chosen by geometry rather than as RMG hands it over:
    gas fragments are superposed first (Kabsch), surface ones compared
    where they stand -- both endpoints share the frozen framework, so a
    surface displacement is already meaningful.
    """
    side = Molecule().from_adjacency_list(adjlist)
    side_map = {}
    for component in molecule_components(side):
        if all(side.atoms[i].is_surface_site() for i in component):
            continue
        fragment = submolecule(side, component)
        for stored in configs:
            candidates = all_isomorphisms(fragment, stored.molecule())
            if not candidates:
                continue
            best, best_rmsd = candidates[0], None
            if targets is not None:
                mapped = [j for j in range(len(component)) if component[j] in targets]
                target_cloud = np.array([targets[component[j]] for j in mapped])
                for mapping in candidates:
                    cloud = np.array([stored.atoms.positions[stored.mol_to_ase[mapping[j]]]
                                      for j in mapped])
                    if stored.role == "gas":
                        rotation, translation = kabsch_transform(cloud, target_cloud)
                        cloud = cloud @ rotation.T + translation
                    rmsd = cloud_rmsd(cloud, target_cloud)
                    if best_rmsd is None or rmsd < best_rmsd:
                        best, best_rmsd = mapping, rmsd
                if verbose:
                    print("    %-16s %-8s %2d isomorphisms, best rmsd %.3f A"
                          % (stored.species, stored.role, len(candidates), best_rmsd))
            for j, mol_index in best.items():
                side_map[component[j]] = (stored, mol_index,
                                          stored.mol_to_ase.get(mol_index))
            break
        else:
            raise ValueError("no config matches side atoms %s" % component)
    return side_map


def _describe_cell(side_map, index):
    if index not in side_map:
        return "-"
    stored, mol_index, ase_index = side_map[index]
    where = "--" if ase_index is None else \
        ("gas %d" % ase_index if stored.role == "gas" else "ase %d" % ase_index)
    return "%s[%d] %s" % (stored.species, mol_index, where)


# ==========================================================================
# what has to happen: the insertion plan
# ==========================================================================

class InsertionPlan:
    """Which side is canonical, how side atoms map to ase atoms on both
    sides, and which atoms play which role in the bond rearrangement.

    Roles (all *side indices* of the assembled side):

    free_site      the X the gas binds to
    old_site       the X whose bond breaks
    migrating      the group atom that leaves ``old_site``
    gas_at_site    the gas atom that bonds to ``free_site``
    gas_at_group   the gas atom that bonds to ``migrating``
    """

    def __init__(self, reaction, initial, final, gas_initial, gas_final,
                 verbose=False):
        reactant = Molecule().from_adjacency_list(reaction["reactant"])
        product = Molecule().from_adjacency_list(reaction["product"])
        check_side_alignment(reactant, product)
        self.reactant, self.product = reactant, product

        if verbose:
            print("\n=== connectivity, mol index vs xyz index ===")
            for stored in [initial] + gas_initial + gas_final + [final]:
                print_graph("%s: %s" % (stored.role, stored.species),
                            stored.molecule(), stored.mol_to_ase)
            print("\n=== reaction sides, whole-side numbering ===")
            print_graph("reactant", reactant)
            print_graph("product", product)

        # the side with no gas is one assembled file already, so its
        # ordering is not negotiable; the other side is built to match it
        if gas_final:
            self.canonical_role, self.canonical_adjlist = "initial", reaction["reactant"]
            self.canonical, self.canonical_gas = initial, []
            self.assembled_role, self.assembled_adjlist = "final", reaction["product"]
            self.assembled, self.assembled_gas = final, gas_final
        else:
            self.canonical_role, self.canonical_adjlist = "final", reaction["product"]
            self.canonical, self.canonical_gas = final, []
            self.assembled_role, self.assembled_adjlist = "initial", reaction["reactant"]
            self.assembled, self.assembled_gas = initial, gas_initial

        if verbose:
            print("\n=== assembly plan ===")
            print("  canonical %-8s %-16s (no gas, defines the ordering)"
                  % (self.canonical_role, self.canonical.species))
            print("  assembled %-8s %-16s + %s"
                  % (self.assembled_role, self.assembled.species,
                     ", ".join(stored.species for stored in self.assembled_gas)))

        self.canonical_map = map_side_atoms(
            self.canonical_adjlist, [self.canonical] + self.canonical_gas,
            verbose=verbose)
        targets = {i: self.canonical.atoms.positions[ase_index]
                   for i, (_, _, ase_index) in self.canonical_map.items()
                   if ase_index is not None}
        self.assembled_map = map_side_atoms(
            self.assembled_adjlist, [self.assembled] + self.assembled_gas,
            targets=targets, verbose=verbose)

        if verbose:
            self.print_side_table()

        self.canonical_mol = Molecule().from_adjacency_list(self.canonical_adjlist)
        self.assembled_mol = Molecule().from_adjacency_list(self.assembled_adjlist)
        self._find_roles(verbose)

    # -- helpers on the assembled side --------------------------------------

    def label(self, index):
        return self.assembled_mol.atoms[index].label or str(index)

    def is_site(self, index):
        return self.assembled_mol.atoms[index].is_surface_site()

    def in_gas(self, index):
        return (index in self.assembled_map
                and self.assembled_map[index][0].role == "gas")

    def canonical_ase(self, index):
        """ase index in the canonical structure of side atom ``index``."""
        return self.canonical_map[index][2]

    def print_side_table(self):
        """Print side index -> xyz index for both sides, and the ase
        permutation assembled -> canonical."""
        reactant, product = self.reactant, self.product
        if self.assembled_role == "initial":
            initial_map, final_map = self.assembled_map, self.canonical_map
        else:
            initial_map, final_map = self.canonical_map, self.assembled_map
        print("\n=== side index -> ase index ===")
        print("  %-3s %-3s %-2s  %-30s %-30s %-14s %s"
              % ("i", "*N", "el", "initial", "final", "r bonds", "p bonds"))
        for i in range(len(reactant.atoms)):
            print("  %-3d %-3s %-2s  %-30s %-30s %-14s %s"
                  % (i, reactant.atoms[i].label or "", reactant.atoms[i].symbol,
                     _describe_cell(initial_map, i), _describe_cell(final_map, i),
                     bond_string(reactant, i), bond_string(product, i)))

        print("\n=== ase permutation, %s -> %s ==="
              % (self.assembled_role, self.canonical_role))
        for i in range(len(reactant.atoms)):
            if i not in self.assembled_map or i not in self.canonical_map:
                continue
            assembled_stored, _, assembled_ase = self.assembled_map[i]
            canonical_stored, _, canonical_ase = self.canonical_map[i]
            if assembled_ase is None or canonical_ase is None:
                continue
            print("  %-3s %-14s %-9s %-4d ->  %-14s %d"
                  % (reactant.atoms[i].label or "", assembled_stored.species,
                     assembled_stored.role, assembled_ase,
                     canonical_stored.species, canonical_ase))

    # -- bond changes and roles ----------------------------------------------

    def _find_roles(self, verbose):
        # relative to the side being assembled: bonds it does not have but
        # the other side does are the ones that form on the way across
        self.forming = sorted(bond_pairs(self.canonical_mol) - bond_pairs(self.assembled_mol))
        self.breaking = sorted(bond_pairs(self.assembled_mol) - bond_pairs(self.canonical_mol))

        if verbose:
            print("\n=== bond changes on the %s side ===" % self.assembled_role)
            for i, j in self.breaking:
                print("  breaks  %-3s %-2s - %-3s %-2s"
                      % (self.label(i), self.assembled_mol.atoms[i].symbol,
                         self.label(j), self.assembled_mol.atoms[j].symbol))
            for i, j in self.forming:
                print("  forms   %-3s %-2s - %-3s %-2s"
                      % (self.label(i), self.assembled_mol.atoms[i].symbol,
                         self.label(j), self.assembled_mol.atoms[j].symbol))

        # the gas is inserted across two forming bonds: one to a framework
        # site, one to the group migrating off its old site
        site_bonds = [(i, j) for i, j in self.forming
                      if (self.is_site(i) and self.in_gas(j))
                      or (self.is_site(j) and self.in_gas(i))]
        group_bonds = [(i, j) for i, j in self.forming
                       if not self.is_site(i) and not self.is_site(j)
                       and self.in_gas(i) != self.in_gas(j)]
        if len(site_bonds) != 1 or len(group_bonds) != 1:
            raise ValueError("expected one site bond and one group bond, got "
                             "%d and %d" % (len(site_bonds), len(group_bonds)))

        (site_a, site_b), = site_bonds
        (group_a, group_b), = group_bonds
        self.free_site = site_a if self.is_site(site_a) else site_b
        self.gas_at_site = site_b if self.is_site(site_a) else site_a
        self.gas_at_group = group_a if self.in_gas(group_a) else group_b
        self.migrating = group_b if self.in_gas(group_a) else group_a

        breaking_sites = [i for pair in self.breaking for i in pair if self.is_site(i)]
        if len(breaking_sites) != 1:
            raise ValueError("expected one X in the breaking bonds, got %s"
                             % breaking_sites)
        self.old_site = breaking_sites[0]

        if verbose:
            print("\n=== insertion geometry ===")
            print("  free site      %-3s" % self.label(self.free_site))
            print("  old site       %-3s" % self.label(self.old_site))
            print("  migrating      %-3s %s"
                  % (self.label(self.migrating),
                     self.assembled_mol.atoms[self.migrating].symbol))
            print("  gas -> site    %-3s %s"
                  % (self.label(self.gas_at_site),
                     self.assembled_mol.atoms[self.gas_at_site].symbol))
            print("  gas -> group   %-3s %s"
                  % (self.label(self.gas_at_group),
                     self.assembled_mol.atoms[self.gas_at_group].symbol))


# ==========================================================================
# building the guess
# ==========================================================================

class TSGuessBuilder:
    """Assemble the TS guess in the canonical atom ordering from an
    :class:`InsertionPlan`.

    ``base`` is the canonical structure with the surface fragment of the
    assembled side written over it (same framework, adsorbate atoms moved to
    where the assembled minimum has them). :meth:`place` pulls the breaking
    bond by ``stretch`` and drops the gas fragment into the gap;
    :meth:`build` escalates the stretch until nothing clashes.
    """

    def __init__(self, plan, verbose=False,
                 framework_tolerance=config.FRAMEWORK_DRIFT_TOLERANCE,
                 roll_steps=config.TS_ROLL_STEPS, print_roll_table=False):
        self.plan = plan
        self.verbose = verbose
        self.roll_steps = roll_steps
        self.print_roll_table = print_roll_table

        canonical, assembled = plan.canonical, plan.assembled
        self.framework = framework_indices(canonical.atoms)
        drift = np.abs(assembled.atoms.positions[self.framework]
                       - canonical.atoms.positions[self.framework]).max()
        if drift > framework_tolerance:
            raise ValueError("endpoints do not share a framework, drift %.2e A"
                             % float(drift))

        # canonical structure with the assembled surface fragment in place
        self.base = canonical.atoms.copy()
        self.surface_slots = []
        for i, (stored, _, ase_index) in plan.assembled_map.items():
            if ase_index is None or stored.role == "gas":
                continue
            if i not in plan.canonical_map or plan.canonical_ase(i) is None:
                continue
            self.base.positions[plan.canonical_ase(i)] = stored.atoms.positions[ase_index]
            self.surface_slots.append(plan.canonical_ase(i))

        # canonical ase indices the gas atoms will occupy
        self.gas_slots = sorted(plan.canonical_ase(i)
                                for gas in plan.assembled_gas
                                for i in plan.assembled_map
                                if plan.assembled_map[i][0] is gas)

        # X atoms carry no coordinates, so each site is the framework O
        # nearest the adsorbate atom that sits on it
        self.migrating_ase = plan.canonical_ase(plan.migrating)
        self.old_oxygen, d_old = self.site_oxygen(self.base, self.migrating_ase)
        self.free_oxygen, d_free = self.site_oxygen(
            self.base, plan.canonical_ase(plan.gas_at_site), exclude=(self.old_oxygen,))
        if verbose:
            print("  %s sits over O %d at %.2f A"
                  % (plan.label(plan.migrating), self.old_oxygen, d_old))
            print("  free site is O %d at %.2f A from where %s ends up"
                  % (self.free_oxygen, d_free, plan.label(plan.gas_at_site)))

        self.break_axis, self.break_length = mic_unit_vector(
            self.base, self.old_oxygen, self.migrating_ase)

    def site_oxygen(self, atoms, ase_index, exclude=()):
        """(index, distance) of the framework O nearest to an adsorbate atom."""
        oxygens = [i for i in self.framework
                   if atoms[i].symbol == "O" and i not in exclude]
        distances = atoms.get_distances(ase_index, oxygens, mic=True)
        return oxygens[int(np.argmin(distances))], float(distances.min())

    def collisions(self, atoms):
        """(framework clashes, gas-against-surface-fragment clashes). The
        second matters as much as the first: the two organics are stacked in
        the same pore, and find_clashes only compares the index sets it is
        given."""
        adsorbate = [i for i in range(len(atoms)) if i not in self.framework]
        return (find_clashes(atoms, self.framework, adsorbate,
                             tolerance=config.CLASH_TOLERANCE),
                find_clashes(atoms, self.surface_slots, self.gas_slots,
                             tolerance=config.CLASH_TOLERANCE))

    def place(self, stretch):
        """Composite at one stretch: pull the breaking bond, then align and
        roll the gas into the gap that opens. Rebuilt from ``base`` each time
        -- the alignment axis moves with the migrating group, so a retry is a
        new placement rather than a nudge to the last one. Returns (atoms,
        details dict)."""
        plan = self.plan
        atoms = self.base.copy()
        atoms.positions[self.surface_slots] += self.break_axis * stretch

        for gas in plan.assembled_gas:
            members = [i for i in plan.assembled_map if plan.assembled_map[i][0] is gas]
            slot_of = {i: k for k, i in enumerate(members)}
            fragment = gas.atoms[[plan.assembled_map[i][2] for i in members]].copy()

            fragment_positions = fragment.positions
            forming_vector = unit(fragment_positions[slot_of[plan.gas_at_group]]
                                  - fragment_positions[slot_of[plan.gas_at_site]])

            gap_axis, gap = mic_unit_vector(atoms, self.free_oxygen, self.migrating_ase)

            if np.linalg.norm(np.cross(forming_vector, gap_axis)) > 1e-6:
                fragment.rotate(forming_vector, gap_axis,
                                center=fragment.positions[slot_of[plan.gas_at_site]])

            gap_midpoint = atoms.positions[self.free_oxygen] + gap_axis * (0.5 * gap)
            fragment_center = 0.5 * (fragment.positions[slot_of[plan.gas_at_site]]
                                     + fragment.positions[slot_of[plan.gas_at_group]])
            fragment.positions += gap_midpoint - fragment_center

            # aligning the forming-bond vector fixes two rotational degrees
            # of freedom; the roll about that vector is still free, and both
            # anchors lie on the axis, so rolling moves only the tail
            tail = [i for i in members if i not in (plan.gas_at_site, plan.gas_at_group)]
            obstacles = [i for i in range(len(atoms)) if i not in self.gas_slots]

            rolls = []
            for k in range(self.roll_steps if tail else 1):
                roll = k * (360.0 / self.roll_steps)
                trial = fragment.copy()
                trial.rotate(roll, gap_axis, center=gap_midpoint)
                if tail:
                    clearance = min_clearance(
                        trial.positions[[slot_of[i] for i in tail]],
                        atoms.positions[obstacles], atoms.cell, atoms.pbc)
                else:
                    clearance = 0.0
                rolls.append((roll, trial, clearance))

            if self.print_roll_table:
                for roll, _, clearance in rolls:
                    print("      %6.1f deg  %.2f A" % (roll, clearance))

            best = max(range(len(rolls)), key=lambda k: rolls[k][2])
            roll, fragment, clearance = rolls[best]

            for i in members:
                atoms.positions[plan.canonical_ase(i)] = fragment.positions[slot_of[i]]

            d_site = atoms.get_distance(plan.canonical_ase(plan.gas_at_site),
                                        self.free_oxygen, mic=True)
            d_group = atoms.get_distance(plan.canonical_ase(plan.gas_at_group),
                                         self.migrating_ase, mic=True)

        return atoms, {"gap": gap, "roll": roll, "clearance": clearance,
                       "site": d_site, "group": d_group}

    def build(self, stretch_start=config.TS_STRETCH_START,
              stretch_step=config.TS_STRETCH_STEP,
              stretch_max=config.TS_STRETCH_MAX):
        """Escalate the stretch until the composite is clash-free (or the
        cap is reached). Returns (atoms, summary dict)."""
        plan = self.plan
        if self.verbose:
            print("\n=== stretch escalation ===")
            print("  breaking bond %s-O%d starts at %.2f A"
                  % (plan.label(plan.migrating), self.old_oxygen, self.break_length))
            print("  %-8s %-7s %-7s %-9s %-9s %-9s %s"
                  % ("stretch", "gap", "roll", "clearance", "d(site)", "d(group)",
                     "clashes fw/gas"))

        stretch = stretch_start
        while True:
            composite, details = self.place(stretch)
            framework_clashes, gas_clashes = self.collisions(composite)
            if self.verbose:
                print("  %-8.2f %-7.2f %-7.1f %-9.2f %-9.2f %-9.2f %d/%d"
                      % (stretch, details["gap"], details["roll"], details["clearance"],
                         details["site"], details["group"],
                         len(framework_clashes), len(gas_clashes)))
            clean = not framework_clashes and not gas_clashes
            if clean or stretch + stretch_step > stretch_max + 1e-9:
                break
            stretch += stretch_step

        if self.verbose and not clean:
            print("  still colliding at the stretch cap -- keeping the last geometry")
            find_clashes(composite, self.framework, self.gas_slots,
                         tolerance=config.CLASH_TOLERANCE, show=True)
            find_clashes(composite, self.surface_slots, self.gas_slots,
                         tolerance=config.CLASH_TOLERANCE, show=True)

        summary = {"stretch": stretch, "clean": clean,
                   "clearance": details["clearance"],
                   "site": details["site"], "group": details["group"],
                   "fw": len(framework_clashes), "gas": len(gas_clashes)}
        return composite, summary


# ==========================================================================
# per pair / per run drivers
# ==========================================================================

def build_pair_guess(reaction, reaction_dir, pair, gas_initial, gas_final,
                     unique_tree, verbose=False, print_roll_table=False):
    """Write ``endpoint_<role>.xyz`` and ``ts_guess.xyz`` for one pair and
    return a summary row."""
    pair_dir = os.path.join(reaction_dir, pair)
    endpoints = reaction["pairs"][pair]
    initial = StoredConfig(os.path.join(pair_dir, INITIAL_XYZ),
                           endpoints["initial"]["species"], "initial", unique_tree)
    final = StoredConfig(os.path.join(pair_dir, FINAL_XYZ),
                         endpoints["final"]["species"], "final", unique_tree)

    plan = InsertionPlan(reaction, initial, final, gas_initial, gas_final, verbose)
    builder = TSGuessBuilder(plan, verbose=verbose, print_roll_table=print_roll_table)
    composite, summary = builder.build()

    endpoint_path = os.path.join(pair_dir, "endpoint_%s.xyz" % plan.canonical_role)
    guess_path = os.path.join(pair_dir, "ts_guess.xyz")
    write(endpoint_path, plan.canonical.atoms)
    write(guess_path, composite)

    if verbose:
        print("\n=== output ===")
        print("  %s side is a relaxed minimum, reordered only" % plan.canonical_role)
        print("  %s side is a constructed saddle guess, not an endpoint"
              % plan.assembled_role)
        print("  wrote %s" % endpoint_path)
        print("  wrote %s" % guess_path)

    summary["pair"] = pair
    return summary


def reaction_dirs(ts_tree, only=None):
    """Sorted ``<i>_rxn`` folder names in the ts_guesses tree."""
    names = sorted(name for name in os.listdir(ts_tree)
                   if _REACTION_DIR.match(name)
                   and os.path.isdir(os.path.join(ts_tree, name)))
    if only:
        names = [name for name in names if name == only]
    return names


def pair_dirs(reaction_dir, only=None):
    names = sorted(name for name in os.listdir(reaction_dir)
                   if _PAIR_DIR.match(name)
                   and os.path.isdir(os.path.join(reaction_dir, name)))
    if only:
        names = [name for name in names if name == only]
    return names


def build_all_guesses(layout, only_reaction=None, only_pair=None, verbose=False,
                      print_roll_table=False):
    """Step 6 over every reaction and pair in the run. Returns
    (n_built, [(reaction, pair, error message)]). ``print_roll_table``
    prints the clearance of every roll angle, not just the winner."""
    if not os.path.isdir(layout.ts_guesses):
        raise FileNotFoundError("%s not found -- run step 5 first" % layout.ts_guesses)
    reactions = reaction_dirs(layout.ts_guesses, only_reaction)
    if not reactions:
        raise FileNotFoundError("no reaction directories in %s" % layout.ts_guesses)

    built, failed = 0, []
    for name in reactions:
        reaction_dir = os.path.join(layout.ts_guesses, name)
        with open(os.path.join(reaction_dir, REACTION_INFO)) as handle:
            reaction = json.load(handle)

        gas_initial = load_gas_configs(reaction["gas"]["initial"], layout.unique)
        gas_final = load_gas_configs(reaction["gas"]["final"], layout.unique)

        print("\n[%d] %s   %s" % (reaction["index"], reaction["reaction"], name))
        if gas_initial and gas_final:
            print("  gas on both sides, nothing to use as canonical order")
            continue
        if not gas_initial and not gas_final:
            print("  no gas species -- reordering only, assembly not implemented")
            continue

        pairs = pair_dirs(reaction_dir, only_pair)
        if not pairs:
            print("  no pair directories")
            continue

        if not verbose:
            print("  %-12s %-8s %-9s %-9s %-9s %s"
                  % ("pair", "stretch", "clearance", "d(site)", "d(group)",
                     "clashes fw/gas"))

        for pair in pairs:
            if verbose:
                print("\n" + "=" * 70)
                print("%s / %s" % (name, pair))
                print("=" * 70)
            try:
                row = build_pair_guess(reaction, reaction_dir, pair,
                                       gas_initial, gas_final, layout.unique,
                                       verbose, print_roll_table)
            except (ValueError, KeyError, FileNotFoundError) as error:
                message = str(error).splitlines()[0]
                failed.append((name, pair, message))
                print("  %-12s failed: %s" % (pair, message))
                if verbose:
                    traceback.print_exc()
                continue
            built += 1
            if not verbose:
                print("  %-12s %-8.2f %-9.2f %-9.2f %-9.2f %d/%d%s"
                      % (row["pair"], row["stretch"], row["clearance"],
                         row["site"], row["group"], row["fw"], row["gas"],
                         "" if row["clean"] else "  (capped)"))

    print("\n%d guesses written, %d failed" % (built, len(failed)))
    for name, pair, message in failed:
        print("  %s/%s: %s" % (name, pair, message))
    return built, failed