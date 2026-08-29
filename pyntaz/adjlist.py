"""Build 3D starting geometries from RMG-style adjacency lists.

An adjacency list gives pure connectivity -- atoms, bond orders, lone
pairs -- and no coordinates::

    1 C u0 p0 c0 {2,D} {3,S} {4,S}
    2 C u0 p0 c0 {1,D} {5,S} {6,S}
    3 H u0 p0 c0 {1,S}
    ...

``AdjacencyStructure`` parses that into a molecular graph and generates a
chemically sensible 3D starting guess, meant to be handed to a MACE / DFT
relaxation rather than used as a final geometry:

* rings are built first, as cyclic polygons whose edge lengths come from
  covalent radii (so heteroatom rings need no special casing), puckered at
  sp3 positions and left planar at sp2 ones;
* fused rings are anchored on the already-built shared edge, spiro rings on
  the shared atom;
* chains grow outward by BFS; each atom's neighbor directions come from its
  steric number (bonded neighbors + lone pairs -- plain VSEPR), distances
  from summed covalent radii scaled by bond order;
* a double bond locks the dihedral across it (ethylene comes out planar);
  a single bond leaves a free torsion that is NOT optimized here --
  ``rotatable_bonds()`` reports every free torsion and ``build(torsions=...)``
  takes the angles you choose, defaulting to anti (180 degrees);
* ``X`` atoms are surface sites, not real atoms: they take part in the
  geometry (so the binding direction is defined) but are kept as anchors in
  ``site_anchors`` and excluded from ``to_ase()`` unless you ask for the
  placeholder.

Only numpy and ase here, same dependency policy as ``geometry.py``, so it
imports fine on a compute node.

Known limits (raise clear errors, not wrong geometry): bridged bicyclics
(two rings sharing more than one edge / two non-adjacent bridgeheads) and
atoms with steric number > 4.
"""

import math
import re
from collections import deque

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii

# bond order symbols in the adjacency list
BOND_ORDERS = {"S": 1.0, "D": 2.0, "T": 3.0, "B": 1.5}

# bond length = factor(order) * (r_i + r_j); factors reproduce the usual
# C-C 1.52 / C=C 1.32 / C~C 1.39 / C#C 1.19 progression from ASE radii
LENGTH_FACTORS = {1.0: 1.00, 1.5: 0.91, 2.0: 0.87, 3.0: 0.78}

SITE_ELEMENT = "X"          # RMG surface-site pseudo element
DEFAULT_SITE_BOND = 2.0     # Angstrom, X-atom anchor distance
DEFAULT_PUCKER = 0.25       # Angstrom, out-of-plane displacement of sp3 ring atoms

TETRA = 109.4712206         # degrees
_ATOM_LINE = re.compile(
    r"^\s*(\d+)\s+(?:\*\d*\s+)?([A-Z][a-z]?)\s+(.*)$")
_BOND_TOKEN = re.compile(r"\{(\d+),([SDTB])\}")
_FLAG_TOKEN = re.compile(r"([upc])([+-]?\d+)")


class AdjacencyListError(ValueError):
    """Malformed adjacency list or a structure this builder cannot place."""


# --------------------------------------------------------------------------
# small vector helpers
# --------------------------------------------------------------------------

def _unit(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise AdjacencyListError("zero-length vector in geometry construction")
    return v / n


def _rotate(v, axis, angle_deg):
    """Rodrigues rotation of ``v`` about the unit vector ``axis``."""
    t = math.radians(angle_deg)
    return (v * math.cos(t)
            + np.cross(axis, v) * math.sin(t)
            + axis * np.dot(axis, v) * (1.0 - math.cos(t)))


def _perp_component(v, axis):
    """Component of ``v`` perpendicular to the unit vector ``axis``."""
    return v - np.dot(v, axis) * axis


def _any_perp(axis):
    """Some unit vector perpendicular to ``axis``."""
    trial = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(trial, axis)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    return _unit(_perp_component(trial, axis))


def _dihedral(p0, p1, p2, p3):
    """Signed dihedral p0-p1-p2-p3 in degrees, right-handed about p1->p2."""
    axis = _unit(p2 - p1)
    m = _perp_component(p0 - p1, axis)
    n = _perp_component(p3 - p2, axis)
    if np.linalg.norm(m) < 1e-8 or np.linalg.norm(n) < 1e-8:
        return 0.0
    m, n = _unit(m), _unit(n)
    return math.degrees(math.atan2(np.dot(np.cross(m, n), axis),
                                   np.dot(m, n)))


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def parse_adjacency_list(text):
    """Parse RMG-style adjacency list text.

    Returns ``{label: {"element", "unpaired", "lone_pairs", "charge",
    "bonds": {other_label: order}}}`` with the original 1-based labels kept,
    since those are what torsion keys and reports refer back to.
    Lines that are blank, a name, or ``multiplicity N`` are skipped.
    """
    atoms = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.lower().startswith("multiplicity"):
            continue
        match = _ATOM_LINE.match(line)
        if match is None:
            continue  # molecule name / comment line
        label, element, rest = int(match.group(1)), match.group(2), match.group(3)
        if label in atoms:
            raise AdjacencyListError("duplicate atom label %d" % label)
        flags = dict((k, int(v)) for k, v in _FLAG_TOKEN.findall(rest))
        bonds = {}
        for other, symbol in _BOND_TOKEN.findall(rest):
            bonds[int(other)] = BOND_ORDERS[symbol]
        atoms[label] = {"element": element,
                        "unpaired": flags.get("u", 0),
                        "lone_pairs": flags.get("p", 0),
                        "charge": flags.get("c", 0),
                        "bonds": bonds}
    if not atoms:
        raise AdjacencyListError("no atom lines found")
    # symmetry / consistency check
    for label, spec in atoms.items():
        for other, order in spec["bonds"].items():
            if other not in atoms:
                raise AdjacencyListError(
                    "atom %d bonds to unknown atom %d" % (label, other))
            back = atoms[other]["bonds"].get(label)
            if back is None or abs(back - order) > 1e-9:
                raise AdjacencyListError(
                    "inconsistent bond between atoms %d and %d" % (label, other))
    return atoms


# --------------------------------------------------------------------------
# ring detection (smallest set of smallest rings, hand rolled)
# --------------------------------------------------------------------------

def _shortest_cycle_through(edge, neighbors):
    """Shortest cycle containing ``edge`` as an ordered atom list, or None.

    BFS from one endpoint to the other with the edge itself removed."""
    a, b = edge
    parent = {a: None}
    queue = deque([a])
    while queue:
        current = queue.popleft()
        for nxt in neighbors[current]:
            if current == a and nxt == b:
                continue  # the removed edge
            if nxt not in parent:
                parent[nxt] = current
                if nxt == b:
                    path = [b]
                    while parent[path[-1]] is not None:
                        path.append(parent[path[-1]])
                    return path[::-1]  # a ... b, closed by edge b-a
                queue.append(nxt)
    return None


def _canonical_cycle(cycle):
    """Rotation/reflection independent key for an ordered cycle."""
    best = None
    n = len(cycle)
    for seq in (cycle, cycle[::-1]):
        start = seq.index(min(seq))
        rotated = tuple(seq[(start + k) % n] for k in range(n))
        if best is None or rotated < best:
            best = rotated
    return best


def find_rings(neighbors):
    """Smallest set of smallest rings as ordered atom lists.

    Candidate rings are the shortest cycle through every edge; a greedy
    pass keeps the smallest ones that are linearly independent over GF(2)
    until the cycle-space rank (E - V + components) is reached.
    """
    edges = sorted({tuple(sorted((a, b)))
                    for a, nbrs in neighbors.items() for b in nbrs})
    labels = list(neighbors)
    # count connected components for the cycle-space rank
    seen, components = set(), 0
    for start in labels:
        if start in seen:
            continue
        components += 1
        stack = [start]
        seen.add(start)
        while stack:
            for nxt in neighbors[stack.pop()]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    rank = len(edges) - len(labels) + components
    if rank == 0:
        return []

    candidates, keys = [], set()
    for edge in edges:
        cycle = _shortest_cycle_through(edge, neighbors)
        if cycle is None:
            continue
        key = _canonical_cycle(cycle)
        if key not in keys:
            keys.add(key)
            candidates.append(cycle)
    candidates.sort(key=len)

    edge_bit = {edge: 1 << k for k, edge in enumerate(edges)}
    basis, rings = [], []
    for cycle in candidates:
        vector = 0
        for k in range(len(cycle)):
            pair = tuple(sorted((cycle[k], cycle[(k + 1) % len(cycle)])))
            vector ^= edge_bit[pair]
        for b in basis:
            vector = min(vector, vector ^ b)
        if vector:
            basis.append(vector)
            rings.append(cycle)
        if len(rings) == rank:
            break
    return rings


def _ring_systems(rings):
    """Group rings that share atoms into fused/spiro systems."""
    systems = []
    for ring in rings:
        ring_atoms = set(ring)
        merged = [ring]
        rest = []
        for system in systems:
            if any(ring_atoms & set(r) for r in system):
                merged.extend(system)
                ring_atoms.update(a for r in system for a in r)
            else:
                rest.append(system)
        systems = rest + [merged]
    return systems


# --------------------------------------------------------------------------
# cyclic polygon: unique convex polygon with given edge lengths
# --------------------------------------------------------------------------

def _cyclic_polygon_angles(lengths):
    """Central angles of the cyclic polygon with the given edge lengths.

    Bisection on the circumradius R: each edge subtends 2*asin(L/2R) and the
    angles must sum to 2 pi. Handles the regular case exactly and slightly
    irregular (heteroatom) rings without any per-size table.
    """
    lengths = np.asarray(lengths, dtype=float)
    lo = lengths.max() / 2.0 + 1e-9

    def angle_sum(radius):
        return 2.0 * np.arcsin(np.clip(lengths / (2.0 * radius), -1, 1)).sum()

    if angle_sum(lo) < 2.0 * math.pi:
        raise AdjacencyListError(
            "no convex cyclic polygon for edge lengths %s" % lengths.round(3))
    hi = lo
    while angle_sum(hi) > 2.0 * math.pi:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if angle_sum(mid) > 2.0 * math.pi:
            lo = mid
        else:
            hi = mid
    radius = 0.5 * (lo + hi)
    return radius, 2.0 * np.arcsin(np.clip(lengths / (2.0 * radius), -1, 1))


def _planar_ring_coords(lengths):
    """(n, 3) coordinates of the ring vertices in the xy plane, centered at
    the origin. ``lengths[k]`` is the edge between vertex k and k+1."""
    radius, angles = _cyclic_polygon_angles(lengths)
    theta = np.concatenate([[0.0], np.cumsum(angles[:-1])])
    coords = np.zeros((len(lengths), 3))
    coords[:, 0] = radius * np.cos(theta)
    coords[:, 1] = radius * np.sin(theta)
    return coords - coords.mean(axis=0)


# --------------------------------------------------------------------------
# the structure class
# --------------------------------------------------------------------------

class AdjacencyStructure:
    """Molecular graph from an adjacency list, plus 3D structure generation.

    Typical use::

        mol = AdjacencyStructure.from_adjlist(text)
        print(mol.report())              # rings, sites, free torsions
        mol.rotatable_bonds()            # {(i, j): {...}, ...}
        atoms = mol.build(torsions={(2, 3): 60.0})   # ase.Atoms, no X
        atoms_with_site = mol.to_ase(include_sites=True)
        mol.site_anchors                 # {x_label: {"position", "bonded_to"}}
    """

    def __init__(self, atom_specs, site_bond_length=DEFAULT_SITE_BOND):
        self.spec = atom_specs
        self.labels = sorted(atom_specs)
        self.site_bond_length = site_bond_length
        self.neighbors = {label: sorted(atom_specs[label]["bonds"])
                          for label in self.labels}
        self.rings = find_rings(self.neighbors)
        self.ring_systems = _ring_systems(self.rings)
        self.ring_bonds = {tuple(sorted((r[k], r[(k + 1) % len(r)])))
                           for r in self.rings for k in range(len(r))}
        self.site_labels = [l for l in self.labels
                            if atom_specs[l]["element"] == SITE_ELEMENT]
        self.positions = None      # {label: np.array}, set by build()
        self.site_anchors = {}     # set by build()
        self._validate()

    # -- constructors ------------------------------------------------------

    @classmethod
    def from_adjlist(cls, text, **kwargs):
        return cls(parse_adjacency_list(text), **kwargs)

    @classmethod
    def from_file(cls, path, **kwargs):
        with open(path) as handle:
            return cls(parse_adjacency_list(handle.read()), **kwargs)

    # -- graph queries -----------------------------------------------------

    def element(self, label):
        return self.spec[label]["element"]

    def bond_order(self, a, b):
        return self.spec[a]["bonds"][b]

    def steric_number(self, label):
        """Bonded neighbors + lone pairs: the VSEPR electron-domain count."""
        return len(self.neighbors[label]) + self.spec[label]["lone_pairs"]

    def hybridization(self, label):
        return {2: "sp", 3: "sp2", 4: "sp3"}.get(self.steric_number(label),
                                                 "other")

    def is_ring_atom(self, label):
        return any(label in ring for ring in self.rings)

    def bond_length(self, a, b):
        if SITE_ELEMENT in (self.element(a), self.element(b)):
            return self.site_bond_length
        radii = sum(covalent_radii[atomic_numbers[self.element(x)]]
                    for x in (a, b))
        return LENGTH_FACTORS.get(self.bond_order(a, b), 1.0) * radii

    def formula(self):
        counts = {}
        for label in self.labels:
            element = self.element(label)
            counts[element] = counts.get(element, 0) + 1
        return "".join(sym + (str(counts[sym]) if counts[sym] > 1 else "")
                       for sym in sorted(counts))

    def _validate(self):
        for label in self.labels:
            element = self.element(label)
            if element != SITE_ELEMENT and element not in atomic_numbers:
                raise AdjacencyListError("unknown element %r on atom %d"
                                         % (element, label))
            if (element != SITE_ELEMENT
                    and self.steric_number(label) > 4
                    and self.neighbors[label]):
                raise AdjacencyListError(
                    "atom %d (%s) has steric number %d; only up to 4 "
                    "(sp3) is supported" % (label, element,
                                            self.steric_number(label)))
        for system in self.ring_systems:
            self._check_ring_system(system)

    def _check_ring_system(self, system):
        """Bridged bicyclics are out of scope for v1: any pair of rings in a
        system may share at most one edge (fused) or one atom (spiro)."""
        for i, ring_a in enumerate(system):
            for ring_b in system[i + 1:]:
                shared = set(ring_a) & set(ring_b)
                if len(shared) > 2:
                    raise AdjacencyListError(
                        "bridged ring system (rings share atoms %s); not "
                        "supported yet" % sorted(shared))
                if len(shared) == 2:
                    pair = tuple(sorted(shared))
                    if pair not in self.ring_bonds:
                        raise AdjacencyListError(
                            "rings share two non-adjacent atoms %s (bridged);"
                            " not supported yet" % sorted(shared))

    # -- torsions ----------------------------------------------------------

    def _start_atom(self):
        """Rings first; else a site X; else the busiest atom."""
        if self.ring_systems:
            biggest = max(self.ring_systems,
                          key=lambda s: len({a for r in s for a in r}))
            return min(a for r in biggest for a in r)
        if self.site_labels:
            return self.site_labels[0]
        return max(self.labels, key=lambda l: (len(self.neighbors[l]), -l))

    def _moving_side(self, i, j):
        """Atoms on the j side when bond (i, j) is cut."""
        seen, stack = {i, j}, [j]
        side = []
        while stack:
            current = stack.pop()
            side.append(current)
            for nxt in self.neighbors[current]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return sorted(side)

    def rotatable_bonds(self):
        """Free torsions: ``{(i, j): {"refs", "moves", "default"}}``.

        A bond is rotatable when it is a single bond, not in a ring, does not
        involve an X site, and both ends have at least two neighbors (there
        is nothing to rotate on a terminal atom). The key is oriented so that
        ``i`` is on the start-atom side and the ``moves`` atoms hang off
        ``j``. The angle you pass to ``build`` is the dihedral
        refs[0] - i - j - refs[1] in degrees; refs are the lowest-label
        neighbor on each side, so the zero is deterministic.
        """
        start = self._start_atom()
        result = {}
        for a in self.labels:
            for b in self.neighbors[a]:
                if a >= b:
                    continue
                if self.bond_order(a, b) != 1.0:
                    continue
                if tuple(sorted((a, b))) in self.ring_bonds:
                    continue
                if SITE_ELEMENT in (self.element(a), self.element(b)):
                    continue
                if len(self.neighbors[a]) < 2 or len(self.neighbors[b]) < 2:
                    continue
                if start == a or start in self._moving_side(b, a):
                    i, j = a, b     # start sits on the a side; b side moves
                else:
                    i, j = b, a
                ref_i = min(n for n in self.neighbors[i] if n != j)
                ref_j = min(n for n in self.neighbors[j] if n != i)
                result[(i, j)] = {
                    "refs": (ref_i, ref_j),
                    "moves": self._moving_side(i, j),
                    "default": 180.0,
                    "note": "dihedral %d-%d-%d-%d" % (ref_i, i, j, ref_j)}
        return result

    def _torsion_for(self, i, j, torsions):
        for key in ((i, j), (j, i)):
            if key in torsions:
                return float(torsions[key])
        return 180.0

    # -- build -------------------------------------------------------------

    def build(self, torsions=None, pucker=DEFAULT_PUCKER):
        """Generate coordinates; returns an ``ase.Atoms`` without X sites.

        ``torsions`` maps rotatable-bond keys (either orientation) to
        dihedral angles in degrees; anything unspecified defaults to 180
        (anti). Unknown or non-rotatable keys raise, so a typo cannot be
        silently ignored. ``pucker`` is the out-of-plane displacement of sp3
        ring atoms in Angstrom (0 gives flat rings).
        """
        torsions = dict(torsions or {})
        allowed = self.rotatable_bonds()
        for key in torsions:
            pair = (key[0], key[1])
            if pair not in allowed and (pair[1], pair[0]) not in allowed:
                raise AdjacencyListError(
                    "torsion key %s is not a rotatable bond; options: %s"
                    % (key, sorted(allowed)))

        pos = {}
        built_systems = [False] * len(self.ring_systems)
        system_of = {}
        for index, system in enumerate(self.ring_systems):
            for ring in system:
                for atom in ring:
                    system_of[atom] = index

        start = self._start_atom()
        queue = deque([start])
        if start not in system_of:
            pos[start] = np.zeros(3)
        enqueued = {start}

        while queue:
            current = queue.popleft()
            index = system_of.get(current)
            if index is not None and not built_systems[index]:
                new_ring_atoms = self._build_ring_system(
                    self.ring_systems[index], current, pos, torsions, pucker)
                built_systems[index] = True
                for atom in new_ring_atoms:
                    if atom not in enqueued:
                        enqueued.add(atom)
                        queue.append(atom)
            if current not in pos:
                continue  # placed later by its ring system
            fresh = self._place_neighbors(current, pos, torsions)
            for atom in fresh:
                if atom not in enqueued:
                    enqueued.add(atom)
                    queue.append(atom)

        missing = [l for l in self.labels if l not in pos]
        if missing:
            raise AdjacencyListError("disconnected atoms never placed: %s"
                                     % missing)
        self.positions = pos
        self.site_anchors = {
            x: {"position": pos[x].copy(),
                "bonded_to": list(self.neighbors[x])}
            for x in self.site_labels}
        return self.to_ase(include_sites=False)

    # -- chain placement ---------------------------------------------------

    def _place_neighbors(self, j, pos, torsions):
        """Place the unplaced neighbors of the placed atom ``j``."""
        new = [n for n in self.neighbors[j] if n not in pos]
        if not new:
            return []
        anchored = [n for n in self.neighbors[j] if n in pos]
        steric = self.steric_number(j)
        if self.element(j) == SITE_ELEMENT:
            steric = max(len(self.neighbors[j]), 1)
        center = pos[j]

        if not anchored:
            directions = self._fresh_directions(steric)
        elif len(anchored) == 1:
            directions = self._directions_one_anchor(
                j, anchored[0], len(new), steric, pos, torsions)
        else:
            directions = self._directions_many_anchors(
                j, anchored, len(new), steric, pos)

        if len(directions) < len(new):
            raise AdjacencyListError(
                "cannot place %d new neighbors of atom %d (steric %d, %d "
                "already placed)" % (len(new), j, steric, len(anchored)))
        for label, direction in zip(new, directions):
            pos[label] = center + direction * self.bond_length(j, label)
        return new

    @staticmethod
    def _fresh_directions(steric):
        if steric <= 1:
            return [np.array([0.0, 0.0, 1.0])]
        if steric == 2:
            return [np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])]
        if steric == 3:
            return [np.array([math.cos(a), math.sin(a), 0.0])
                    for a in (0.0, 2 * math.pi / 3, -2 * math.pi / 3)]
        root = 1.0 / math.sqrt(3.0)
        return [np.array(v) * root for v in
                ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))]

    def _directions_one_anchor(self, j, i, n_new, steric, pos, torsions):
        """Directions for j's new neighbors when only ``i`` is placed.

        A single bond gives a free torsion (user angle or anti default); a
        double/triple bond locks the azimuth to i's substituent plane, which
        is what keeps pi systems planar.
        """
        axis = _unit(pos[j] - pos[i])        # i -> j, dihedral axis
        toward_parent = -axis
        placed_refs = [n for n in self.neighbors[i] if n != j and n in pos]
        if placed_refs:
            reference = _perp_component(pos[min(placed_refs)] - pos[i], axis)
            m_hat = (_unit(reference) if np.linalg.norm(reference) > 1e-8
                     else _any_perp(axis))
        else:
            m_hat = _any_perp(axis)

        locked = self.bond_order(i, j) > 1.0
        if locked or self.element(i) == SITE_ELEMENT:
            tau = 0.0
        else:
            tau = self._torsion_for(i, j, torsions)

        if steric >= 4:
            beta, azimuths = TETRA, [tau, tau + 120.0, tau - 120.0]
        elif steric == 3:
            beta, azimuths = 120.0, [tau, tau + 180.0]
        elif steric == 2:
            beta, azimuths = 180.0, [tau]
        else:
            beta, azimuths = TETRA, [tau + k * 360.0 / max(n_new, 1)
                                     for k in range(n_new)]

        directions = []
        for azimuth in azimuths[:max(n_new, 1)]:
            n_hat = _unit(_rotate(m_hat, axis, azimuth))
            directions.append(_unit(math.cos(math.radians(beta)) * toward_parent
                                    + math.sin(math.radians(beta)) * n_hat))
        return directions

    def _directions_many_anchors(self, j, anchored, n_new, steric, pos):
        """Directions when two or more neighbors of ``j`` are already placed
        (ring atoms, junctions): the local frame is fully determined."""
        units = [_unit(pos[n] - pos[j]) for n in anchored]
        if len(units) >= 3 or steric <= len(units) + 1:
            filler = -sum(units)
            if np.linalg.norm(filler) < 1e-8:
                filler = _any_perp(units[0])
            return [_unit(filler)]
        u1, u2 = units[0], units[1]
        bisector_out = -(u1 + u2)
        if np.linalg.norm(bisector_out) < 1e-8:      # anchors collinear
            bisector_out = _any_perp(u1)
        bisector_out = _unit(bisector_out)
        if steric == 3:
            return [bisector_out]
        normal = np.cross(u1, u2)
        if np.linalg.norm(normal) < 1e-8:
            normal = _any_perp(u1)
        normal = _unit(normal)
        half = math.radians(TETRA / 2.0)
        return [_unit(math.cos(half) * bisector_out + math.sin(half) * normal),
                _unit(math.cos(half) * bisector_out - math.sin(half) * normal)]

    # -- ring systems ------------------------------------------------------

    def _build_ring_system(self, system, entry, pos, torsions, pucker):
        """Place every atom of a fused/spiro ring system.

        The first ring is chosen to contain ``entry`` (the BFS atom that
        reached the system); each further ring must share atoms with what is
        already built -- one shared edge is a fusion, one shared atom a
        spiro center.
        """
        remaining = sorted(system, key=len)
        ordered = []
        containing = [r for r in remaining if entry in r]
        first = min(containing, key=len)
        ordered.append(first)
        remaining.remove(first)
        placed_atoms = set(first)
        while remaining:
            nxt = next((r for r in remaining if set(r) & placed_atoms), None)
            if nxt is None:
                raise AdjacencyListError("ring system is not connected")
            ordered.append(nxt)
            remaining.remove(nxt)
            placed_atoms.update(nxt)

        newly_placed = []
        planes = []
        for ring in ordered:
            fresh = self._place_one_ring(ring, entry, pos, torsions,
                                         pucker, planes)
            newly_placed.extend(fresh)
        return newly_placed

    def _ring_edge_lengths(self, ring, pos):
        lengths = []
        for k in range(len(ring)):
            a, b = ring[k], ring[(k + 1) % len(ring)]
            if a in pos and b in pos:
                lengths.append(float(np.linalg.norm(pos[a] - pos[b])))
            else:
                lengths.append(self.bond_length(a, b))
        return lengths

    def _pucker_pattern(self, ring, fixed, pucker):
        """Alternating out-of-plane displacement, zero at sp2 and fixed
        atoms; on an odd all-sp3 ring the wrap-around atom stays flat
        (envelope-like). An sp3-by-VSEPR atom sitting between two sp2 ring
        neighbors also stays flat -- that covers conjugation the electron
        count alone misses (the O of furan, the CH2 of cyclopentadiene)."""
        deltas, sign = [], 1.0
        n = len(ring)
        for k, atom in enumerate(ring):
            prev_h = self.hybridization(ring[k - 1])
            next_h = self.hybridization(ring[(k + 1) % n])
            conjugated = prev_h != "sp3" and next_h != "sp3"
            flat = (self.hybridization(atom) != "sp3" or atom in fixed
                    or conjugated
                    or (n % 2 == 1 and k == n - 1))
            deltas.append(0.0 if flat else sign * pucker)
            sign = -sign
        return deltas

    def _place_one_ring(self, ring, entry, pos, torsions, pucker, planes):
        shared = [a for a in ring if a in pos]
        fixed = set(shared)
        deltas = self._pucker_pattern(ring, fixed, pucker)

        # planar solve on effective edge lengths so the puckered 3D bond
        # comes back to the target length
        lengths = self._ring_edge_lengths(ring, pos)
        effective = []
        for k in range(len(ring)):
            dz = deltas[k] - deltas[(k + 1) % len(ring)]
            effective.append(math.sqrt(max(lengths[k] ** 2 - dz ** 2,
                                           0.25 * lengths[k] ** 2)))
        flat = _planar_ring_coords(effective)
        local = {atom: flat[k] + np.array([0.0, 0.0, deltas[k]])
                 for k, atom in enumerate(ring)}

        if len(shared) == 0:
            placed = self._orient_first_ring(ring, entry, local, pos, torsions)
        elif len(shared) == 1:
            placed = self._orient_shared_atom(ring, shared[0], local, pos,
                                              torsions)
        elif len(shared) == 2:
            placed = self._orient_shared_edge(ring, shared, local, pos, planes)
        else:
            raise AdjacencyListError(
                "ring shares %d placed atoms; bridged systems are not "
                "supported yet" % len(shared))

        centroid = np.mean([pos[a] for a in ring], axis=0)
        span = np.array([pos[a] for a in ring]) - centroid
        _, _, vt = np.linalg.svd(span)
        planes.append({"atoms": list(ring), "centroid": centroid,
                       "normal": vt[2]})
        return placed

    def _orient_first_ring(self, ring, entry, local, pos, torsions):
        """No atom placed yet (molecule starts here): local coords are
        final, up to putting the entry atom where BFS expects the origin."""
        offset = -local[entry] if entry in local else np.zeros(3)
        for atom, coord in local.items():
            pos[atom] = coord + offset
        return list(ring)

    def _frame_transform(self, x_local, z_local, x_target, z_target):
        """Rotation taking the local (x, z) frame onto the target frame."""
        def frame(x, z):
            x = _unit(x)
            z = _unit(_perp_component(z, x)) if np.linalg.norm(
                _perp_component(z, x)) > 1e-8 else _any_perp(x)
            return np.column_stack([x, np.cross(z, x), z])
        return frame(x_target, z_target) @ frame(x_local, z_local).T

    def _orient_shared_atom(self, ring, s, local, pos, torsions):
        """One shared placed atom: chain entry into a ring, or a spiro
        center. The ring's inward bisector at ``s`` is aimed away from
        (entry) or between (spiro) the existing bonds."""
        k = ring.index(s)
        prev_atom, next_atom = ring[k - 1], ring[(k + 1) % len(ring)]
        inward_local = _unit((local[prev_atom] - local[s])
                             + (local[next_atom] - local[s]))
        normal_local = np.array([0.0, 0.0, 1.0])

        anchors = [n for n in self.neighbors[s] if n in pos]
        units = [_unit(pos[n] - pos[s]) for n in anchors]
        if len(units) == 1:
            inward_target = -units[0]           # ring points away from parent
            normal_target = _any_perp(inward_target)
        else:
            # spiro: the new ring's bonds occupy the directions the old
            # bonds left free -- inward along the outward bisector of the
            # old bonds, ring plane perpendicular to the old bond plane
            inward_target = -sum(units)
            if np.linalg.norm(inward_target) < 1e-8:
                inward_target = _any_perp(units[0])
            inward_target = _unit(inward_target)
            cross = np.cross(units[0], units[1])
            normal_target = (_unit(cross) if np.linalg.norm(cross) > 1e-8
                             else _any_perp(inward_target))

        rotation = self._frame_transform(inward_local, normal_local,
                                         inward_target, normal_target)
        placed = []
        for atom in ring:
            if atom == s:
                continue
            pos[atom] = pos[s] + rotation @ (local[atom] - local[s])
            placed.append(atom)

        # chain entry keeps its reported torsion meaningful: rotate the ring
        # about the parent bond to hit the requested dihedral
        if len(units) == 1:
            parent = anchors[0]
            parent_refs = [n for n in self.neighbors[parent]
                           if n != s and n in pos]
            ref_s = min(n for n in self.neighbors[s] if n != parent)
            key_pair = tuple(sorted((parent, s)))
            if parent_refs and key_pair not in self.ring_bonds:
                tau = self._torsion_for(parent, s, torsions)
                current = _dihedral(pos[min(parent_refs)], pos[parent],
                                    pos[s], pos[ref_s])
                axis = _unit(pos[s] - pos[parent])
                spin = tau - current
                for atom in placed:
                    pos[atom] = pos[s] + _rotate(pos[atom] - pos[s], axis,
                                                 spin)
        return placed

    def _orient_shared_edge(self, ring, shared, local, pos, planes):
        """Two shared placed atoms: ordinary fusion on an edge. The new ring
        starts coplanar with the old one on the far side of the edge, then
        folds out of plane when the junction is saturated."""
        p, q = shared
        if tuple(sorted((p, q))) not in self.ring_bonds:
            raise AdjacencyListError(
                "rings share non-adjacent atoms %s; bridged systems are "
                "not supported yet" % sorted(shared))
        edge_local = _unit(local[q] - local[p])
        # interior of the local ring, perpendicular to the edge
        centroid_local = np.mean([local[a] for a in ring], axis=0)
        in_local = _unit(_perp_component(centroid_local - local[p],
                                         edge_local))

        edge_target = _unit(pos[q] - pos[p])
        old = next((pl for pl in planes if p in pl["atoms"]
                    and q in pl["atoms"]), planes[-1] if planes else None)
        if old is not None:
            away = _perp_component(0.5 * (pos[p] + pos[q]) - old["centroid"],
                                   edge_target)
            in_target = (_unit(away) if np.linalg.norm(away) > 1e-8
                         else _any_perp(edge_target))
        else:
            in_target = _any_perp(edge_target)

        rotation = self._frame_transform(edge_local, np.cross(edge_local,
                                                              in_local),
                                         edge_target, np.cross(edge_target,
                                                               in_target))
        placed = []
        for atom in ring:
            if atom in (p, q):
                continue
            pos[atom] = pos[p] + rotation @ (local[atom] - local[p])
            placed.append(atom)

        # saturated junction: fold the new ring out of the old plane
        if (self.hybridization(p) == "sp3" or self.hybridization(q) == "sp3"):
            fold = 180.0 - 115.0
            axis = edge_target
            for atom in placed:
                pos[atom] = pos[p] + _rotate(pos[atom] - pos[p], axis, fold)
        return placed

    # -- output ------------------------------------------------------------

    def to_ase(self, include_sites=False):
        """``ase.Atoms`` in adjacency-list label order. X sites are left out
        unless ``include_sites`` is set, in which case they appear as ASE's
        'X' placeholder (Z = 0) so viewers can show the anchor.

        ``atoms.info['adjlist_labels']`` maps ase index -> original label.
        """
        if self.positions is None:
            raise AdjacencyListError("call build() before to_ase()")
        labels = [l for l in self.labels
                  if include_sites or self.element(l) != SITE_ELEMENT]
        atoms = Atoms([self.element(l) for l in labels],
                      positions=[self.positions[l] for l in labels])
        atoms.info["adjlist_labels"] = labels
        return atoms

    def report(self):
        """Human-readable summary: what to look at before choosing torsions."""
        lines = ["AdjacencyStructure: %s (%d atoms)"
                 % (self.formula(), len(self.labels))]
        if self.rings:
            for ring in self.rings:
                kinds = "/".join(self.hybridization(a) for a in ring)
                lines.append("  ring %s  (%s)"
                             % ("-".join("%s%d" % (self.element(a), a)
                                         for a in ring), kinds))
            if len(self.ring_systems) != len(self.rings):
                lines.append("  (%d ring systems)" % len(self.ring_systems))
        else:
            lines.append("  no rings")
        for x in self.site_labels:
            lines.append("  site X%d bound to %s" % (
                x, ", ".join("%s%d" % (self.element(n), n)
                             for n in self.neighbors[x])))
        rotatable = self.rotatable_bonds()
        if rotatable:
            lines.append("  rotatable bonds (key: dihedral refs, default 180):")
            for (i, j), info in sorted(rotatable.items()):
                lines.append("    (%d, %d)  %s   moves %s"
                             % (i, j, info["note"], info["moves"]))
        else:
            lines.append("  no rotatable bonds")
        return "\n".join(lines)