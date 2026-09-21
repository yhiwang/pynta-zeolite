import os
import yaml
import numpy as np
from itertools import product, combinations
from ase import Atoms
from ase.io.trajectory import Trajectory
from molecule.molecule import Molecule
from pyntaz.adjlist import AdjacencyStructure
from pyntaz.framework import build_framework
from pyntaz.geometry import min_clearance, rotate_vector, unit

ZEOLITE_CODE = "MOR"
T_LABEL = "T4"
SITE_INDICES = [71, 73]
SITE_BOND_SCALE_RANGE = np.arange(1.0, 2.01, 0.05)
SPAN_TOLERANCE = 0.2
TORSION_STEPS = 24
ROLL_STEPS = 24
CLEARANCE_MIN = 1.5
TRAJ_OUT = "ts_guesses.traj"

REACTION_YAML = """
- index: 0
  reactant: |
    multiplicity 1
    1 *1 C u0 p0 c0 {2,D} {3,S} {4,S}
    2 *2 C u0 p0 c0 {1,D} {5,S} {6,S}
    3 H u0 p0 c0 {1,S}
    4 H u0 p0 c0 {1,S}
    5 H u0 p0 c0 {2,S}
    6 H u0 p0 c0 {2,S}
    7 *3 H u0 p0 c0 {8,S}
    8 *4 X u0 p0 c0 {7,S}
    9 *5 X u0 p0 c0
  product: |
    multiplicity 1
    1 *1 C u0 p0 c0 {2,S} {3,S} {4,S} {7,S}
    2 *2 C u0 p0 c0 {1,S} {5,S} {6,S} {9,S}
    3 H u0 p0 c0 {1,S}
    4 H u0 p0 c0 {1,S}
    5 H u0 p0 c0 {2,S}
    6 H u0 p0 c0 {2,S}
    7 *3 H u0 p0 c0 {1,S}
    8 *4 X u0 p0 c0
    9 *5 X u0 p0 c0 {2,S}
  reaction: C=C + [H][Pt] <=> CC[Pt]
  reaction_family: Surface_Protonation

# - index: 1
#   reactant: |
#     multiplicity 1
#     1 *1 C u0 p0 c0 {2,S} {3,S} {4,S} {5,S}
#     2 *2 C u0 p0 c0 {1,S} {6,S} {7,S} {8,S}
#     3 H u0 p0 c0 {1,S}
#     4 H u0 p0 c0 {1,S}
#     5 *3 X u0 p0 c0 {1,S}
#     6 H u0 p0 c0 {2,S}
#     7 H u0 p0 c0 {2,S}
#     8 H u0 p0 c0 {2,S}
#     9 *4 C u0 p0 c0 {10,D} {11,S} {12,S}
#     10 *5 C u0 p0 c0 {9,D} {13,S} {14,S}
#     11 H u0 p0 c0 {9,S}
#     12 H u0 p0 c0 {9,S}
#     13 H u0 p0 c0 {10,S}
#     14 H u0 p0 c0 {10,S}
#     15 *6 X u0 p0 c0
#   product: |
#     multiplicity 1
#     1 *1 C u0 p0 c0 {2,S} {3,S} {4,S} {9,S}
#     2 *2 C u0 p0 c0 {1,S} {6,S} {7,S} {8,S}
#     3 H u0 p0 c0 {1,S}
#     4 H u0 p0 c0 {1,S}
#     5 *3 X u0 p0 c0
#     6 H u0 p0 c0 {2,S}
#     7 H u0 p0 c0 {2,S}
#     8 H u0 p0 c0 {2,S}
#     9 *4 C u0 p0 c0 {1,S} {10,S} {11,S} {12,S}
#     10 *5 C u0 p0 c0 {9,S} {13,S} {14,S} {15,S}
#     11 H u0 p0 c0 {9,S}
#     12 H u0 p0 c0 {9,S}
#     13 H u0 p0 c0 {10,S}
#     14 H u0 p0 c0 {10,S}
#     15 *6 X u0 p0 c0 {10,S}
#   reaction: CC[Pt] + C=C <=> CCCC[Pt]
#   reaction_family: Surface_Oligomerization
"""

ORDER_STR = {1.0: "S", 1.5: "B", 2.0: "D", 3.0: "T"}
STRETCH = {"form": 1.0, "break": 1.0}


def bond_map(mol):
    ix = {id(a): i for i, a in enumerate(mol.atoms)}
    return {tuple(sorted((ix[id(a)], ix[id(o)]))): b.get_order_num()
            for a in mol.atoms for o, b in a.bonds.items()}


class TSGraph:

    def __init__(self, reactant, product):
        self.elements = [a.element.symbol for a in reactant.atoms]
        self.stars = {i: a.label for i, a in enumerate(reactant.atoms) if a.label}
        self.flags = [(a.radical_electrons, a.lone_pairs, a.charge)
                      for a in reactant.atoms]
        self.multiplicity = reactant.multiplicity
        self.adj = {i: {} for i in range(len(self.elements))}
        self.bonds = {}
        rb, pb = bond_map(reactant), bond_map(product)
        for i, j in sorted(set(rb) | set(pb)):
            before, after = rb.get((i, j)), pb.get((i, j))
            change = ("form" if before is None else
                      "break" if after is None else
                      "order" if before != after else "none")
            order = max(o for o in (before, after) if o is not None)
            self.bonds[(i, j)] = {"order": order, "reactant": before,
                                  "product": after, "change": change,
                                  "stretch": STRETCH.get(change, 1.0)}
            self.adj[i][j] = self.adj[j][i] = order

    def tag(self, i):
        return "%s%d%s" % (self.elements[i], i + 1, self.stars.get(i, ""))

    def sites(self):
        return sorted(i for i, e in enumerate(self.elements) if e == "X")

    def bond_scales(self):
        return {(i + 1, j + 1): info["stretch"]
                for (i, j), info in self.bonds.items()
                if info["stretch"] != 1.0}

    def adjlist(self):
        lines = ["multiplicity %d" % self.multiplicity]
        for i, symbol in enumerate(self.elements):
            u, p, c = self.flags[i]
            bonds = " ".join("{%d,%s}" % (j + 1, ORDER_STR[o])
                             for j, o in sorted(self.adj[i].items()))
            lines.append(" ".join(x for x in (
                str(i + 1), self.stars.get(i, ""), symbol,
                "u%d p%d c%d" % (u, p, c), bonds) if x))
        return "\n".join(lines)


def axial_options(struct):
    """{label: [(a, b), ...]} -- every 2-combination of a five-coordinate
    atom's neighbours, i.e. every candidate axial pair."""
    return {label: list(combinations(neighbors, 2))
            for label, neighbors in struct.hypervalent_atoms().items()}


def effective_torsions(struct):
    """Rotatable bonds whose rotation actually moves the geometry.

    A bond only counts when each end carries at least one non-X neighbour
    besides the other end -- otherwise there is nothing real hanging off
    that side to rotate (a bond to a lone migrating H whose only other
    neighbour is a site X is not a usable torsion). Also skips single-atom
    moving sides and symmetric tops (methyl and the like).
    """
    keep = []
    for key, info in struct.rotatable_bonds().items():
        a, b = key
        a_real = [n for n in struct.neighbors[a]
                  if n != b and struct.element(n) != "X"]
        b_real = [n for n in struct.neighbors[b]
                  if n != a and struct.element(n) != "X"]
        if not a_real or not b_real:
            continue
        moves = info["moves"]
        if len(moves) == 1:
            continue
        hub = key[1]
        others = [m for m in moves if m != hub]
        if others and all(struct.element(m) == "H"
                          and struct.neighbors[m] == [hub] for m in others):
            continue
        keep.append(key)
    return keep


def built_positions(struct, n_atoms):
    out = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        label = i + 1
        if label in struct.site_anchors:
            out[i] = struct.site_anchors[label]["position"]
        else:
            out[i] = struct.positions[label]
    return out


def build_at(adjlist, site_elements, scale, n_atoms, torsions=None, axial=None,
             bond_scales=None):
    struct = AdjacencyStructure.from_adjlist(adjlist, site_elements=site_elements,
                                             site_bond_scale=scale,
                                             bond_scales=bond_scales)
    struct.build(torsions=torsions, axial=axial)
    return built_positions(struct, n_atoms)


def fit_site_scale(adjlist, site_elements, n_atoms, index_a, index_b,
                   position_a, position_b, torsions=None, axial=None,
                   bond_scales=None):
    target = float(np.linalg.norm(np.asarray(position_a) - np.asarray(position_b)))
    best = None
    for scale in SITE_BOND_SCALE_RANGE:
        positions = build_at(adjlist, site_elements, float(scale), n_atoms,
                             torsions, axial, bond_scales)
        span = float(np.linalg.norm(positions[index_a] - positions[index_b]))
        error = abs(span - target)
        if best is None or error < best[2]:
            best = (float(scale), span, error)
    return best


def seat(positions, index_a, index_b, target_a, target_b, roll_deg):
    axis_from = positions[index_b] - positions[index_a]
    axis_to = np.asarray(target_b) - np.asarray(target_a)
    moved = positions - 0.5 * (positions[index_a] + positions[index_b])

    cross = np.cross(unit(axis_from), unit(axis_to))
    if np.linalg.norm(cross) > 1e-8:
        angle = np.degrees(np.arcsin(np.clip(np.linalg.norm(cross), -1.0, 1.0)))
        if np.dot(unit(axis_from), unit(axis_to)) < 0:
            angle = 180.0 - angle
        moved = np.array([rotate_vector(p, unit(cross), angle) for p in moved])

    moved = np.array([rotate_vector(p, unit(axis_to), roll_deg) for p in moved])
    return moved + 0.5 * (np.asarray(target_a) + np.asarray(target_b))


reaction = yaml.safe_load(REACTION_YAML)[0]
react = Molecule().from_adjacency_list(reaction["reactant"])
prod = Molecule().from_adjacency_list(reaction["product"])
react.multiplicity = react.get_radical_count() + 1
prod.multiplicity = prod.get_radical_count() + 1
ts = TSGraph(react, prod)

print(reaction["reaction"], " ", reaction["reaction_family"])
for (i, j), info in sorted(ts.bonds.items()):
    if info["change"] != "none":
        print("  %-6s %-14s %s -> %s   x%.2f"
              % (info["change"], "%s-%s" % (ts.tag(i), ts.tag(j)),
                 ORDER_STR.get(info["reactant"], "-"),
                 ORDER_STR.get(info["product"], "-"), info["stretch"]))
print()
print(ts.adjlist())

bond_scales = ts.bond_scales()
print("\nbond scales:", bond_scales)

survey = AdjacencyStructure.from_adjlist(ts.adjlist())
rotatable = survey.rotatable_bonds()
torsion_keys = sorted(effective_torsions(survey))

options = axial_options(survey)
axial_labels = sorted(options)
axial_grid = list(product(*(options[l] for l in axial_labels)))

print("\n%d rotatable bond(s); %d effective after symmetry filter"
      % (len(rotatable), len(torsion_keys)))
for key in sorted(rotatable):
    mark = "sweep" if key in torsion_keys else "skip (symmetric)"
    print("  %s  %s   moves %s   [%s]"
          % (key, rotatable[key]["note"], rotatable[key]["moves"], mark))
print("\n%d hypervalent atom(s), %d axial combination(s)"
      % (len(axial_labels), len(axial_grid)))
for label in axial_labels:
    print("  atom %d pairs: %s" % (label, options[label]))

zeolite = build_framework(ZEOLITE_CODE, T_LABEL)
available = [site["indices"][0] for site in zeolite.mono_sites]
print("\n%s %s: %d atoms, Al %d" % (ZEOLITE_CODE, T_LABEL,
                                    len(zeolite.atoms), zeolite.al_index))
print("first-shell O indices: %s" % available)

graph_sites = ts.sites()
if len(SITE_INDICES) != len(graph_sites):
    raise SystemExit("reaction has %d site(s) [%s] but %d index(es) given"
                     % (len(graph_sites),
                        ", ".join(ts.stars[i] for i in graph_sites),
                        len(SITE_INDICES)))

unknown = [o for o in SITE_INDICES if o not in available]
if unknown:
    raise SystemExit("given indices %s do not match the site indices %s"
                     % (unknown, available))

site_elements = {}
targets = {}
for graph_index, o_index in zip(graph_sites, SITE_INDICES):
    site_elements[graph_index + 1] = zeolite.atoms[o_index].symbol
    targets[graph_index] = zeolite.atoms.positions[o_index]
    print("  %s -> %s%d %s" % (ts.stars[graph_index],
                               zeolite.atoms[o_index].symbol, o_index,
                               zeolite.atoms.positions[o_index].round(3)))

old_index, free_index = graph_sites
target_old, target_free = targets[old_index], targets[free_index]
target_span = float(np.linalg.norm(target_old - target_free))
print("  span %.2f A" % target_span)

keep = [i for i, e in enumerate(ts.elements) if e != "X"]
n_atoms = len(ts.elements)
step = 360.0 / TORSION_STEPS
n_combos = TORSION_STEPS ** len(torsion_keys)
site_set = set(ts.sites())
binders = [b for x in ts.sites() for b in ts.adj[x]]
tail = [i for i in keep if i not in site_set and i not in binders]

# -- stage 1: fit torsion x axial, keep the ones that seat --------------
print("\ntarget span %.2f A, %d torsion x %d axial = %d builds"
      % (target_span, n_combos, len(axial_grid), n_combos * len(axial_grid)))
survivors = []
for combo in product(range(TORSION_STEPS), repeat=len(torsion_keys)):
    torsions = {key: k * step for key, k in zip(torsion_keys, combo)}
    angles = tuple(round(a) for a in torsions.values())
    for axial_combo in axial_grid:
        axial = {l: pair for l, pair in zip(axial_labels, axial_combo)}
        scale, span, error = fit_site_scale(
            ts.adjlist(), site_elements, n_atoms, old_index, free_index,
            target_old, target_free, torsions, axial, bond_scales)
        if error <= SPAN_TOLERANCE:
            survivors.append((error, angles, axial, scale, span))

survivors.sort(key=lambda s: s[0])
print("%d of %d combos within %.2f A"
      % (len(survivors), n_combos * len(axial_grid), SPAN_TOLERANCE))
for error, angles, axial, scale, span in survivors[:15]:
    print("  torsions %-10s axial %-16s scale %.2f  span %.2f  err %.2f"
          % (angles, axial, scale, span, error))
if not survivors:
    raise SystemExit("no combo within tolerance; loosen SPAN_TOLERANCE")

# -- stage 2: seat each survivor, sweep roll, score tail clearance ------
framework = zeolite.atoms
print("\nseating %d survivors, %d rolls each" % (len(survivors), ROLL_STEPS))
traj = Trajectory(TRAJ_OUT, "w")
written = 0
for error, angles, axial, scale, span in survivors:
    torsions = {key: a for key, a in zip(torsion_keys, angles)}
    positions = build_at(ts.adjlist(), site_elements, scale, n_atoms,
                         torsions, axial, bond_scales)
    best = None
    for k in range(ROLL_STEPS):
        roll = k * (360.0 / ROLL_STEPS)
        placed = seat(positions, old_index, free_index,
                      target_old, target_free, roll)
        clearance = min_clearance(placed[tail], framework.positions,
                                  framework.cell, framework.pbc)
        molecule = Atoms([ts.elements[i] for i in keep], positions=placed[keep],
                         cell=framework.cell, pbc=framework.pbc)
        molecule.info["angles"] = angles
        molecule.info["axial"] = str(axial)
        molecule.info["roll"] = round(roll)
        molecule.info["clearance"] = round(clearance, 2)
        traj.write(framework + molecule)
        written += 1
        if best is None or clearance > best:
            best = clearance
    print("  torsions %-10s axial %-16s span %.2f  best clearance %.2f A"
          % (angles, axial, span, best))
traj.close()
print("\nwrote %d frames to %s" % (written, os.path.abspath(TRAJ_OUT)))