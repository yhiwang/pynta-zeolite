"""Connectivity of a structure against an adjacency list.

``read_graph`` reads RMG-style adjacency list text into the element of every
label and the bonds between labels. It reads the text only (through
``adjlist.parse_adjacency_list``), so it needs no RMG. Labels are the numbers
at the start of the lines, 1-based, as in the text. ``X`` atoms are framework
sites.

A map is ``{label: ASE index}``. A structure fits the graph under a map when

* every mapped atom has the element of its label (an ``X`` label can be any
  atom: you give its index),
* every graph bond is bonded: r < cutoff x (r_i + r_j), r = ASE covalent
  radius (the rule of ``harmonic_relax.bond_cutoff``), minimum image,
* ``strict`` only: no other pair with at least one mapped atom is bonded.
  Not checked: a pair between an ``X`` atom and a framework atom (the
  framework bonds of a seated O are not in the graph), and a pair of two
  atoms that are both outside the map (another molecule, for example).

``connectivity_problems`` lists every way one full map breaks these rules
(the "check" use: you know which atom is which). ``matching_maps`` returns
every map that fits, given the part of the map you know, which must hold
every ``X`` label: with only the ``X`` labels it searches the rest (the
"identify" use), with every label it tests that one map. Equivalent atoms
(the two H on one carbon, for example) give one map each.

Only numpy, ase and networkx here, so it imports without RMG.
"""

import numpy as np
from ase.data import covalent_radii
from networkx import Graph
from networkx.algorithms.isomorphism import GraphMatcher

from .adjlist import parse_adjacency_list


def read_graph(adjlist):
    """{"elements": {label: element}, "bonds": {(a, b), ...}} from adjacency
    list text, labels 1-based as in the text, every bond once with a < b."""
    atoms = parse_adjacency_list(adjlist)
    return {"elements": {label: spec["element"] for label, spec in atoms.items()},
            "bonds": {tuple(sorted((a, b))) for a, spec in atoms.items() for b in spec["bonds"]}}


def _distances_and_limits(atoms, i, others, cutoff):
    """Minimum-image distances from atom i to ``others`` and the bond limit
    cutoff x (r_i + r_j) of each pair."""
    z = atoms.get_atomic_numbers()
    others = np.asarray(others, dtype=int)
    r = atoms.get_distances(i, others, mic=True)
    return r, cutoff * (covalent_radii[z[i]] + covalent_radii[z[others]])


def _check_map(graph, atom_map):
    """Raise ValueError for labels not in the graph or an index used twice."""
    unknown = sorted(set(atom_map) - set(graph["elements"]))
    if unknown:
        raise ValueError("atom_map has labels %s that are not in the graph" % unknown)
    if len(set(atom_map.values())) != len(atom_map):
        raise ValueError("atom_map gives one ASE index to two labels: %s" % atom_map)


def connectivity_problems(atoms, graph, frame, atom_map, strict, cutoff):
    """Every way ``atoms`` breaks the graph under one full map; [] means it fits.

    atoms     ase Atoms
    graph     from read_graph
    frame     indices of the framework atoms in ``atoms``
    atom_map  {label: ASE index} for every label of the graph
    strict    True: also no bond outside the graph on a mapped atom
    cutoff    bonded: r < cutoff x (r_i + r_j)

    Each problem is a dict with "kind" and the "labels" and ASE "atoms" it
    is about (label None: the atom is not in the map):
        element  {"expected", "found"}
        missing  a graph bond is not bonded: {"r", "limit"}, r >= limit
        extra    strict, a bond outside the graph: {"r", "limit"}, r < limit
    """
    _check_map(graph, atom_map)
    unmapped = sorted(set(graph["elements"]) - set(atom_map))
    if unmapped:
        raise ValueError("atom_map has no index for labels %s" % unmapped)
    frame = set(frame)
    problems = []
    for label, i in sorted(atom_map.items()):
        expected = graph["elements"][label]
        if expected != "X" and atoms[i].symbol != expected:
            problems.append({"kind": "element", "labels": (label,), "atoms": (i,),
                             "expected": expected, "found": atoms[i].symbol})
    for a, b in sorted(graph["bonds"]):
        i, j = atom_map[a], atom_map[b]
        r, limit = _distances_and_limits(atoms, i, [j], cutoff)
        if r[0] >= limit[0]:
            problems.append({"kind": "missing", "labels": (a, b), "atoms": (i, j),
                             "r": float(r[0]), "limit": float(limit[0])})
    if not strict:
        return problems

    label_of = {i: label for label, i in atom_map.items()}
    sites = {atom_map[label] for label, element in graph["elements"].items() if element == "X"}
    in_graph = {tuple(sorted((atom_map[a], atom_map[b]))) for a, b in graph["bonds"]}
    seen = set()
    for i in sorted(label_of):
        others = [j for j in range(len(atoms)) if j != i]
        r, limit = _distances_and_limits(atoms, i, others, cutoff)
        for j, rij, lij in zip(others, r, limit):
            pair = (min(i, j), max(i, j))
            if rij >= lij or pair in seen or pair in in_graph:
                continue
            seen.add(pair)
            if (i in sites and j in frame) or (j in sites and i in frame):
                continue                                 # framework bond of a seated O
            problems.append({"kind": "extra", "labels": (label_of.get(pair[0]), label_of.get(pair[1])),
                             "atoms": pair, "r": float(rij), "limit": float(lij)})
    return problems


def matching_maps(atoms, graph, frame, atom_map, strict, cutoff):
    """Every full map {label: ASE index} under which ``atoms`` fits the graph;
    [] means no map fits.

    atom_map is the part you know and must hold every X label. Each other
    label is searched among the atoms that are not in ``frame`` and not
    already in ``atom_map``, with the same element. frame, strict and cutoff
    as in connectivity_problems. Maps come sorted by their ASE indices in
    label order.
    """
    _check_map(graph, atom_map)
    no_site = sorted(label for label, element in graph["elements"].items()
                     if element == "X" and label not in atom_map)
    if no_site:
        raise ValueError("atom_map has no index for the X labels %s" % no_site)
    if set(atom_map) == set(graph["elements"]):
        return [dict(atom_map)] if not connectivity_problems(
            atoms, graph, frame, atom_map, strict, cutoff) else []

    # pattern: the graph; a known label may only meet its own index
    pattern = Graph()
    for label, element in graph["elements"].items():
        pattern.add_node(label, key=("index", atom_map[label]) if label in atom_map
                         else ("element", element))
    pattern.add_edges_from(graph["bonds"])
    # target: the atoms a label can be, bonded by the same rule
    known = {i: label for label, i in atom_map.items()}
    nodes = sorted((set(range(len(atoms))) - set(frame)) | set(known))
    target = Graph()
    for i in nodes:
        target.add_node(i, key=("index", i) if i in known else ("element", atoms[i].symbol))
    for n, i in enumerate(nodes):
        if n + 1 == len(nodes):
            break
        r, limit = _distances_and_limits(atoms, i, nodes[n + 1:], cutoff)
        target.add_edges_from((i, j) for j, rij, lij in zip(nodes[n + 1:], r, limit) if rij < lij)

    matcher = GraphMatcher(target, pattern, node_match=lambda a, b: a["key"] == b["key"])
    maps = []
    for found in matcher.subgraph_monomorphisms_iter():        # {ASE index: label}
        candidate = {label: i for i, label in found.items()}
        if not connectivity_problems(atoms, graph, frame, candidate, strict, cutoff):
            maps.append(dict(sorted(candidate.items())))
    return sorted(maps, key=lambda m: list(m.values()))