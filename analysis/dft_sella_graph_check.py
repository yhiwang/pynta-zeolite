#!/usr/bin/env python
"""TS graph check of the DFT Sella runs (step 12): do the start and the end of
every run still have the connectivity of the reaction's TS graph?

    start   frame 0 of the first trajectory part (sella.0.traj after a resume,
            else sella.traj): DFT at the MACE NEB TS, before Sella moved
    end     ts_dft.xyz: where Sella stopped

A structure matches the TS graph (TS_unique/<i>_rxn/info.json, in the same
run folder as DFT_TS) when
    - every graph bond that does not form or break is bonded:
      r < DFT_SELLA_BOND_CUTOFF x (r_i + r_j), r = ASE covalent radius
    - every forming/breaking bond is partly bonded:
      r < DFT_SELLA_BOND_CUTOFF x (r_i + r_j) + DFT_SELLA_TS_BOND_TOLERANCE
    - no other pair with an adsorbate atom is bonded (step 7's
      adsorbate_bonds, which checks every adsorbate atom against every atom)

Graph verdict of a TS:
    ok          the start and the end both match
    not ok      the start or the end does not match
    no start    frame 0 cannot be read; only the end is checked
    no result   no result.json yet; nothing is checked

Before it checks a TS, it maps the graph atoms to structure indices
(guess_atom_map, with the ordered oxygens of ts_info.json) and stops if the
mapped forming/breaking bonds differ from the ones in ts_info.json, or if a
mapped index holds the wrong element (check_atom_order).

Runs alone and prints every TS with its problems; dft_sella_table.py imports
it to rank the TSs of a cell.

    python analysis/dft_sella_graph_check.py
"""

import glob
import json
import os
import re
import sys

from ase.io import read

import _common  # noqa: F401
import layout
from pyntaz.harmonic_relax import (REACTING_CHANGES, adsorbate_bonds, bond_cutoff,
                                   check_atom_order, guess_atom_map, guess_bond_indices)
from pyntaz.reactions import molecule_from_adjlist
from pyntaz.ts_graph import TSGraph

# ---- settings -------------------------------------------------------------------
DFT_TS_DIR = "runs/MOR_T4/DFT_TS"     # TS_unique is read from the same run folder

DFT_SELLA_BOND_CUTOFF = 1.2          # bonded: r < this x (r_i + r_j)
DFT_SELLA_TS_BOND_TOLERANCE = 0.8    # A, forming/breaking: r < cutoff x (r_i + r_j) + this


# ---- files ------------------------------------------------------------------------
def dft_sella_dirs(dft_ts_dir):
    """Every dft_sella/ folder under ``dft_ts_dir``, in sorted order."""
    found = []
    for root, dirs, files in os.walk(dft_ts_dir):
        dirs.sort()
        if os.path.basename(root) == layout.DFT_SELLA_DIR:
            found.append(root)
            dirs[:] = []                                 # not into vasp/
    return found


def load_info(folder):
    with open(os.path.join(folder, layout.TS_COLLECT_INFO)) as handle:
        return json.load(handle)


def first_part(folder):
    """sella.0.traj after a resume, else sella.traj: the file with frame 0."""
    number = re.compile(r"sella\.(\d+)\.traj$")
    parts = sorted((int(number.search(p).group(1)), p)
                   for p in glob.glob(os.path.join(folder, "sella.*.traj")) if number.search(p))
    return parts[0][1] if parts else os.path.join(folder, layout.TS_SELLA_TRAJECTORY)


def read_start(folder):
    """(frame 0, its DFT energy), or (None, None) when the file is missing,
    empty or unreadable, or the frame has no energy."""
    path = first_part(folder)
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return None, None
    try:
        atoms = read(path, 0)
        return atoms, float(atoms.get_potential_energy())
    except Exception:                                    # noqa: BLE001
        return None, None


# ---- the check --------------------------------------------------------------------
def distance(atoms, i, j):
    return float(atoms.get_distance(i, j, mic=True))


def graph_problems(atoms, fixed, changing, n_framework):
    """[(kind, i, j, r, limit)] for every way ``atoms`` differs from the TS
    graph; [] means it matches. kind is "missing" (a graph bond that does not
    form or break is not bonded: r >= limit), "too long" (a forming/breaking
    bond past the tolerance: r >= limit) or "extra" (a bonded pair outside
    the graph: r < limit). r and limit in A."""
    out = []
    for i, j in sorted(fixed):
        r, limit = distance(atoms, i, j), bond_cutoff(atoms, i, j, DFT_SELLA_BOND_CUTOFF)
        if r >= limit:
            out.append(("missing", i, j, r, limit))
    for i, j in changing:
        r = distance(atoms, i, j)
        limit = bond_cutoff(atoms, i, j, DFT_SELLA_BOND_CUTOFF) + DFT_SELLA_TS_BOND_TOLERANCE
        if r >= limit:
            out.append(("too long", i, j, r, limit))
    found = adsorbate_bonds(atoms, n_framework, DFT_SELLA_BOND_CUTOFF, skip=changing)
    for i, j in sorted(found - fixed):
        out.append(("extra", i, j, distance(atoms, i, j),
                    bond_cutoff(atoms, i, j, DFT_SELLA_BOND_CUTOFF)))
    return out


_graphs = {}


def ts_graph(ts_unique_dir, index):
    """TSGraph of one reaction from TS_unique/<i>_rxn/info.json (as step 7)."""
    if (ts_unique_dir, index) not in _graphs:
        folders = {int(name.split("_")[0]): path
                   for name, path in layout.species_dirs(ts_unique_dir)
                   if name.split("_")[0].isdigit()}
        if index not in folders:
            sys.exit("no folder for reaction %d in %s" % (index, ts_unique_dir))
        with open(os.path.join(folders[index], layout.REACTION_INFO)) as handle:
            info = json.load(handle)
        _graphs[(ts_unique_dir, index)] = TSGraph(molecule_from_adjlist(info["reactant"]),
                                                  molecule_from_adjlist(info["product"]))
    return _graphs[(ts_unique_dir, index)]


def check_ts(folder, dft_ts_dir):
    """Graph check of one dft_sella/ folder:
    {"id", "status", "problems": {"start": [...], "end": [...]}, "labels", "symbols"}."""
    info = load_info(folder)
    index = info["where"]["reaction_index"]
    oxygens = info["site"]["oxygens"]
    n_framework = info["atoms"]["n_framework"]
    ts_unique_dir = os.path.join(os.path.dirname(os.path.normpath(dft_ts_dir)), "TS_unique")
    ts = ts_graph(ts_unique_dir, index)

    mapped = {tuple(sorted(b["indices"])) for b in guess_bond_indices(ts, oxygens, n_framework)}
    stored = {tuple(sorted(b["indices"])) for b in info["bonds"]}
    if mapped != stored:
        sys.exit("%s: the TS graph maps the forming/breaking bonds to %s, ts_info.json has %s"
                 % (info["id"], sorted(mapped), sorted(stored)))
    where = guess_atom_map(ts, oxygens, n_framework)
    fixed = {tuple(sorted((where[a], where[b]))) for (a, b), bond in ts.bonds.items()
             if bond["change"] not in REACTING_CHANGES}
    changing = [tuple(sorted(b["indices"])) for b in info["bonds"]]
    check = {"id": info["id"], "status": "no result", "problems": {}, "symbols": [],
             "labels": {i: ts.tag(g) + ("=O" if ts.elements[g] == "X" else "")
                        for g, i in where.items()}}

    if not os.path.isfile(os.path.join(folder, layout.TS_SELLA_RESULT)):
        return check
    end = read(os.path.join(folder, layout.DFT_SELLA_STRUCTURE))
    start, _ = read_start(folder)
    for name, atoms in (("end", end), ("start", start)):
        if atoms is not None:
            try:
                check_atom_order(atoms, ts, oxygens, n_framework)
            except ValueError as error:
                sys.exit("%s (%s): %s" % (info["id"], name, error))
    check["symbols"] = end.get_chemical_symbols()
    check["problems"]["end"] = graph_problems(end, fixed, changing, n_framework)
    if start is None:
        check["status"] = "no start"
        return check
    check["problems"]["start"] = graph_problems(start, fixed, changing, n_framework)
    check["status"] = "not ok" if any(check["problems"].values()) else "ok"
    return check


def atom_name(check, i):
    """Graph tag (or element) and index: H7*3(296), X8*4=O(71), O(113)."""
    return "%s(%d)" % (check["labels"].get(i) or check["symbols"][i], i)


def problem_lines(check):
    """One line per problem with the limit it broke, e.g.
    'end:   extra    H5(9)-O(113)  0.860 A  (limit 1.164 A)'."""
    return ["%-6s %-8s %s-%s  %.3f A  (limit %.3f A)"
            % (test + ":", kind, atom_name(check, i), atom_name(check, j), r, limit)
            for test in ("start", "end")
            for kind, i, j, r, limit in check["problems"].get(test, [])]


# ---- alone: print every TS ---------------------------------------------------------
def main():
    folders = dft_sella_dirs(DFT_TS_DIR)
    if not folders:
        sys.exit("no dft_sella/ folder under %s" % DFT_TS_DIR)
    counts = {}
    for folder in folders:
        info = load_info(folder)
        check = check_ts(folder, DFT_TS_DIR)
        counts[check["status"]] = counts.get(check["status"], 0) + 1
        print("R%d  %-14s %-50s graph %s"
              % (info["where"]["reaction_index"],
                 " · ".join("O%d" % o for o in info["site"]["oxygens"]), check["id"],
                 check["status"]))
        for line in problem_lines(check):
            print("    " + line)
    print("\n%d TSs: %s" % (len(folders), ", ".join("%s %d" % (status, counts[status])
                                                   for status in ("ok", "not ok", "no start",
                                                                  "no result")
                                                   if status in counts)))
    print("bonded r < %.2f x (r_i + r_j); forming/breaking r < %.2f x (r_i + r_j) + %.2f A"
          % (DFT_SELLA_BOND_CUTOFF, DFT_SELLA_BOND_CUTOFF, DFT_SELLA_TS_BOND_TOLERANCE))


if __name__ == "__main__":
    main()