#!/usr/bin/env python
"""Compare two run directories file by file (used to check that a code change
did not alter the numbers).

    python scripts/compare_runs.py old_run new_run [--tol 1e-6]

Every *.xyz present in both trees is compared on chemical symbols, cell and
positions; *.json files are compared as parsed objects. Files present in only
one tree are listed.
"""

import argparse
import json
import os

import numpy as np
from ase.io import read


def tree_files(root):
    return sorted(os.path.relpath(os.path.join(dirpath, name), root)
                  for dirpath, _, names in os.walk(root) for name in names)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("old")
    parser.add_argument("new")
    parser.add_argument("--tol", type=float, default=1e-6, help="A")
    args = parser.parse_args()

    old_files, new_files = set(tree_files(args.old)), set(tree_files(args.new))
    for relative in sorted(old_files - new_files):
        print("only in %s: %s" % (args.old, relative))
    for relative in sorted(new_files - old_files):
        print("only in %s: %s" % (args.new, relative))

    n_xyz, n_json, worst, differing = 0, 0, 0.0, []
    for relative in sorted(old_files & new_files):
        old_path, new_path = os.path.join(args.old, relative), os.path.join(args.new, relative)
        if relative.endswith(".xyz"):
            old_atoms, new_atoms = read(old_path), read(new_path)
            n_xyz += 1
            if (old_atoms.get_chemical_symbols() != new_atoms.get_chemical_symbols()
                    or not np.allclose(old_atoms.cell, new_atoms.cell)):
                differing.append(relative)
                continue
            deviation = float(np.abs(old_atoms.positions - new_atoms.positions).max())
            worst = max(worst, deviation)
            if deviation > args.tol:
                differing.append(relative)
        elif relative.endswith(".json"):
            n_json += 1
            with open(old_path) as old_handle, open(new_path) as new_handle:
                if json.load(old_handle) != json.load(new_handle):
                    differing.append(relative)

    for relative in differing:
        print("differs: %s" % relative)
    print("%d xyz and %d json compared, largest position deviation %.2e A, %d differ"
          % (n_xyz, n_json, worst, len(differing)))


if __name__ == "__main__":
    main()
