#!/usr/bin/env python
"""
Build test_run/ts_guesses/<index>_rxn/ with one directory per endpoint pair.

    python TS_generation.py

Per reaction: info.json with the reaction record and a pairs manifest, then
one pair_NNN/ directory holding the two endpoint configs copied from
Adsorbates_relax_unique as initial.xyz and final.xyz, plus gas.xyz when a
side has a gas species.

Pair directories are indexed rather than named after their endpoints: an
endpoint stem is degrees_045 for a monodentate config and
flip0_phi105_psi240 for a bidentate one, so a composite name would be both
long and unparseable. info.json carries the mapping.

Site pairing follows the X count in the reactant template: one X pairs the
same site only, two X pair different sites only.
"""

import os
import json
import shutil

from molecule.molecule import Molecule

from generate_mol_dict import load_reactions, generate_mol_dict

UNIQUE_DIR = os.path.join("test_run", "Adsorbates_relax_unique")
TS_DIR = os.path.join("test_run", "ts_guesses")
RXNS_FILE = "reaction.yaml"


def find_species_configs(name):
    """[(site, stem, path)] for one species, from the unique tree.

    The stem is opaque -- whatever placement wrote. os.listdir, not glob:
    names like C[CH2][Pt] contain brackets, which glob reads as character
    classes."""
    species_dir = os.path.join(UNIQUE_DIR, name)
    if not os.path.isdir(species_dir):
        raise SystemExit("no configs for %s -- expected %s"
                         % (name, species_dir))
    configs = []
    for site in sorted(os.listdir(species_dir)):
        site_path = os.path.join(species_dir, site)
        if not (site.isdigit() and os.path.isdir(site_path)):
            continue
        for fname in sorted(os.listdir(site_path)):
            if fname.endswith(".xyz"):
                configs.append((site, os.path.splitext(fname)[0],
                                os.path.join(site_path, fname)))
    if not configs:
        raise SystemExit("no configs for %s in %s" % (name, species_dir))
    return configs


def split_side(names, name_to_adjlist_dict):
    """((surface_name, its configs), [gas config paths]) for one side.
    Surface means the molecule has an X atom; exactly one per side."""
    surface, gas = [], []
    for name in names:
        mol = Molecule().from_adjacency_list(name_to_adjlist_dict[name])
        if any(a.is_surface_site() for a in mol.atoms):
            surface.append((name, find_species_configs(name)))
        else:
            gas.append(find_species_configs(name)[0][2])
    if len(surface) != 1:
        raise SystemExit("side %s has %d surface species, need exactly 1"
                         % (names, len(surface)))
    return surface[0], gas


rxns_list, spcs_list = load_reactions(RXNS_FILE)
mol_dict, name_to_adjlist_dict, rxns_list = generate_mol_dict(
    rxns_list, spcs_list, calculate_thermodynamic_references=False)

for r in rxns_list:
    ts_dir = os.path.join(TS_DIR, "%d_rxn" % r["index"])
    os.makedirs(ts_dir, exist_ok=True)

    reactant = Molecule().from_adjacency_list(r["reactant"])
    n_x = sum(1 for a in reactant.atoms if a.is_surface_site())

    (ini_name, ini_configs), ini_gas = split_side(r["reactant_names"],
                                                  name_to_adjlist_dict)
    (fin_name, fin_configs), fin_gas = split_side(r["product_names"],
                                                  name_to_adjlist_dict)
    gas_paths = ini_gas + fin_gas
    if len(gas_paths) > 1:
        raise SystemExit("[%d] %d gas species, expected at most 1"
                         % (r["index"], len(gas_paths)))

    print("\n[%d] %s   (%d X -> %s-site pairing)"
          % (r["index"], r["reaction"], n_x,
             "same" if n_x == 1 else "different"))
    print("  initial: %-20s %d configs" % (ini_name, len(ini_configs)))
    print("  final:   %-20s %d configs" % (fin_name, len(fin_configs)))

    pairs = {}
    for site_i, stem_i, path_i in ini_configs:
        for site_f, stem_f, path_f in fin_configs:
            if n_x == 1 and site_i != site_f:
                continue
            if n_x == 2 and site_i == site_f:
                continue

            key = "pair_%04d" % len(pairs)
            pair_dir = os.path.join(ts_dir, key)
            os.makedirs(pair_dir, exist_ok=True)
            shutil.copy2(path_i, os.path.join(pair_dir, "initial.xyz"))
            shutil.copy2(path_f, os.path.join(pair_dir, "final.xyz"))
            if gas_paths:
                shutil.copy2(gas_paths[0], os.path.join(pair_dir, "gas.xyz"))

            pairs[key] = {"initial": {"species": ini_name,
                                      "site": site_i, "stem": stem_i},
                          "final": {"species": fin_name,
                                    "site": site_f, "stem": stem_f}}

    record = dict(r)
    record["pairs"] = pairs
    with open(os.path.join(ts_dir, "info.json"), "w") as f:
        json.dump(record, f, indent=2)

    print("  %d endpoint pairs" % len(pairs))