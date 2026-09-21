# """Load TS-pair structures into StoredConfig objects for analysis.
# Run with ``python -i scripts/analyze_pair.py`` to keep the objects around."""

# import _common  # noqa: F401  -- puts pyntaz on sys.path
# from pyntaz.ts_guess import StoredConfig

# initial_file = "test_run/ts_guesses/0_rxn/pair_0014/initial.xyz"
# final_file = "test_run/ts_guesses/0_rxn/pair_0014/final.xyz"

# initial = StoredConfig.from_path(initial_file)
# final = StoredConfig.from_path(final_file)

# for stored in (initial, final):
#     print("\n=== %s: %s ===" % (stored.role, stored.species))
#     print("path         ", stored.path)
#     print("atoms        ", stored.atoms)
#     print("n_framework  ", stored.n_framework)
#     print("framework    ", stored.framework[:5], "...", stored.framework[-1])
#     print("adsorbate    ", stored.adsorbate_indices())
#     print("mol_to_ase   ", stored.mol_to_ase)
#     print("adjlist")
#     print(stored.adjlist)

#     # molecule index -> xyz index, one line per RMG atom
#     mol = stored.molecule()
#     for mol_index, atom in enumerate(mol.atoms):
#         xyz_index = stored.mol_to_ase.get(mol_index)        # None for X (no coordinates)
#         print("  mol %-2d %-2s -> xyz %s" % (mol_index, atom.symbol, xyz_index))

# # example: xyz index of molecule atom 1 of the final species, and its position
# xyz_index = final.mol_to_ase[1]
# print("\nfinal: mol atom 1 is xyz atom", xyz_index, "at", final.atoms.positions[xyz_index])

# ------------------------------------------------------------------------
# reaction.yaml side index  <->  xyz index, via the InsertionPlan that step 6 builds
# ------------------------------------------------------------------------
"""Load one TS pair the way step 6 does and look at the index bookkeeping.
Run with ``python -i scripts/analyze_pair.py`` to keep the objects around."""

import _common  # noqa: F401  -- puts pyntaz on sys.path
from pyntaz.ts_guess import load_pair, InsertionPlan

pair_dir = "test_run/ts_guesses/0_rxn/pair_0014"

reaction, initial, final, gas_initial, gas_final = load_pair(pair_dir)

# StoredConfig: .atoms .species .role .adjlist .n_framework .framework .mol_to_ase .molecule()
for stored in (initial, final):
    print("\n=== %s: %s ===" % (stored.role, stored.species))
    print("mol_to_ase ", stored.mol_to_ase)
    print(stored.adjlist)

# reaction.yaml side index -> xyz index on both sides (what map_side_atoms works out)
# the reaction as written in reaction.yaml (whole-side adjacency lists, *N labels)
print("\n=== reaction: %s ===" % reaction["reaction"])
print("reactant side")
print(reaction["reactant"])
print("product side")
print(reaction["product"])

plan = InsertionPlan(reaction, initial, final, gas_initial, gas_final)
plan.print_side_table()

# the maps themselves: {side index: (StoredConfig, mol index, xyz index)}
side_to_xyz_initial = plan.assembled_map if plan.assembled_role == "initial" else plan.canonical_map
side_to_xyz_final = plan.canonical_map if plan.assembled_role == "initial" else plan.assembled_map

print(plan.assembled_map)
print(plan.canonical_map)