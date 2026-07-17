from zeolite_bare import make_zeolite_bare
from main import PyntaZ

import numpy as np

bare = make_zeolite_bare("MOR", "T4")
bare.analyze_zeolite()
for i, s in enumerate(bare.single_sites_lists):
    print(f"site {i}: oxygen index {s[0]['indices']}, position {s[0]['position']}")
p = PyntaZ(bare, "reaction.yaml")
p.generate_mol_dict()
p.generate_atom_maps()
p.setup_adsorbates("test_run", nprocs=1)

