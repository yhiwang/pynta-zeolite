from zeolite_bare import make_zeolite_bare
from main import PyntaZ

bare = make_zeolite_bare("MOR", "T4")
bare.analyze_zeolite()
for i, s in enumerate(bare.single_sites_lists):
    print("site %d: oxygen %s at %s"
          % (i, s[0]["indices"], s[0]["position"]))

p = PyntaZ(bare, "reaction.yaml")
p.setup_adsorbates("test_run")