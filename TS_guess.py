import os
import json

from ase.io import read

from functions_v2 import (find_framework_indices, extra_framework_indices,
                          nearest_framework)

RXN_DIR = os.path.join("test_run", "ts_guesses", "0_rxn")
UNIQUE_DIR = os.path.join("test_run", "Adsorbates_relax_unique")
ONLY = "pair_0000"          # e.g. "pair_0000"


class RelaxedConfig:
    """One relaxed endpoint, with the placement facts geometry cannot give.

    mol._combine appends the adsorbate after the framework and nothing
    downstream reorders, so adsorbate atom i of the standalone molecule is
    nslab + i here. That offset is the whole reason info.json indices are
    usable against the xyz.
    """

    def __init__(self, path, species, site, stem):
        self.path = path
        self.species = species
        self.site = site
        self.stem = stem
        self.atoms = read(path)

        with open(os.path.join(UNIQUE_DIR, species, "info.json")) as f:
            info = json.load(f)
        self.nslab = info["nslab"]
        self.tag = info["configs"]["%s/%s" % (site, stem)]
        self.intended = self.tag["site_indices"]

        self.framework = find_framework_indices(self.atoms)
        self.adsorbate = extra_framework_indices(self.atoms)
        # a mismatch means the flood-fill swallowed the adsorbate or lost
        # part of the wall; every index below would be shifted silently
        if len(self.framework) != self.nslab:
            raise ValueError("%s: %d framework atoms by connectivity, "
                             "info.json says nslab=%d"
                             % (path, len(self.framework), self.nslab))

        self.al = next(a.index for a in self.atoms if a.symbol == "Al")
        self.binders = sorted(self.nslab + int(k) for k
                              in info["gratom_to_molecule_surface_atom_map"])

    def anchors(self):
        """[(binder, anchor, distance)] -- the framework atom each binder
        actually ended up on, which need not be the one placement chose."""
        return [(b,) + nearest_framework(self.atoms, b, self.framework)
                for b in self.binders]

    def formula(self):
        return "".join(self.atoms[i].symbol for i in self.adsorbate)


with open(os.path.join(RXN_DIR, "info.json")) as f:
    rxn = json.load(f)

pairs = {ONLY: rxn["pairs"][ONLY]} if ONLY else rxn["pairs"]

# print("[%d] %s   (%s)" % (rxn["index"], rxn["reaction"], rxn["reaction_family"]))
# print("  reactants: %s" % ", ".join(rxn["reactant_names"]))
# print("  products:  %s" % ", ".join(rxn["product_names"]))
# print("  %d pairs" % len(pairs))

for name in sorted(pairs):
    entry = pairs[name]
    pair_dir = os.path.join(RXN_DIR, name)
    print("\n%s" % name)

    for side in ("initial", "final"):
        e = entry[side]
        config = RelaxedConfig(os.path.join(pair_dir, side + ".xyz"),
                               e["species"], e["site"], e["stem"])
        print("  %-8s %-14s site %s/%-18s nslab %d  Al %d  %s"
              % (side, config.species, config.site, config.stem,
                 config.nslab, config.al, config.formula()))
        for binder, anchor, d in config.anchors():
            moved = "" if anchor in config.intended else "   <- not %s" % config.intended
            print("           binder %3d -> anchor %4d  %.2f A%s"
                  % (binder, anchor, d, moved))