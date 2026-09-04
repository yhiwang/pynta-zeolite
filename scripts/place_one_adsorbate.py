#!/usr/bin/env python
"""Place one adsorbate, given as an adjacency list, on the bare zeolite --
a quick way to test a single species without editing reaction.yaml.

    python scripts/place_one_adsorbate.py --code MOR --t-site T4 --run-dir test_run
    python scripts/place_one_adsorbate.py --adjlist my_species.txt

Without --adjlist the example below is used (ethylene bridging two framework
oxygens, a bidentate species). No *N labels are needed: those mark the atoms
a reaction family acts on and nothing here reads them.
"""

from _common import step_parser

EXAMPLE_ADJLIST = """
multiplicity 1
1 C u0 p0 c0 {2,S} {3,S} {4,S} {7,S}
2 C u0 p0 c0 {1,S} {5,S} {6,S} {8,S}
3 H u0 p0 c0 {1,S}
4 H u0 p0 c0 {1,S}
5 H u0 p0 c0 {2,S}
6 H u0 p0 c0 {2,S}
7 X u0 p0 c0 {1,S}
8 X u0 p0 c0 {2,S}
"""


def main():
    parser = step_parser(__doc__)
    parser.add_argument("--code", default="MOR")
    parser.add_argument("--t-site", default="T4")
    parser.add_argument("--adjlist", help="file holding an RMG adjacency list")
    args = parser.parse_args()

    from pyntaz.pynta_mol import get_name
    from pyntaz.framework import build_framework
    from pyntaz.reactions import molecule_from_adjlist
    from pyntaz.adsorbates import write_species_guesses

    adjlist = EXAMPLE_ADJLIST if args.adjlist is None else open(args.adjlist).read()
    mol = molecule_from_adjlist(adjlist)
    name = get_name(mol)

    framework = build_framework(args.code, args.t_site)
    n_binders = len(mol.get_adatoms())
    print("%s  %d adatoms -> %s"
          % (name, n_binders, ["gas", "monodentate", "bidentate"][n_binders]))
    print("%d single sites, %d pairs\n"
          % (len(framework.mono_sites), len(framework.bi_sites)))

    xyz_paths = write_species_guesses(mol, name, args.run_dir, framework)
    print("\n%d structures" % len(xyz_paths))


if __name__ == "__main__":
    main()
