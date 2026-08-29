#!/usr/bin/env python
"""Step 1 -- build the bare zeolite and write every initial adsorbate guess.

    python scripts/01_build_adsorbates.py --code MOR --t-site T4

Reads reaction.yaml, places every species on every first-shell oxygen of the
Al (monodentate) or every pair of them (bidentate), and writes
<run_dir>/Adsorbates/<species>/<site>/<stem>/<stem>_init.xyz plus info.json.
Species folders that already exist are reused untouched.
"""

from _common import step_parser, layout_from, config


def main():
    parser = step_parser(__doc__)
    parser.add_argument("--code", default="MOR", help="IZA framework code")
    parser.add_argument("--t-site", default="T4", help="T-site label to substitute by Al")
    parser.add_argument("--reactions", default=config.DEFAULT_REACTIONS_FILE,
                        help="reaction yaml (default: %(default)s)")
    args = parser.parse_args()

    from pyntaz.framework import build_framework
    from pyntaz.reactions import load_reaction_set
    from pyntaz.adsorbates import write_all_guesses

    framework = build_framework(args.code, args.t_site)
    print("%s %s: %d atoms, Al at %d, %d monodentate sites, %d site pairs"
          % (framework.code, framework.t_label, len(framework.atoms),
             framework.al_index, len(framework.mono_sites), len(framework.bi_sites)))
    framework.describe()

    reaction_set = load_reaction_set(args.reactions)
    print("\nspecies: %s" % ", ".join(reaction_set.species))

    written = write_all_guesses(reaction_set, args.run_dir, framework)
    print("\n%d structures in %s" % (sum(len(paths) for paths in written.values()),
                                     layout_from(args).adsorbates))


if __name__ == "__main__":
    main()
