"""pyntaz -- adsorbates and transition-state guesses in zeolite pores.

Module map, in workflow order:

    config       paths, run-directory layout, thresholds
    geometry     neighbor lists, framework/adsorbate split, clashes, RMSD, Kabsch
    framework    build_framework(code, t_site) -> ZeoliteFramework with sites
    reactions    load_reaction_set(yaml) -> species, adjacency lists, reactions
    placement    place_monodentate / place_bidentate orientation sweeps
    adsorbates   write initial guesses for every species on every site   (step 1)
    runtree      walk / read the run directory
    relax        MACE relaxation of one structure, SLURM job text        (step 2)
    filtering    bond-survival test and RMSD deduplication               (step 3)
    plotting     energy-vs-orientation figures                           (step 4)
    ts_pairs     pair reactant/product minima into TS endpoint folders   (step 5)
    ts_guess     construct the TS guess for one endpoint pair            (step 6)

Nothing is imported here on purpose: ``relax`` must be importable on a
compute node without maze or RMG installed, so import the module you need
(``from pyntaz import framework``).
"""

__version__ = "0.2.0"
