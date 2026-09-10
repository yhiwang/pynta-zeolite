"""pyntaz -- adsorbates and transition-state guesses in zeolite pores.

The chemistry and the algorithms, and nothing else: every function here
takes and returns in-memory objects (ase Atoms, RMG Molecules, plain dicts)
and knows nothing about run directories or file names. Where things live on
disk, and the settings of a particular run, are the business of the step
scripts in ``scripts/``.

Module map, in workflow order:

    geometry     neighbor lists, framework/adsorbate split, clashes, RMSD, Kabsch
    framework    build_framework(code, t_labels) -> ZeoliteFramework with sites
    reactions    yaml text -> ReactionSet: species, adjacency lists, reactions
    pynta_mol    get_adsorbate / get_name, vendored from pynta
    placement    place_monodentate / place_bidentate orientation sweeps
    adsorbates   species_guesses: every species on every site       (step 1)
    relax        MACE relaxation of one structure, framework frozen  (step 2)
    filtering    bond-survival test and RMSD deduplication           (step 3)
    plotting     energy-vs-orientation figures                       (step 4)
    adjlist      AdjacencyStructure: 3D coordinates from an adjacency list
    ts_graph     TSGraph + PairSweep: TS guesses from the reaction graph (step 5)

Every numeric default lives next to the function that uses it, as a
keyword argument; the step scripts pass their own values from
``scripts/settings.py``.

Nothing is imported here on purpose: ``relax`` must be importable on a
compute node without maze or RMG installed, so import the module you need
(``from pyntaz import framework``).
"""

__version__ = "0.3.0"
