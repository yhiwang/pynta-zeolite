# pyntaz — adsorbates and transition-state guesses in zeolite pores

Pynta-style workflow for a Brønsted-acid zeolite: build the framework, place
every species from `reaction.yaml` on every first-shell oxygen of the Al,
relax with MACE on HPC2, keep what survived, pair the minima into TS
endpoints and construct a TS guess for each pair.

```
new/
├── README.md, MIGRATION.md    this file; old-name -> new-name table
├── reaction.yaml              reaction set (RMG adjacency lists, X = framework site)
├── data/MOR.cif               maze reads ./data/<CODE>.cif (falls back to IZA download)
├── models/                    put mace-mpa-0-medium.model here (or set PYNTAZ_MODEL)
├── pyntaz/                    importable library -- no script logic in here
│   ├── config.py     machine paths (PYNTAZ_* env overrides), RunLayout, every threshold
│   ├── geometry.py   neighbor lists, framework/adsorbate split, clashes, clearance, RMSD, Kabsch
│   ├── framework.py  build_framework("MOR", "T4") -> ZeoliteFramework + sites
│   ├── reactions.py  load_reaction_set("reaction.yaml") -> ReactionSet(species, adjlists, reactions)
│   ├── placement.py  place_monodentate / place_bidentate sweeps; orientation stem <-> tag
│   ├── adsorbates.py every species x every site -> Adsorbates/ tree + info.json     (step 1)
│   ├── runtree.py    walk / read / copy the run directory (the only module that knows the layout)
│   ├── relax.py      relax_structure(xyz) with MACE; slurm_job_script(...)          (step 2)
│   ├── filtering.py  bond-survival test + RMSD clustering                            (step 3)
│   ├── plotting.py   energy-vs-orientation figures                                   (step 4)
│   ├── ts_pairs.py   reactant/product minima -> ts_guesses/<i>_rxn/pair_NNNN         (step 5)
│   └── ts_guess.py   graph matching, InsertionPlan, TSGuessBuilder                   (step 6)
└── scripts/          one runnable per workflow step, in order; all take --run-dir and -h
    ├── 01_build_adsorbates.py     --code MOR --t-site T4
    ├── 02_submit_relax.py         dry run; --submit to sbatch
    ├── relax_one.py               SLURM worker, copied into every job directory
    ├── 03_filter_relax.py
    ├── 04_plot_relax_sweep.py     optional
    ├── 05_setup_ts_pairs.py
    ├── 06_build_ts_guesses.py     --rxn 1_rxn --pair pair_0001 for a verbose trace
    ├── place_one_adsorbate.py     try one adjacency list without editing reaction.yaml
    └── compare_runs.py            diff two run directories numerically
```

## Running

From the repo root (`scripts/_common.py` puts the repo on `sys.path`, so no
install is needed):

```bash
python scripts/01_build_adsorbates.py --code MOR --t-site T4      # -> test_run/Adsorbates
python scripts/02_submit_relax.py                                 # dry run: job dirs + job.sh
python scripts/02_submit_relax.py --submit                        # on HPC2
python scripts/03_filter_relax.py                                 # -> *_filtered, *_unique
python scripts/04_plot_relax_sweep.py                             # -> test_run/<species>_sweep.png
python scripts/05_setup_ts_pairs.py                               # -> test_run/ts_guesses
python scripts/06_build_ts_guesses.py                             # -> ts_guess.xyz per pair
python scripts/06_build_ts_guesses.py --rxn 1_rxn --pair pair_0001   # one pair, full trace
```

Every script takes `--run-dir` (default `test_run`). Machine-specific settings
live at the top of `pyntaz/config.py` and can be overridden without editing:
`PYNTAZ_REPO`, `PYNTAZ_PYTHON`, `PYNTAZ_MODEL`, `PYNTAZ_SLURM_ACCOUNT`,
`PYNTAZ_SLURM_PARTITION`. `job.sh` exports `PYNTAZ_REPO` as `PYTHONPATH`,
which is how `relax_one.py` finds `pyntaz` on the compute node.

Dependencies: ase (maze 0.1.1 needs ase <= 3.23), maze-sim, RMG `molecule`,
pynta (for `get_adsorbate` / `get_name`), mace-torch (only for relaxation),
networkx, pyyaml, matplotlib. `pyntaz.relax` / `pyntaz.filtering` import only
ase + numpy, so the relax worker and the filter run without maze or RMG.

## Data layout (unchanged)

```
<run_dir>/Adsorbates/<species>/<site>/<stem>/<stem>_init.xyz     initial guesses
<run_dir>/Adsorbates/<species>/info.json                         index maps, tags per config
<run_dir>/Adsorbates_relax/<species>/<site>/<stem>/relax.{xyz,traj,log}
<run_dir>/Adsorbates_relax_filtered/<species>/<site>/<stem>.xyz  survivors
<run_dir>/Adsorbates_relax_unique/<species>/<site>/<stem>.xyz    deduplicated minima
<run_dir>/ts_guesses/<i>_rxn/info.json                            reaction + pairs manifest
<run_dir>/ts_guesses/<i>_rxn/pair_NNNN/{initial,final,endpoint_*,ts_guess}.xyz
```

`<site>` is the zero-padded index into `framework.mono_sites` (`0` for
gas); `<stem>` is `degrees_045` (monodentate spin angle from the O→Al
direction), `flip0_phi105_psi240` (bidentate) or `gas`.
`placement.orientation_stem` / `parse_orientation_stem` convert between stems
and tags. `info.json` keys are unchanged, so existing run directories keep
working with the new scripts.

## Reading the library

Start with `pyntaz/__init__.py` (module map), then read the modules in
workflow order. Each module docstring says *why*, each function docstring
*what*. Nothing in `pyntaz/` reads `sys.argv` or hard-codes `test_run`;
that is all in `scripts/`. Variable names are spelled out (`bridge_axis`,
`bond_length_o1`, `framework_positions`), single letters only for `i, j`
loop indices.

## Checking a change did not alter the numbers

```bash
python scripts/compare_runs.py old_run new_run
```
compares every `.xyz` (symbols, cell, positions) and `.json` in both trees.
This reorganisation was checked that way against the previous scripts:
197 initial guesses + info.json (step 1, plus a 3456-structure bidentate
sweep), 104 filtered / 25 unique configs (step 3), the three sweep PNGs
(step 4, byte-identical), 118 pair manifests (step 5) and all 118
`ts_guess.xyz` / `endpoint_*.xyz` (step 6) are identical to the old output,
as are the step 3/5/6 logs and the verbose one-pair trace.
