# pyntaz — transition states in zeolite pores, from reaction graph to DFT

A Pynta-style workflow adapted to Brønsted-acid zeolites. From an IZA
framework and a set of reactions written as RMG adjacency lists, pyntaz

1. builds the bare framework with one or two Al and finds the acid-site oxygens,
2. places every species on every site and relaxes it with MACE,
3. keeps the relaxed configurations that held together and deduplicates them,
4. builds transition-state guesses straight from each reaction's graph,
   seated on pairs of framework oxygens, and deduplicates them,
5. pre-relaxes every guess with MACE plus harmonic springs on the reacting
   bonds, toward the TS and toward its initial and final states,
6. searches for the saddle point with Sella and checks it with frequencies,
7. collects initial state, TS and final state of every guess for NEB.

Two side stages close the loop: `dft/` runs VASP single points on frames
from the Sella searches, and `mace_ft/` fine-tunes the MACE foundation model
on them; the pipeline then reruns steps 7–8 with the fine-tuned model.

The pipeline and the DFT single points run on CPU nodes; MACE fine-tuning
runs on a GPU. All jobs are submitted through SLURM.

---

## Installation

Tested on Hive (UC Davis), Python 3.9. All commands run from the repo root.

### 1. Code and environment

```bash
git clone https://github.com/yhiwang/pynta-zeolite.git
cd pynta-zeolite

module load conda                        # on Hive
conda env create -f environment.yml      # creates pyntaz-slim
conda activate pyntaz-slim
pip install -e .                         # pyntaz itself, editable
```

The versions in `environment.yml` are pinned on purpose:

| package | pin | why |
|---|---|---|
| numpy | 1.26.4 | rmgmolecule breaks on numpy 2 |
| cython | 0.29 | rmgmolecule needs cython < 3 |
| ase | 3.26.0 | the version maze, MACE and Sella were tested with |
| torch | 2.8.0+cpu | CPU build; the default PyPI torch is the multi-GB CUDA build |
| matscipy | 1.1.1 | 1.2.0 requires numpy >= 2 |
| sella | 2.5.0 | 2.6.0 has no Python 3.9 wheel |
| setuptools | < 81 | maze imports `pkg_resources`, which setuptools 81 removes |

Don't `pip install` into this environment without pinning versions. An
unpinned install can quietly upgrade numpy or ASE and break rmgmolecule or
maze. Run `pip install --dry-run ...` first and read the "Would install" line.

### 2. MACE model

Compute nodes have no internet, so the model has to be a local file in
`models/` (gitignored). Start from the foundation model:

```bash
mkdir -p models
wget -P models https://github.com/ACEsuit/mace-foundations/releases/download/mace_mpa_0/mace-mpa-0-medium.model
export PYNTAZ_MODEL=$PWD/models/mace-mpa-0-medium.model
```

`scripts/settings.py` currently defaults to the fine-tuned model
`models/mace-mpa0-ft-MOR_T4-e1f10.model` (mpa-0 fine-tuned on MOR_T4
RPBE-D3(BJ) single points, see [MACE fine-tuning](#mace-fine-tuning-mace_ft)).
That file is not in the repo, so on a fresh clone set `PYNTAZ_MODEL` as
above, or produce your own with `dft/` and `mace_ft/`.

### 3. Check

```bash
python -c "import importlib.metadata, sella, ase, numpy, torch; from mace.calculators import mace_mp; print('sella', importlib.metadata.version('sella'), '| ase', ase.__version__, '| numpy', numpy.__version__, '| torch', torch.__version__)"
python -c "from molecule.molecule import Molecule; print('rmgmolecule ok')"
python -c "from maze.zeolite import Zeolite; print('maze ok,', len(Zeolite.make('MOR')), 'atoms in MOR')"
```

Expected: `sella 2.5.0 | ase 3.26.0 | numpy 1.26.4 | torch 2.8.0+cpu`,
`rmgmolecule ok`, `maze ok, 144 atoms in MOR`. Two warnings are harmless:
`pkg_resources is deprecated` (from maze) and `crystal system 'orthorhombic'
is not interpreted ...` (from ASE reading the cif).

### 4. Optional: DFT and fine-tuning

- **VASP** (`dft/`): `dft/dft_settings.py` sets `VASP_MODULE`
  (`vasp/6.5.1%nvhpc@24.9` on Hive, run on CPU nodes with `vasp_gam`) and
  `VASP_PP_PATH` (default `~/vasp_pp`, which must hold `potpaw_PBE/`).
- **MACE fine-tuning** (`mace_ft/`): a separate GPU environment, `mace-ft`
  (tested with PyTorch 2.11 cu128 and mace-torch 0.3.16).
  `ft_settings.FT_MACE_RUN_TRAIN` points to its `mace_run_train`. The
  Materials Project replay file is read from `~/.cache/mace/`; download it
  once on the login node.

### Troubleshooting

- **Imports pick up the wrong versions.** Check `echo $PYTHONPATH`. A
  leftover entry from an old environment or a pynta checkout wins over the
  env; `unset PYTHONPATH` before activating.
- **`conda activate` does nothing in a SLURM job.** Jobs call the env's
  python directly: `settings.PYTHON`, default
  `~/.conda/envs/pyntaz-slim/bin/python`, override with `PYNTAZ_PYTHON`.
- **MACE tries to download a model on a compute node.** It was given a model
  name instead of a file path; see step 2.
- **`FileNotFoundError` for the model.** The default fine-tuned model is not
  in the repo; set `PYNTAZ_MODEL` (step 2).
- **A job was submitted but never finished.** Steps 7, 8 and `dft/` write
  `job.id` and skip folders that have one. Check `job.err`, then delete
  `job.id` (or the folder) to resubmit.
- **`import openmm` fails.** Only `openmm/02_openmm_restrain.py` needs it.

---

## Repository layout

```
pynta-zeolite/
├── README.md, environment.yml, pyproject.toml
├── reaction.yaml         the reaction set (RMG adjacency lists, X = framework oxygen site)
├── data/                 MOR.cif, MFI.cif; maze reads ./data/<CODE>.cif, else downloads from IZA
├── models/               MACE model files (gitignored)
├── pyntaz/               the library: chemistry only, no paths, nothing reads sys.argv
│   ├── framework.py      build_framework(): supercell, Al substitution, sites and site pairs
│   ├── adjlist.py        3D geometry from an adjacency list: rings, torsions, X sites
│   ├── reactions.py      reaction.yaml -> RMG molecules
│   ├── pynta_mol.py      get_adsorbate / get_name, vendored from pynta
│   ├── placement.py      monodentate and bidentate orientation sweeps
│   ├── adsorbates.py     every species on every site                  (step 1)
│   ├── relax.py          MACE relaxation, framework frozen             (step 2)
│   ├── filtering.py      bond-survival test and RMSD clustering        (steps 3, 6)
│   ├── ts_graph.py       TS guesses from the merged reaction graph     (step 5)
│   ├── harmonic_relax.py staged MACE relax with springs; TS, initial, final  (step 7)
│   ├── ts_sella.py       Sella saddle search, frequency check, verdict (step 8)
│   └── geometry.py       neighbor lists, framework/adsorbate split, clashes, RMSD
├── scripts/              one script per step, run in order
│   ├── settings.py       EVERY setting for a run: framework, sites, thresholds, model, SLURM
│   ├── layout.py         the only file that knows the run-directory layout
│   ├── 00_build_framework.py ... 09_collect_neb.py
│   └── workers and helpers: relax_one.py, harmonic_one.py, ts_sella_one.py
│                (run inside SLURM jobs), place_one_adsorbate.py,
│                rotate_chain.py, demo_framework.py
├── analysis/             plots and tables from a run; read-only, safe to rerun
│   ├── plot_relax_sweep.py       energy vs orientation after step 2
│   ├── plot_pair_counts.py       site x site TS guess counts after step 6
│   ├── summarize_endpoints.py    endpoint_summary.csv + .png from step 7
│   ├── summarize_ts.py           ts_summary.csv + .png from step 8
│   ├── plot_mace_dft_energy.py   MACE vs DFT energies for DFT_sp/
│   ├── compare_runs.py           diff two run dirs after a code change
│   └── plotting.py, ts_summary.py, endpoint_summary.py   code the scripts above use
├── dft/                  VASP single points -> MACE training set; settings in dft_settings.py
├── mace_ft/              MACE fine-tuning on that set; settings in ft_settings.py
├── openmm/               the test bed steps 7-8 grew out of, kept for reference
├── tests/                standalone checks, print PASS/FAIL
└── archive/              the old endpoint-pair TS route, kept for reference
```

---

## Running the main workflow

Edit `scripts/settings.py`, then run the steps in order from the repo root.
The scripts take no command-line flags; `settings.py` is the one place a
run is defined (`dft/` and `mace_ft/` read `CODE`, `SITES` and the run
directory from it too).

The key settings:

- `CODE`: the framework, e.g. `"MOR"`.
- `SITES`: one T label (`"T4"`) for a single Al, or a pair name from the
  table step 0 prints (`"T2-T4_5.65"`) for two Al.
- `SITE_RULE`: with two Al, `"cross"` keeps only O pairs with one O on each
  Al; `"all"` keeps every first-shell O and pair. It is saved with the
  framework, so don't change it inside an existing run directory.
- `MACE_MODEL`: the model file for steps 2, 7 and 8 (`PYNTAZ_MODEL`).
- `SUBMIT`: `False` writes every job folder and `job.sh` but submits
  nothing (dry run).

Output goes to `runs/<CODE>_<SITES>/`.

```bash
# framework and adsorbates
python scripts/00_build_framework.py     # bare framework + sites
python scripts/01_build_adsorbates.py    # every species on every site
python scripts/02_submit_relax.py        # one MACE job per config
python scripts/03_filter_relax.py        # after the jobs finish: survivors, unique minima
python analysis/plot_relax_sweep.py      # optional: energy vs orientation

# transition states
python scripts/05_build_ts_guesses.py    # TS guesses from reaction.yaml, needs only step 0
python scripts/06_filter_ts_guesses.py   # distinct TS guesses
python analysis/plot_pair_counts.py      # optional: site x site guess-count table
python scripts/07_submit_harmonic.py     # harmonic MACE relax: ts, initial, final per guess
python analysis/summarize_endpoints.py   # which initial/final states survived (--reasons: why not)
python scripts/08_submit_ts_sella.py     # Sella from the harmonic and the raw start
python analysis/summarize_ts.py          # which searches found the TS (--reasons: why not)
python scripts/09_collect_neb.py         # IS / TS / FS per guess into NEB/, barrier table
```

Steps 5–9 don't need steps 1–4: TS guesses are built from the reaction
graph and the framework alone.

Steps 7 and 8 submit one job per guess (and per state or start). Rerun them
once jobs finish: finished folders are skipped, folders with a running job
(`job.id`) are left alone, and step 8 picks up harmonic starts that were
still waiting.

- **Step 7** starts every state from the same raw step 5 guess, with springs
  toward (r_i + r_j) × `mult` (`TS_HARMONIC_MODES`): `ts` pulls every forming
  and breaking bond; `initial` holds only the breaking bonds at bonded
  length, then relaxes spring-free and checks the bonds (`endpoint.json`);
  `final` does the reverse. The relax alternates framework, spectator and
  reacting-atom stages until the energy settles.
- **Step 8** runs Sella (order 1) from each start in `TS_SELLA_STARTS`, then a
  finite-difference frequency check. `result.json` gives one verdict: `ts`,
  `not_converged`, `minimum`, `higher_order` or `wrong_mode` (the one
  imaginary mode does not stretch a reacting bond).
- **Step 9** collects a guess only when both endpoints are ok and at least one
  start gave `ts` (the lower-energy one if both did). The barriers in
  `barriers.csv` are MACE numbers, guesses until NEB / DFT confirm the TS.

### Run directory

```
runs/<CODE>_<SITES>/
├── bare.xyz, framework.json, <CODE>_pairs.json                    step 0
├── Adsorbates/<species>/<site>/<stem>/<stem>_init.xyz             step 1  (+ info.json per species)
├── Adsorbates_relax/<species>/<site>/<stem>/relax.{xyz,traj,log}  step 2
├── Adsorbates_relax_filtered/<species>/<site>/<stem>.xyz          step 3
├── Adsorbates_relax_unique/<species>/<site>/<stem>.xyz            step 3
├── <species>_sweep.png                                            plot_relax_sweep.py
├── TS_guesses/<i>_rxn/pair_<k>/<stem>/<stem>_init.xyz             step 5  (+ info.json, sweep.traj)
├── TS_unique/<i>_rxn/pair_<k>/<stem>/<stem>_init.xyz              step 6  (+ info.json, pair_counts.png)
├── TS_harmonic/<i>_rxn/pair_<k>/<stem>/                           step 7
│   ├── ts_bonds.json, ts_harmonic.{xyz,traj,log}                          ts state
│   ├── initial/initial.{xyz,traj,log}, endpoint.json                      initial state
│   └── final/final.{xyz,traj,log}, endpoint.json                          final state
├── TS_sella/<i>_rxn/pair_<k>/<stem>/<start>/                      step 8  (<start> = harmonic, raw)
│   └── start.xyz, ts_sella.xyz, sella.{traj,log}, vib.0.traj, result.json
├── NEB/<i>_rxn/pair_<k>/<stem>/{initial,ts,final}.xyz, neb.json   step 9  (+ barriers.csv, .png)
├── DFT_sp/<i>_rxn/pair_<k>/<stem>/<start>/f<frame>/               dft/ 01-02
├── DFT_sp/training/{train,valid,test}.extxyz, split.csv           dft/ 03
├── DFT_atoms/<element>/                                           dft/ 04
└── MACE_ft/<FT_NAME>/                                             mace_ft/ 01
```

`<site>` is the zero-padded index into the framework's site list (`0/gas`
for a gas-phase molecule). `<stem>` names the orientation:
`degrees_045` (monodentate), `flip0_phi105_psi240` (bidentate),
`flip0_tor045-120_ax1_roll090` (TS guess). The summaries land next to the
data they describe: `TS_harmonic/endpoint_summary.{csv,png}`,
`TS_sella/ts_summary.{csv,png}`.

---

## DFT single points (`dft/`)

Settings live in `dft/dft_settings.py`: RPBE-D3(BJ), 400 eV, Γ-point, with
PBE POTCARs. Every script has a dry-run switch (`COLLECT_WRITE`,
`SP_SUBMIT`, `ATOM_SUBMIT`).

```bash
python dft/01_collect_configs.py       # frames from every finished Sella run -> DFT_sp/
python dft/02_submit_sp.py             # one VASP single point per frame (SP_LANES running at once)
python dft/04_isolated_atoms.py        # isolated-atom energies, the E0s MACE needs
python dft/03_build_training_set.py    # finished OUTCARs -> train / valid / test extxyz
python analysis/plot_mace_dft_energy.py   # optional: how far MACE is from DFT
```

- `01` keeps the end points of each `sella.traj` plus frames that moved at
  least `COLLECT_MIN_DISP` (at most `COLLECT_MAX_FRAMES` per run), from every
  verdict: failed searches are training data too. MACE energies and forces
  are stored as `mace_energy` / `mace_forces`, never as the reference labels.
- `02` and `04` report each folder as done, in the queue, failed or
  submitted; rerun them to check progress.
- `03` writes `REF_energy` / `REF_forces`, drops unconverged SCFs and
  broken geometries, and splits by config or by whole Sella run
  (`TRAIN_SPLIT_BY`), within each reaction. Run it after `04` has finished
  so the isolated atoms go into `train.extxyz`; it is safe to rerun as more
  single points finish.

---

## MACE fine-tuning (`mace_ft/`)

```bash
python mace_ft/01_train.py             # FT_SUBMIT = False: config.yaml + job.sh only
```

Multihead replay fine-tuning of `settings.MACE_MODEL` on
`DFT_sp/training/`: the DFT data becomes its own head and
`FT_NUM_SAMPLES_PT` Materials Project structures are replayed alongside, so
the model keeps its general chemistry. One GPU job on
`FT_SLURM_PARTITION`; everything lands in `runs/<run>/MACE_ft/<FT_NAME>/`,
including `*_cpu.model` files for the CPU pipeline. A new `FT_NAME` gives a
second run side by side.

To use a fine-tuned model, copy its CPU file into `models/`, point
`MACE_MODEL` (or `PYNTAZ_MODEL`) at it and rerun steps 7–8. Rename the old
`TS_harmonic/` and `TS_sella/` first (e.g. `TS_harmonic_mpa0/`) to keep the
earlier results.

---

## reaction.yaml

Each reaction gives reactant and product as RMG adjacency lists over the
same labelled atoms. `X` atoms are framework oxygen sites, bonded to
whatever sits on that oxygen (the acid proton, an alkoxide carbon). Starred labels
(`*1`, `*2`, ...) mark the atoms that change bonds.

```yaml
- index: 0
  reactant: |
    1 *1 C u0 p0 c0 {2,D} {3,S} {4,S}
    ...
    7 *3 H u0 p0 c0 {8,S}
    8 *4 X u0 p0 c0 {7,S}
    9 *5 X u0 p0 c0
  product: |
    ...
  reaction: C=C + [H][Pt] <=> CC[Pt]
  reaction_family: Surface_Protonation
```

The current set follows ethylene through dimerization, β-H elimination,
reprotonation and methyl shift (reactions 0–6).

---

## `openmm/` (reference)

The self-contained test bed where the TS refinement was developed before it
moved into steps 7–8 (`pyntaz/harmonic_relax.py`, `pyntaz/ts_sella.py`).
It builds one guess per reaction on one O pair and runs the staged MACE
relax and Sella on it, with settings at the top of each script.
`02_openmm_restrain.py` is the earlier spring-only relax in OpenMM. Use the
main pipeline for real runs.

---

## Tests

Each test runs as a plain script and prints PASS/FAIL:

```bash
python tests/test_adjlist.py
python tests/test_framework_sites.py
python tests/test_beta_h_elim.py
python tests/test_methyl_shift.py
```

To check that a code change didn't alter any numbers, run the old and new
code into two run directories and compare them file by file:

```bash
python analysis/compare_runs.py runs/old runs/new
```

---

## Machine settings

Machine-specific values in `scripts/settings.py` can be overridden without
editing, through environment variables:

| variable | default |
|---|---|
| `PYNTAZ_PYTHON` | `~/.conda/envs/pyntaz-slim/bin/python` |
| `PYNTAZ_MODEL` | `models/mace-mpa0-ft-MOR_T4-e1f10.model` |
| `PYNTAZ_SLURM_ACCOUNT` | `ark245grp` |
| `PYNTAZ_SLURM_PARTITION` | `high` |

The DFT and fine-tuning stages keep their machine values (VASP module,
POTCAR path, GPU partition, `mace_run_train` path) in `dft/dft_settings.py`
and `mace_ft/ft_settings.py`.