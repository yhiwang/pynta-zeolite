# pyntaz — adsorbates and transition-state guesses in zeolite pores

A Pynta-style workflow adapted to Brønsted-acid zeolites. From an IZA
framework and a set of reactions written as RMG adjacency lists, pyntaz

1. builds the bare framework with one or two Al and finds the acid-site oxygens,
2. places every species on every site and relaxes it with MACE,
3. keeps the relaxed configurations that held together and deduplicates them,
4. builds transition-state guesses straight from each reaction's graph,
   seated on pairs of framework oxygens,

and (in `openmm/`, still experimental) refines those guesses and optimizes
them to saddle points with Sella.

Everything runs on CPU. MACE relaxations are submitted as SLURM jobs.

---

## Installation

Tested on Hive (UC Davis), CPU-only, Python 3.9. All commands run from the
repo root.

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

Compute nodes have no internet, so the model has to be a local file:

```bash
mkdir -p models
wget -P models https://github.com/ACEsuit/mace-foundations/releases/download/mace_mpa_0/mace-mpa-0-medium.model
```

`scripts/settings.py` expects `models/mace-mpa-0-medium.model`. Set
`PYNTAZ_MODEL` to use another file. `models/` is gitignored.

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

### Troubleshooting

- **Imports pick up the wrong versions.** Check `echo $PYTHONPATH`. A
  leftover entry from an old environment or a pynta checkout wins over the
  env; `unset PYTHONPATH` before activating.
- **`conda activate` does nothing in a SLURM job.** Jobs call the env's
  python directly: `settings.PYTHON`, default
  `~/.conda/envs/pyntaz-slim/bin/python`, override with `PYNTAZ_PYTHON`.
- **MACE tries to download a model on a compute node.** It was given a model
  name instead of a file path; see step 2.
- **`import openmm` fails.** Only `openmm/02_openmm_restrain.py` needs it.

---

## Repository layout

```
pynta-zeolite/
├── README.md, environment.yml, pyproject.toml
├── reaction.yaml         the reaction set (RMG adjacency lists, X = framework oxygen site)
├── data/                 MOR.cif, MFI.cif; maze reads ./data/<CODE>.cif, else downloads from IZA
├── models/               the MACE model file (gitignored)
├── pyntaz/               the library; no script logic, nothing reads sys.argv
│   ├── framework.py      build_framework(): supercell, Al substitution, sites and site pairs
│   ├── adjlist.py        3D geometry from an adjacency list: rings, torsions, X sites
│   ├── reactions.py      reaction.yaml -> RMG molecules
│   ├── pynta_mol.py      get_adsorbate / get_name, vendored from pynta
│   ├── placement.py      monodentate and bidentate orientation sweeps
│   ├── adsorbates.py     every species on every site                  (step 1)
│   ├── relax.py          MACE relaxation, framework frozen             (step 2)
│   ├── filtering.py      bond-survival test and RMSD clustering        (step 3)
│   ├── plotting.py       energy across the placement sweep             (step 4)
│   ├── ts_graph.py       TS guesses from the merged reaction graph     (step 5)
│   └── geometry.py       neighbor lists, framework/adsorbate split, clashes, RMSD
├── scripts/              one script per step, run in order
│   ├── settings.py       EVERY setting for a run: framework, sites, thresholds, SLURM
│   ├── layout.py         the only file that knows the run-directory layout
│   ├── 00_build_framework.py ... 06_filter_ts_guesses.py
│   └── helpers: relax_one.py (SLURM worker), place_one_adsorbate.py,
│                rotate_chain.py, demo_framework.py, compare_runs.py
├── openmm/               experimental TS refinement, see below
├── tests/                standalone checks, print PASS/FAIL
└── archive/              the old endpoint-pair TS route, kept for reference
```

---

## Running the main workflow

Edit `scripts/settings.py`, then run the steps in order from the repo root.
The scripts take no command-line flags; `settings.py` is the one place a
run is defined.

The key settings:

- `CODE`: the framework, e.g. `"MOR"`.
- `SITES`: one T label (`"T4"`) for a single Al, or a pair name from the
  table step 0 prints (`"T2-T4_5.65"`) for two Al.
- `SITE_RULE`: with two Al, `"cross"` keeps only O pairs with one O on each
  Al; `"all"` keeps every first-shell O and pair. It is saved with the
  framework, so don't change it inside an existing run directory.

Output goes to `runs/<CODE>_<SITES>/`.

```bash
python scripts/00_build_framework.py     # bare framework + sites
python scripts/01_build_adsorbates.py    # every species on every site
python scripts/02_submit_relax.py        # one MACE job per config (SUBMIT=False: dry run)
python scripts/03_filter_relax.py        # after the jobs finish: survivors, unique minima
python scripts/04_plot_relax_sweep.py    # optional: energy vs orientation
python scripts/05_build_ts_guesses.py    # TS guesses from reaction.yaml, needs only step 0
python scripts/06_filter_ts_guesses.py   # distinct TS guesses + site x site count plot
```

Steps 5–6 don't need steps 1–4: TS guesses are built from the reaction
graph and the framework alone.

### Run directory

```
runs/<CODE>_<SITES>/
├── bare.xyz, framework.json, <CODE>_pairs.json                  step 0
├── Adsorbates/<species>/<site>/<stem>/<stem>_init.xyz           step 1  (+ info.json per species)
├── Adsorbates_relax/<species>/<site>/<stem>/relax.{xyz,traj,log}  step 2
├── Adsorbates_relax_filtered/<species>/<site>/<stem>.xyz        step 3
├── Adsorbates_relax_unique/<species>/<site>/<stem>.xyz          step 3
├── <species>_sweep.png                                          step 4
├── TS_guesses/<i>_rxn/pair_<k>/<stem>/<stem>_init.xyz           step 5  (+ info.json, sweep.traj)
└── TS_unique/<i>_rxn/pair_<k>/<stem>/<stem>_init.xyz            step 6  (+ info.json, pair_counts.png)
```

`<site>` is the zero-padded index into the framework's site list (`0/gas`
for a gas-phase molecule). `<stem>` names the orientation:
`degrees_045` (monodentate), `flip0_phi105_psi240` (bidentate),
`flip0_tor045-120_ax1_roll090` (TS guess).

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

## TS refinement (`openmm/`, experimental)

A self-contained test bed for turning TS guesses into saddle points before
it moves into `scripts/`. Settings sit at the top of each script.

```bash
cd openmm
python 01_make_ts_guess.py            # one guess per reaction on one O pair -> guesses/<i>_rxn/
python 03_mace_fixed.py               # staged MACE pre-relax with springs on the reacting bonds
python 04_sella_ts.py guesses/0_rxn   # Sella saddle search from the raw and pre-relaxed guesses
```

- `01` writes `ts_guess.xyz` and `ts_guess.json` (the framework size, the
  seated oxygens and every forming or breaking bond as atom indices).
- `02_openmm_restrain.py` is an earlier spring-only relax in OpenMM, kept
  for reference.
- `03` alternates a spectator relax and a restrained relax of the reacting
  atoms, framework fixed, and writes `ts_guess_mace.xyz`.
- `04` runs Sella (order 1) with MACE from each start, then checks the
  saddle with finite-difference frequencies: how many imaginary modes, and
  how much the first one stretches each reacting bond.

Every script loops over `guesses/*_rxn/`, or only the directories named on
the command line.

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
python scripts/compare_runs.py runs/old runs/new
```

---

## Machine settings

Machine-specific values in `scripts/settings.py` can be overridden without
editing, through environment variables:

| variable | default |
|---|---|
| `PYNTAZ_PYTHON` | `~/.conda/envs/pyntaz-slim/bin/python` |
| `PYNTAZ_MODEL` | `models/mace-mpa-0-medium.model` |
| `PYNTAZ_SLURM_ACCOUNT` | `ark245grp` |
| `PYNTAZ_SLURM_PARTITION` | `high` |