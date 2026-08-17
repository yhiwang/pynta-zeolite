#!/usr/bin/env python
"""
Plot relaxation energy across the placement sweep, one PNG per species.

    python plot_relax_sweep.py

Reads test_run/Adsorbates_relax/<species>/<site>/<stem>/relax.log and picks
the layout from the stem. Monodentate stems (degrees_045) give energy vs
angle, one column per site. Bidentate stems (flip0_phi105_psi240) give a
phi/psi heatmap, one column per (site pair, flip).

Survivors come from test_run/Adsorbates_relax_filtered/, written by
filter_relax.py -- the presence of <stem>.xyz is the whole test. They are
ringed on the line plots and outlined on the heatmaps.
"""

import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RELAX_DIR = os.path.join("test_run", "Adsorbates_relax")
FILTERED_DIR = os.path.join("test_run", "Adsorbates_relax_filtered")
OUT_DIR = "test_run"
LOG_NAME = "relax.log"

MONO = re.compile(r"^degrees_(\d+)$")
BI = re.compile(r"^flip(\d)_phi(\d+)_psi(\d+)$")
ROW = re.compile(r"^\s*\w+:\s+(\d+)\s+\S+\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s*$")


def parse_relax_log(path):
    """(first_energy, last_energy), or (None, None) if the log has no
    parsable optimizer lines. Handles any optimizer prefix and ignores
    restart headers, blank lines and warnings."""
    energies = []
    with open(path) as f:
        for line in f:
            m = ROW.match(line)
            if m:
                energies.append(float(m.group(2)))
    if not energies:
        return None, None
    return energies[0], energies[-1]


def collect_species(species_dir):
    """({site: {stem: (e_initial, e_relaxed)}}, [skipped]).

    os.listdir, not glob: species names like C[CH2][Pt] contain brackets,
    which glob reads as character classes."""
    data, skipped = {}, []
    for site in sorted(os.listdir(species_dir)):
        site_path = os.path.join(species_dir, site)
        if not (site.isdigit() and os.path.isdir(site_path)):
            continue
        for stem in sorted(os.listdir(site_path)):
            log = os.path.join(site_path, stem, LOG_NAME)
            if not os.path.isfile(log):
                continue
            e0, ef = parse_relax_log(log)
            if e0 is None:
                skipped.append(log)
                continue
            data.setdefault(site, {})[stem] = (e0, ef)
    return data, skipped


def collect_survivors(name):
    """{site: {stem}} that survived filtering. Empty when the filter has not
    run yet, which just leaves the plots unmarked."""
    survivors = {}
    species_dir = os.path.join(FILTERED_DIR, name)
    if not os.path.isdir(species_dir):
        return survivors
    for site in sorted(os.listdir(species_dir)):
        site_path = os.path.join(species_dir, site)
        if not (site.isdigit() and os.path.isdir(site_path)):
            continue
        survivors[site] = {os.path.splitext(f)[0]
                           for f in os.listdir(site_path)
                           if f.endswith(".xyz")}
    return survivors


def plot_mono(name, data, survivors, out_path):
    """Two rows per site: initial vs relaxed, then relaxed zoomed.

    Relaxed energies span ~1 eV against an ~80 eV initial spread, so the
    shared axis in the first row flattens them."""
    sites = sorted(data)
    fig, axes = plt.subplots(2, len(sites), figsize=(4.4 * len(sites), 7),
                             squeeze=False, constrained_layout=True)

    for col, site in enumerate(sites):
        stems = sorted(data[site], key=lambda s: int(MONO.match(s).group(1)))
        surv = survivors.get(site, set())
        x = [int(MONO.match(s).group(1)) for s in stems]
        e0 = [data[site][s][0] for s in stems]
        ef = [data[site][s][1] for s in stems]
        xs = [int(MONO.match(s).group(1)) for s in stems if s in surv]
        es = [data[site][s][1] for s in stems if s in surv]

        ax = axes[0][col]
        ax.plot(x, e0, "o-", ms=3, color="tab:orange", label="initial")
        ax.plot(x, ef, "o-", ms=3, color="tab:blue", label="relaxed")
        ax.plot(xs, es, "o", ms=7, mfc="none", mec="tab:green", mew=1.4,
                label="survived")
        ax.set_title("site %s  --  %d/%d survived" % (site, len(xs), len(x)))
        ax.set_ylabel("energy (eV)" if col == 0 else "")
        ax.legend(fontsize=8)

        ax = axes[1][col]
        ax.plot(x, ef, "o-", ms=3, color="tab:blue")
        ax.plot(xs, es, "o", ms=7, mfc="none", mec="tab:green", mew=1.4)
        emin = min(es) if es else min(ef)
        ax.axhline(emin, ls=":", lw=0.8, color="gray")
        ax.set_ylabel("relaxed energy (eV)" if col == 0 else "")
        ax.set_title("min %.3f eV%s" % (emin, "" if es else "  (no survivors)"),
                     fontsize=9)
        ax.set_xlabel("angle from O->Al direction (deg)")

        for row in (0, 1):
            axes[row][col].set_xticks(range(0, 361, 90))
            axes[row][col].set_xlim(-10, 370)
            axes[row][col].grid(alpha=0.3)

    fig.suptitle(name, fontsize=13)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bi(name, data, survivors, out_path):
    """Relaxed energy over the phi/psi grid, one panel per (site, flip).

    Cells with no config are left blank; survivors get a green outline."""
    panels = []
    for site in sorted(data):
        for flip in ("0", "1"):
            cells = {}
            for stem, (_, ef) in data[site].items():
                m = BI.match(stem)
                if m and m.group(1) == flip:
                    cells[(int(m.group(2)), int(m.group(3)))] = (stem, ef)
            if cells:
                panels.append((site, flip, cells))
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4),
                             squeeze=False, constrained_layout=True)

    energies = [ef for _, _, cells in panels for _, ef in cells.values()]
    vmin, vmax = min(energies), max(energies)

    for col, (site, flip, cells) in enumerate(panels):
        phis = sorted({p for p, _ in cells})
        psis = sorted({q for _, q in cells})
        grid = np.full((len(psis), len(phis)), np.nan)
        for (p, q), (_, ef) in cells.items():
            grid[psis.index(q)][phis.index(p)] = ef

        ax = axes[0][col]
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                       vmin=vmin, vmax=vmax)

        surv = survivors.get(site, set())
        n_surv = 0
        for (p, q), (stem, _) in cells.items():
            if stem in surv:
                n_surv += 1
                ax.add_patch(plt.Rectangle((phis.index(p) - 0.5,
                                            psis.index(q) - 0.5), 1, 1,
                                           fill=False, ec="lime", lw=1.4))

        ax.set_xticks(range(0, len(phis), max(1, len(phis) // 6)))
        ax.set_xticklabels([phis[i] for i in
                            range(0, len(phis), max(1, len(phis) // 6))],
                           fontsize=7)
        ax.set_yticks(range(len(psis)))
        ax.set_yticklabels(psis, fontsize=7)
        ax.set_xlabel("phi (deg)")
        ax.set_ylabel("psi (deg)" if col == 0 else "")
        ax.set_title("site %s  flip %s  --  %d/%d survived"
                     % (site, flip, n_surv, len(cells)), fontsize=9)

    fig.colorbar(im, ax=axes[0][-1], label="relaxed energy (eV)")
    fig.suptitle(name, fontsize=13)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def safe_filename(name):
    """Species names contain =, [, ] -- keep them out of the filename."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


if not os.path.isdir(RELAX_DIR):
    raise SystemExit("%s not found -- run from the repo root" % RELAX_DIR)

for name in sorted(os.listdir(RELAX_DIR)):
    species_dir = os.path.join(RELAX_DIR, name)
    if not os.path.isdir(species_dir):
        continue

    data, skipped = collect_species(species_dir)
    if not data:
        print("%-20s no site logs (gas phase, or not yet run)" % name)
        continue
    survivors = collect_survivors(name)

    stems = [s for site in data for s in data[site]]
    if all(MONO.match(s) for s in stems):
        plot = plot_mono
    elif all(BI.match(s) for s in stems):
        plot = plot_bi
    else:
        print("%-20s mixed or unrecognized stems, skipped" % name)
        continue

    out_path = os.path.join(OUT_DIR, safe_filename(name) + "_sweep.png")
    plot(name, data, survivors, out_path)

    total = sum(len(v) for v in data.values())
    n_surv = sum(len(v) for v in survivors.values())
    print("%-20s %d sites, %d configs, %d survived -> %s"
          % (name, len(data), total, n_surv, out_path))
    for log in skipped:
        print("    unparsable: %s" % log)