"""Step 4 (optional): relaxation energy across the placement sweep, one
figure per species.

Monodentate species give energy vs spin angle, one column per site;
bidentate species give a phi/psi heat map, one panel per (site pair, flip).
Survivors of the filter step are ringed / outlined.

The figure functions take ``energies`` = {site: {stem: (e_initial,
e_relaxed)}} and ``survivors`` = {site: {stem}}; collecting those from a
run directory is the caller's job.
"""

import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .placement import (GAS_STEM, parse_orientation_stem, is_monodentate_stem,
                        is_bidentate_stem)

# "BFGS:   12 15:04:11   -2274.123456   0.0345"
_OPTIMIZER_ROW = re.compile(
    r"^\s*\w+:\s+(\d+)\s+\S+\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s*$")


def parse_optimizer_log(lines):
    """(first energy, last energy) from the lines of an ASE optimizer log,
    or (None, None) if none parse. Any optimizer prefix works; restart
    headers, blank lines and warnings are ignored."""
    energies = []
    for line in lines:
        match = _OPTIMIZER_ROW.match(line)
        if match:
            energies.append(float(match.group(2)))
    if not energies:
        return None, None
    return energies[0], energies[-1]


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def plot_monodentate(name, energies, survivors, out_path):
    """Two rows per site: initial vs relaxed, then relaxed zoomed.

    Relaxed energies span ~1 eV against an ~80 eV initial spread, so the
    shared axis in the first row flattens them."""
    sites = sorted(energies)
    fig, axes = plt.subplots(2, len(sites), figsize=(4.4 * len(sites), 7),
                             squeeze=False, constrained_layout=True)

    def angle_of(stem):
        return parse_orientation_stem(stem)["angle"]

    for col, site in enumerate(sites):
        stems = sorted(energies[site], key=angle_of)
        survived = survivors.get(site, set())
        angles = [angle_of(stem) for stem in stems]
        e_initial = [energies[site][stem][0] for stem in stems]
        e_relaxed = [energies[site][stem][1] for stem in stems]
        angles_survived = [angle_of(stem) for stem in stems if stem in survived]
        e_survived = [energies[site][stem][1] for stem in stems if stem in survived]

        ax = axes[0][col]
        ax.plot(angles, e_initial, "o-", ms=3, color="tab:orange", label="initial")
        ax.plot(angles, e_relaxed, "o-", ms=3, color="tab:blue", label="relaxed")
        ax.plot(angles_survived, e_survived, "o", ms=7, mfc="none",
                mec="tab:green", mew=1.4, label="survived")
        ax.set_title("site %s  --  %d/%d survived"
                     % (site, len(angles_survived), len(angles)))
        ax.set_ylabel("energy (eV)" if col == 0 else "")
        ax.legend(fontsize=8)

        ax = axes[1][col]
        ax.plot(angles, e_relaxed, "o-", ms=3, color="tab:blue")
        ax.plot(angles_survived, e_survived, "o", ms=7, mfc="none",
                mec="tab:green", mew=1.4)
        e_min = min(e_survived) if e_survived else min(e_relaxed)
        ax.axhline(e_min, ls=":", lw=0.8, color="gray")
        ax.set_ylabel("relaxed energy (eV)" if col == 0 else "")
        ax.set_title("min %.3f eV%s"
                     % (e_min, "" if e_survived else "  (no survivors)"), fontsize=9)
        ax.set_xlabel("angle from O->Al direction (deg)")

        for row in (0, 1):
            axes[row][col].set_xticks(range(0, 361, 90))
            axes[row][col].set_xlim(-10, 370)
            axes[row][col].grid(alpha=0.3)

    fig.suptitle(name, fontsize=13)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bidentate(name, energies, survivors, out_path):
    """Relaxed energy over the phi/psi grid, one panel per (site, flip).
    Cells with no config are left blank; survivors get a green outline."""
    panels = []
    for site in sorted(energies):
        for flip in (0, 1):
            cells = {}
            for stem, (_, e_relaxed) in energies[site].items():
                tag = parse_orientation_stem(stem)
                if tag and "flip" in tag and tag["flip"] == flip:
                    cells[(tag["phi"], tag["psi"])] = (stem, e_relaxed)
            if cells:
                panels.append((site, flip, cells))
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4),
                             squeeze=False, constrained_layout=True)

    all_energies = [e for _, _, cells in panels for _, e in cells.values()]
    vmin, vmax = min(all_energies), max(all_energies)

    for col, (site, flip, cells) in enumerate(panels):
        phis = sorted({phi for phi, _ in cells})
        psis = sorted({psi for _, psi in cells})
        grid = np.full((len(psis), len(phis)), np.nan)
        for (phi, psi), (_, e_relaxed) in cells.items():
            grid[psis.index(psi)][phis.index(phi)] = e_relaxed

        ax = axes[0][col]
        image = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                          vmin=vmin, vmax=vmax)

        survived = survivors.get(site, set())
        n_survived = 0
        for (phi, psi), (stem, _) in cells.items():
            if stem in survived:
                n_survived += 1
                ax.add_patch(plt.Rectangle((phis.index(phi) - 0.5,
                                            psis.index(psi) - 0.5), 1, 1,
                                           fill=False, ec="lime", lw=1.4))

        step = max(1, len(phis) // 6)
        ax.set_xticks(range(0, len(phis), step))
        ax.set_xticklabels([phis[i] for i in range(0, len(phis), step)], fontsize=7)
        ax.set_yticks(range(len(psis)))
        ax.set_yticklabels(psis, fontsize=7)
        ax.set_xlabel("phi (deg)")
        ax.set_ylabel("psi (deg)" if col == 0 else "")
        ax.set_title("site %s  flip %d  --  %d/%d survived"
                     % (site, flip, n_survived, len(cells)), fontsize=9)

    fig.colorbar(image, ax=axes[0][-1], label="relaxed energy (eV)")
    fig.suptitle(name, fontsize=13)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_species_sweep(name, energies, survivors, out_path):
    """Pick the monodentate or bidentate figure from the stems. Returns the
    figure kind ("mono" / "bi"), "gas" when there is nothing to plot, or
    None when the stems are mixed."""
    stems = [stem for site in energies for stem in energies[site]]
    if all(stem == GAS_STEM for stem in stems):
        return "gas"
    if all(is_monodentate_stem(stem) for stem in stems):
        plot_monodentate(name, energies, survivors, out_path)
        return "mono"
    if all(is_bidentate_stem(stem) for stem in stems):
        plot_bidentate(name, energies, survivors, out_path)
        return "bi"
    return None


def plot_pair_counts(title, sites, counts, out_path, row_label="rows: X14 on",
                     col_label="cols: X15 on"):
    """A site x site table of TS guess counts, one PNG.

    ``sites`` = [(t_label, o_label, atom_index)] in display order, Al by Al;
    ``counts`` = {(row, col): n} over the *directed* pairs that were tried,
    row/col indexing ``sites``; ``n`` is an int, or (kept, collected) once a
    filter has run, shown as "kept/collected" and coloured by kept. Pairs
    absent from ``counts`` were not tried (out of span, or excluded by the
    site rule) and are drawn as X.
    """
    n = len(sites)
    grid = np.full((n, n), np.nan)
    text = [[None] * n for _ in range(n)]
    for (row, col), value in counts.items():
        if isinstance(value, tuple):
            kept, collected = value
            text[row][col] = "%d/%d" % (kept, collected)
            grid[row][col] = kept
        else:
            text[row][col] = "%d" % value
            grid[row][col] = value

    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * n, 1.2 + 0.62 * n))
    vmax = max(1.0, np.nanmax(grid)) if np.isfinite(grid).any() else 1.0
    ax.imshow(grid, cmap=plt.get_cmap("Blues"), vmin=0, vmax=vmax, aspect="equal")

    for row in range(n):
        for col in range(n):
            if text[row][col] is None:
                ax.text(col, row, "X", ha="center", va="center",
                        fontsize=9, color="0.65")
            else:
                dark = grid[row][col] > 0.6 * vmax
                ax.text(col, row, text[row][col], ha="center", va="center",
                        fontsize=9, fontweight="bold" if grid[row][col] > 0 else None,
                        color="white" if dark else "0.15")

    # cell grid, heavier lines between Al blocks
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="0.8", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    boundaries = [i for i in range(1, n) if sites[i][0] != sites[i - 1][0]]
    for b in boundaries:
        ax.axhline(b - 0.5, color="0.2", linewidth=1.6)
        ax.axvline(b - 0.5, color="0.2", linewidth=1.6)

    labels = ["%s\n(%d)" % (o_label, index) for _, o_label, index in sites]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(which="major", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Al labels spanning their block, above the columns and left of the rows
    starts, ends = [0] + boundaries, boundaries + [n]
    for start, end in zip(starts, ends):
        mid = (start + end - 1) / 2.0
        ax.text(mid, -1.55, sites[start][0], ha="center", va="bottom",
                fontsize=10, fontweight="bold")
        ax.text(-1.85, mid, sites[start][0], ha="right", va="center",
                fontsize=10, fontweight="bold")
        ax.plot([start - 0.45, end - 0.55], [-1.45, -1.45], color="0.2", lw=1.2,
                clip_on=False)
        ax.plot([-1.75, -1.75], [start - 0.45, end - 0.55], color="0.2", lw=1.2,
                clip_on=False)
    ax.text(-1.85, -1.55, col_label, ha="right", va="bottom", fontsize=8,
            style="italic", color="0.35")
    ax.text(-1.85, -0.9, row_label, ha="right", va="center", fontsize=8,
            style="italic", color="0.35")

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_title(title, fontsize=11, pad=95)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)