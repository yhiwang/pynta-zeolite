"""Figure for analysis/: the site x site TS guess-count table.

The relaxation-sweep figures of the old adsorbate branch are in
``archive/adsorbates/analysis/plotting.py``.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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