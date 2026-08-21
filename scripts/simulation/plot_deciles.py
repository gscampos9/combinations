"""Per decile of combined CDS length: distribution of pair size (violin, log scale)
and proportion of pairs excluded as length-explained (q >= CUTOFF)."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

DATA_DIR = Path(__file__).resolve().parents[2] / "simulation"  # comb/simulation -- data lives there, code here
RESULTS = str(DATA_DIR / "simulation_results_empirical.tsv")
QCOL = "q_bh_onesided"
CUTOFF = 0.05
N_BINS = 10
OUT = str(DATA_DIR / "deciles_length_excluded.png")
DPI = 300

BLUE_ARM = ["#003D5B", "#0072B2", "#56B4E9"]
ORANGE_ARM = ["#F0B93E", "#E69F00", "#8C3D00"]
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#dcdcdc"


def bin_colors(n):
    half = n // 2
    blue = LinearSegmentedColormap.from_list("b", BLUE_ARM)(np.linspace(0, 1, half))
    orange = LinearSegmentedColormap.from_list("o", ORANGE_ARM)(np.linspace(0, 1, n - half))
    return np.vstack([blue, orange])


d = pd.read_csv(RESULTS, sep="\t").dropna(subset=["z"])
d["combined_length"] = d.length1 + d.length2  # not a column in simulation_results_empirical.tsv itself
d["bin"] = pd.qcut(d.combined_length, N_BINS, labels=False)
d["excluded"] = d[QCOL] >= CUTOFF

colors = bin_colors(N_BINS)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                               gridspec_kw={"height_ratios": [1.35, 1]})

groups = [np.log10(d.combined_length[d.bin == b].values) for b in range(N_BINS)]
parts = ax1.violinplot(groups, positions=range(1, N_BINS + 1), widths=0.85,
                       showextrema=False, showmedians=False)
for body, color in zip(parts["bodies"], colors):
    body.set_facecolor(color)
    body.set_edgecolor("white")
    body.set_linewidth(0.6)
    body.set_alpha(0.9)
ax1.boxplot(groups, positions=range(1, N_BINS + 1), widths=0.12, showfliers=False,
            medianprops=dict(color=INK, lw=1.4),
            boxprops=dict(color=INK, lw=0.8),
            whiskerprops=dict(color=INK, lw=0.8),
            capprops=dict(color=INK, lw=0.8))

ticks = [3, np.log10(3000), 4, np.log10(30000), 5, np.log10(300000)]
ax1.set_yticks(ticks)
ax1.set_yticklabels(["1 kb", "3 kb", "10 kb", "30 kb", "100 kb", "300 kb"])
ax1.set_ylabel("combined CDS length", fontsize=10.5, color=INK)
ax1.set_title("Pair size distribution and length-explained pairs, by decile",
              fontsize=11.5, color=INK, loc="left", pad=12)

pct = [100 * d.excluded[d.bin == b].mean() for b in range(N_BINS)]
n_exc = [int(d.excluded[d.bin == b].sum()) for b in range(N_BINS)]
n_tot = [int((d.bin == b).sum()) for b in range(N_BINS)]
ax2.bar(range(1, N_BINS + 1), pct, color=colors, edgecolor="white", linewidth=0.6)
for i, (p, e, t) in enumerate(zip(pct, n_exc, n_tot), start=1):
    ax2.text(i, p + 1.5, f"{e}/{t}", ha="center", fontsize=8, color=MUTED)

ax2.set_ylim(0, max(pct) * 1.22 if max(pct) else 1)
ax2.set_ylabel(f"% pairs excluded\n({QCOL} ≥ {CUTOFF:g})", fontsize=10.5, color=INK)
ax2.set_xlabel("combined CDS length decile", fontsize=10.5, color=INK)
ax2.set_xticks(range(1, N_BINS + 1))
ax2.set_xticklabels([f"D{b + 1}" for b in range(N_BINS)])

for ax in (ax1, ax2):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(OUT, dpi=DPI, facecolor="white")
print(OUT)
print("decile  n     excluded  pct_excluded  len_min  len_med  len_max")
for b in range(N_BINS):
    s = d[d.bin == b]
    print(f"D{b + 1:<6} {len(s):<5} {n_exc[b]:<9} {pct[b]:<13.1f} "
          f"{s.combined_length.min():<8.0f} {s.combined_length.median():<8.0f} "
          f"{s.combined_length.max():.0f}")
