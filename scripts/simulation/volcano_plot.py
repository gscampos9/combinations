"""Volcano plot of the pair permutation test, coloured by decile of combined CDS
length, to show whether length bias still drives significance after correction.

The unit is the pair, never the gene: x is the z-score of the difference
between observed and expected variant count, (n_obs - null_mean) / null_sd,
so each co-occurrence is plotted exactly once. Colour is a diverging Okabe-Ito
ramp (blue = short pairs, orange = long pairs); marker AREA is n_obs (raw
number of variants observed in the pair); marker OUTLINE encodes the BH tier.
Reads/writes comb/simulation/ regardless of the current working directory.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

DATA_DIR = Path(__file__).resolve().parents[2] / "simulation"  # comb/simulation -- data lives there, code here
RESULTS = str(DATA_DIR / "simulation_results_empirical.tsv")   # comb gene1 gene2 length1 length2 n_obs null_mean null_sd z p_onesided p_two_sided q_bh_onesided q_bh_two_sided
OUT = str(DATA_DIR / "volcano_length_empirical.png")
N_BINS = 10                             # deciles of combined_length
TAIL = "two-sided"                      # "greater" = enrichment only; "two-sided" for both directions
CUTOFFS = (0.05, 0.001)                 # BH thresholds, loosest first
SIZE_RANGE = (10, 160)                  # marker area range, mapped from n_obs
EDGE_STYLE = ("white", 0.5), (INK := "#1a1a1a", 1.2), (INK, 2.2)  # ns, q<0.05, q<0.001
N_LABEL = 3                             # top pairs annotated
DPI = 300

# Okabe-Ito

BLUE_ARM = ["#003D5B", "#0072B2", "#56B4E9"]
ORANGE_ARM = ["#F0B93E", "#E69F00", "#8C3D00"]
MUTED, GRID = "#5c5c5c", "#dcdcdc"


def bin_colors(n):
    half = n // 2
    blue = LinearSegmentedColormap.from_list("b", BLUE_ARM)(np.linspace(0, 1, half))
    orange = LinearSegmentedColormap.from_list("o", ORANGE_ARM)(np.linspace(0, 1, n - half))
    return np.vstack([blue, orange])


def load(path):
    """Read the results table; add combined_length, decile, size, and -log10(p)."""
    d = pd.read_csv(path, sep="\t").dropna(subset=["z", "null_mean"])
    d = d[d.null_mean > 0].reset_index(drop=True)
    d["combined_length"] = d.length1 + d.length2
    d["p"] = d.p_onesided if TAIL == "greater" else d.p_two_sided
    d["q"] = d.q_bh_onesided if TAIL == "greater" else d.q_bh_two_sided
    d["nlp"] = -np.log10(d.p)
    d["bin"] = pd.qcut(d.combined_length, N_BINS, labels=False)
    d["msize"] = np.interp(d.n_obs, (d.n_obs.min(), d.n_obs.max()), SIZE_RANGE)
    return d


def tier_style(q):
    """(edgecolor, linewidth) for one q-value, ns -> q<loose -> q<tight."""
    loose, tight = max(CUTOFFS), min(CUTOFFS)
    if q < tight:
        return EDGE_STYLE[2]
    if q < loose:
        return EDGE_STYLE[1]
    return EDGE_STYLE[0]


def tier_labels():
    loose, tight = max(CUTOFFS), min(CUTOFFS)
    return [(EDGE_STYLE[0], f"q ≥ {loose:g}"),
            (EDGE_STYLE[1], f"q < {loose:g}"),
            (EDGE_STYLE[2], f"q < {tight:g}")]


def size_legend_values(d):
    lo, hi = d.n_obs.min(), d.n_obs.max()
    mid = int(round(d.n_obs.median()))
    return sorted(set(int(v) for v in (lo, mid, hi)))


def bin_labels(d):
    labels = []
    for b in range(N_BINS):
        lo, hi = d.combined_length[d.bin == b].agg(["min", "max"])
        labels.append(f"D{b + 1}   {lo / 1000:.1f}–{hi / 1000:.1f} kb")
    return labels


def columns(anchors, span=0.22, gap=0.035, char=0.010):
    """Group anchors that sit within `span` on x and give each group one shared
    label column, placed to the right of every point in it. Text then never sits
    to the left of another label's leader line, so leaders cannot cross labels."""
    groups, current = [], []
    for a in sorted(anchors, key=lambda a: a["fx"]):
        if current and a["fx"] - current[-1]["fx"] > span:
            groups.append(current)
            current = []
        current.append(a)
    groups.append(current)

    for group in groups:
        widest = max(len(a["text"]) for a in group)
        x = min(max(a["fx"] for a in group) + gap, 1.0 - widest * char)
        for a in group:
            a["lx"] = max(x, a["fx"] + 0.012)
    return anchors


def repel(ax, rows, dy=0.032, top=0.97, bottom=0.02):
    """Annotate rows: one label column per cluster, then nudge labels apart
    vertically. Offsets alternate up/down so a dense cluster fans out instead of
    running off the top of the panel."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    anchors = columns([{"text": r.comb, "x": r.z, "y": r.nlp,
                        "fx": (r.z - x0) / (x1 - x0),
                        "fy": (r.nlp - y0) / (y1 - y0)}
                       for _, r in rows.iterrows()])

    used = []
    free = lambda y, x: (bottom <= y <= top and
                         not any(abs(y - u) < dy and abs(x - v) < 0.20 for u, v in used))
    for a in sorted(anchors, key=lambda a: -a["fy"]):
        fy = a["fy"]
        if not free(fy, a["lx"]):
            steps = (s * dy * k for k in range(1, 40) for s in (1, -1))
            fy = next((fy + o for o in steps if free(fy + o, a["lx"])), min(fy, top))
        used.append((fy, a["lx"]))
        ax.annotate(a["text"], (a["x"], a["y"]), xycoords="data",
                    xytext=(a["lx"], fy), textcoords="axes fraction",
                    fontsize=7.5, color=INK, va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED,
                                    shrinkA=1, shrinkB=3))


def volcano(d, out):
    colors = bin_colors(N_BINS)
    fig, ax = plt.subplots(figsize=(9.5, 7))
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)

    # draw largest bubbles first so small ones stay visible on top
    d_sorted = d.sort_values("msize", ascending=False)
    facecolors = [colors[b] for b in d_sorted.bin]
    edge = [tier_style(q) for q in d_sorted.q]
    edgecolors = [e[0] for e in edge]
    linewidths = [e[1] for e in edge]
    ax.scatter(d_sorted.z, d_sorted.nlp, s=d_sorted.msize, c=facecolors,
               edgecolors=edgecolors, linewidths=linewidths, zorder=3)

    ax.axvline(0, color=MUTED, lw=1.0, ls=":", zorder=1)
    for cut, dash in zip(CUTOFFS, ((6, 4), (2, 2))):
        sig = d.q < cut
        if sig.any():
            y = -math.log10(d.p[sig].max())
            ax.axhline(y, color=INK, lw=1.0, dashes=dash, zorder=2)
            ax.text(0.995, y, f"q<{cut:g}", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=8, color=INK, zorder=6,
                    bbox=dict(fc="white", ec="none", pad=1.5))

    ax.margins(y=0.06)
    half = d.z.abs().max() * 1.18
    ax.set_xlim(-half, half)
    ax.set_ylim(0, d.nlp.max() + 0.2)
    # depleted extremes are only worth naming when the test looks both ways
    extremes = [d.nlargest(2, "z")]
    if TAIL != "greater":
        extremes.append(d.nsmallest(2, "z"))
    repel(ax, pd.concat([d.nlargest(N_LABEL, "nlp")] + extremes).drop_duplicates("comb"))

    length_handles = [Line2D([], [], marker="o", ls="", ms=7.5, color=colors[b],
                             mec="white", mew=0.5, label=lab)
                      for b, lab in enumerate(bin_labels(d))]
    leg = ax.legend(handles=length_handles, title="sum_length CDS", loc="upper left",
                    bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8,
                    title_fontsize=8.5, labelspacing=0.6, handletextpad=0.4)
    leg._legend_box.align = "left"
    ax.add_artist(leg)

    tier_handles = [Line2D([], [], marker="o", ls="", ms=9, mfc="#8f8f8f",
                           mec=ec, mew=lw, label=lab)
                    for (ec, lw), lab in tier_labels()]
    leg2 = ax.legend(handles=tier_handles, title="empirical q (BH)", loc="lower left",
                     bbox_to_anchor=(1.02, 0.28), frameon=False, fontsize=8,
                     title_fontsize=8.5, labelspacing=0.6, handletextpad=0.4)
    leg2._legend_box.align = "left"
    ax.add_artist(leg2)

    size_handles = [Line2D([], [], marker="o", ls="", mfc="#8f8f8f", mec="white", mew=0.5,
                           ms=math.sqrt(np.interp(v, (d.n_obs.min(), d.n_obs.max()), SIZE_RANGE)),
                           label=f"n_obs = {v}")
                    for v in size_legend_values(d)]
    leg3 = ax.legend(handles=size_handles, title="n_obs (variants in pair)", loc="lower left",
                     bbox_to_anchor=(1.02, 0.0), frameon=False, fontsize=8,
                     title_fontsize=8.5, labelspacing=0.9, handletextpad=0.6)
    leg3._legend_box.align = "left"

    ax.set_xlabel("z = (n_obs − null_mean) / null_sd", fontsize=10.5, color=INK,
                  labelpad=8)
    ax.set_ylabel("−log$_{10}$(p)  [one-sided]" if TAIL == "greater"
                  else "−log$_{10}$(two-sided p)", fontsize=10.5, color=INK)
    ax.annotate("← fewer variants than expected by length", (0.485, -0.105),
                xycoords="axes fraction", ha="right", fontsize=9, color=MUTED)
    ax.annotate("more variants than expected by length →", (0.515, -0.105),
                xycoords="axes fraction", ha="left", fontsize=9, color=MUTED)

    n = [int((d.q < c).sum()) for c in CUTOFFS]
    ax.set_title("n_variant per gene pair, by sum_length CDS decile\n"
                 f"{len(d)} pairs total, {n[0]} at q<{CUTOFFS[0]:g}",
                 fontsize=11.5, color=INK, loc="left", pad=12)

    fig.subplots_adjust(left=0.085, right=0.72, top=0.895, bottom=0.135)
    fig.savefig(out, dpi=DPI, facecolor="white")
    plt.close(fig)
    return out


def main():
    d = load(RESULTS)
    print(volcano(d, OUT))
    print("decile  n     median_len  %q<0.05  %depleted  median_z")
    for b in range(N_BINS):
        s = d[d.bin == b]
        print(f"D{b + 1:<6} {len(s):<5} {s.combined_length.median():<11.0f} "
              f"{100 * (s.q < CUTOFFS[0]).mean():<8.1f} {100 * (s.z < 0).mean():<10.0f} "
              f"{s.z.median():.2f}")


if __name__ == "__main__":
    main()