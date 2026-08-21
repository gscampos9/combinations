"""
Figure 8 -- NDD genes within case and control gene pairs.

"NDD gene" is already the gene1_ndd_gene/gene2_ndd_gene column used
throughout annotated_gene_pairs_*.tsv -- a gene with >=1 hit across the
published NDD/ASD/DD gene-discovery lists aggregated in
databases/NDDgenes.txt (2,041 genes with >=1 hit). Panel A makes that
definition transparent instead of a
black box: a schematic of the 9 source lists (a 10th column, NDD_com, is
"y" for every single row in the file -- a rollup flag, not an independent
source, so it's left out of the schematic) feeding into the one ndd_gene
flag used everywhere downstream.

Panel B is the actual gene-pair-level comparison: how many of a pair's 2
genes are NDD genes (0/1/2), cases vs. controls.

Usage: python fig8_ndd_genes.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import _common as c

_SOURCE_EXCLUDE = {"NDD_com"}  # rollup column, "y" for every row -- not an independent source


def source_counts() -> tuple[dict, int]:
    """Per-source gene counts from databases/NDDgenes.txt, and the union
    (== _common.py's own ndd_gene flag, cross-checked in main())."""
    ev = c.load_ndd_evidence(str(c.DB_DIR))
    cols = [col for col in ev.columns if col not in _SOURCE_EXCLUDE]
    counts = {col: int(ev[col].sum()) for col in cols}
    total = int(ev[cols].any(axis=1).sum())
    return counts, total


def panel_schematic(ax, counts: dict, total: int):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    sources = sorted(counts.items(), key=lambda kv: -kv[1])
    n = len(sources)
    ys = np.linspace(0.93, 0.07, n)
    box_x, box_w, box_h = 0.02, 0.40, 0.085

    right_x, right_y, right_w, right_h = 0.68, 0.5, 0.30, 0.26

    for (name, count), y in zip(sources, ys):
        ax.annotate("", xy=(right_x, right_y), xytext=(box_x + box_w, y),
                    arrowprops=dict(arrowstyle="-", color="#b5b5b5", linewidth=0.7,
                                     shrinkA=0, shrinkB=0))

    for (name, count), y in zip(sources, ys):
        ax.add_patch(FancyBboxPatch((box_x, y - box_h / 2), box_w, box_h,
                                     boxstyle="round,pad=0.006,rounding_size=0.012",
                                     facecolor="#eeeeee", edgecolor="#8a8a8a", linewidth=0.8, zorder=3))
        ax.text(box_x + box_w / 2, y, f"{name}\n(n={count:,})", ha="center", va="center",
                fontsize=6.8, zorder=4)

    ax.add_patch(FancyBboxPatch((right_x, right_y - right_h / 2), right_w, right_h,
                                 boxstyle="round,pad=0.012,rounding_size=0.02",
                                 facecolor=c.CASE_COLOR, edgecolor=c.CASE_COLOR, linewidth=1.2, zorder=3))
    ax.text(right_x + right_w / 2, right_y, f"NDD gene\n(n={total:,})\n≥ 1 of {n} sources",
            ha="center", va="center", fontsize=10, color="white", fontweight="bold", zorder=4)

    ax.text(box_x + box_w / 2, 1.005, "Published NDD/ASD/DD gene-discovery lists\n(databases/NDDgenes.txt)",
            ha="center", va="bottom", fontsize=7.5, style="italic", transform=ax.transAxes)
    ax.set_title("What counts as an NDD gene", fontsize=10.5)


def classify_pairs(stats_df: pd.DataFrame, ndd_genes: pd.Series) -> pd.DataFrame:
    out = stats_df[["gene1", "gene2"]].copy()
    out["gene1_ndd"] = out["gene1"].map(ndd_genes).fillna(False)
    out["gene2_ndd"] = out["gene2"].map(ndd_genes).fillna(False)
    out["n_ndd_genes"] = out["gene1_ndd"].astype(int) + out["gene2_ndd"].astype(int)
    return out


def main():
    c.set_style()

    counts, total_from_sources = source_counts()
    print(f"NDD gene sources (databases/NDDgenes.txt, excluding the {sorted(_SOURCE_EXCLUDE)} rollup column):")
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name}: {n:,} genes")
    print(f"  union: {total_from_sources:,} genes")

    case_df = c.load_stats(str(c.CASE_OUTPUT))
    control_df = c.load_stats(str(c.CONTROL_OUTPUT))
    all_genes = sorted(set(case_df.gene1) | set(case_df.gene2) | set(control_df.gene1) | set(control_df.gene2))
    db = c.build_database_hits(all_genes).set_index("gene")
    ndd_genes = db["ndd_gene"]
    print(f"NDD genes among the {len(all_genes):,} genes in any case/control gene pair: {int(ndd_genes.sum())}")

    case_pairs = classify_pairs(case_df, ndd_genes)
    control_pairs = classify_pairs(control_df, ndd_genes)
    for k in (0, 1, 2):
        print(f"  {k} NDD gene(s): {int((case_pairs['n_ndd_genes'] == k).sum())} case pairs, "
              f"{int((control_pairs['n_ndd_genes'] == k).sum())} control pairs")

    # ── stats: % pairs by NDD-gene count, cases vs. controls ────────────────
    rows = [c.fisher_fraction(case_pairs["n_ndd_genes"] == k, control_pairs["n_ndd_genes"] == k) for k in (0, 1, 2)]
    raw_p = [r[3] for r in rows]
    p_bh = c.bh_adjust(raw_p)

    # ── plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1.35, 1]},
                              constrained_layout=True)

    panel_schematic(axes[0], counts, total_from_sources)
    c.panel_label(axes[0], "A")

    case_counts = [int((case_pairs["n_ndd_genes"] == k).sum()) for k in (0, 1, 2)]
    control_counts = [int((control_pairs["n_ndd_genes"] == k).sum()) for k in (0, 1, 2)]

    c.grouped_bar(axes[1], ["0 NDD genes", "1 NDD gene", "2 NDD genes"],
                  [r[0] for r in rows], [r[1] for r in rows],
                  len(case_pairs), len(control_pairs), "% gene pairs", "NDD genes per pair",
                  [c.stars(p) for p in p_bh], p_bh,
                  counts_a=case_counts, counts_b=control_counts)
    c.panel_label(axes[1], "B")

    fig.suptitle("NDD genes within case and control gene pairs", fontsize=12)
    c.savefig(fig, "Fig8_ndd_genes")
    c.save_panels(fig, axes, "Fig8_ndd_genes")
    plt.close(fig)

    # ── source data ──────────────────────────────────────────────────────
    src_df = pd.DataFrame(sorted(counts.items(), key=lambda kv: -kv[1]), columns=["source", "n_genes"])
    src_df.loc[len(src_df)] = ["UNION (ndd_gene)", total_from_sources]
    src_path = c.FIGURES_DIR / "Fig8_ndd_gene_sources.tsv"
    src_df.to_csv(src_path, sep="\t", index=False)
    print(f"  saved {src_path}")

    case_pairs["group"] = "case"
    control_pairs["group"] = "control"
    data_df = pd.concat([case_pairs, control_pairs], ignore_index=True)
    data_path = c.FIGURES_DIR / "Fig8_ndd_genes_data.tsv"
    data_df.to_csv(data_path, sep="\t", index=False)
    print(f"  saved {data_path}")

    stats_df = pd.DataFrame([
        {"n_ndd_genes": k, "n_case_hit": ca, "n_control_hit": co,
         "pct_case": r[0], "pct_control": r[1], "test": "fisher_exact",
         "n_case": len(case_pairs), "n_control": len(control_pairs), "p_raw": r[3], "p_bh": p}
        for k, r, ca, co, p in zip((0, 1, 2), rows, case_counts, control_counts, p_bh)
    ])
    stats_df["stars"] = stats_df["p_bh"].apply(c.stars)
    stats_path = c.FIGURES_DIR / "Fig8_ndd_genes_stats.tsv"
    stats_df.to_csv(stats_path, sep="\t", index=False)
    print(f"  saved {stats_path}")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
