"""
Figure 4 -- overlap between case gene pairs and control gene pairs, at both
the gene-pair level (does the exact same 2-gene pair recur in both
directions?) and the gene level (does an individual gene show up in a gene
pair on both sides, just paired differently?). These are genuinely
different questions with different answers here, so both get a panel.

Panel A (pairs): manual 2-circle diagram, not proportional-area (matplotlib_
venn isn't installed and isn't needed for an exact 0 -- the circles are
simply drawn non-overlapping since that's the true relationship).
Panel B (genes): proportional-ish 2-circle diagram for the real overlap,
plus a hypergeometric test for whether that overlap is more than expected
by chance given how many genes exist in the RareComb matrix at all.

Also saves the explicit gene-level membership list (case_only/shared/
control_only) and the case gene pairs that involve a case-only gene.

Usage: python fig4_overlap.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sstats
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import _common as c


def draw_two_circles(ax, r1, r2, overlap_frac, label1, label2, n1_only, n_both, n2_only, title):
    """overlap_frac in [0,1] controls how much the two circles interpenetrate
    (0 = side by side, touching)."""
    gap = (r1 + r2) * (1 - overlap_frac) * 0.95
    x1, x2 = -gap / 2, gap / 2
    ax.add_patch(Circle((x1, 0), r1, facecolor=c.CASE_COLOR, alpha=0.55, edgecolor=c.CASE_COLOR, linewidth=1.2))
    ax.add_patch(Circle((x2, 0), r2, facecolor=c.CONTROL_COLOR, alpha=0.55, edgecolor="#7a7a7a", linewidth=1.2))

    ax.text(x1 - r1 * 0.55, 0, f"{n1_only:,}", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(x2 + r2 * 0.55, 0, f"{n2_only:,}", ha="center", va="center", fontsize=11, fontweight="bold")
    if n_both:
        ax.text((x1 + x2) / 2, 0, f"{n_both:,}", ha="center", va="center", fontsize=11, fontweight="bold")
    else:
        ax.text((x1 + x2) / 2, max(r1, r2) * 1.35, "0 shared", ha="center", va="bottom",
                fontsize=9.5, style="italic")

    ax.text(x1, -r1 - max(r1, r2) * 0.22, label1, ha="center", va="top", fontsize=9, color=c.CASE_COLOR,
            fontweight="bold")
    ax.text(x2, -r2 - max(r1, r2) * 0.22, label2, ha="center", va="top", fontsize=9, color="#5a5a5a",
            fontweight="bold")

    span = max(r1, r2) * 1.9
    ax.set_xlim(x1 - span, x2 + span)
    ax.set_ylim(-span * 1.35, span * 1.35)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    ax.set_title(title, fontsize=10)


def main():
    c.set_style()

    case_df = c.load_stats(str(c.CASE_OUTPUT))
    control_df = c.load_stats(str(c.CONTROL_OUTPUT))
    case_pairs = c.combo_pairs(case_df)
    control_pairs = c.combo_pairs(control_df)
    n_pair_overlap = len(case_pairs & control_pairs)

    case_genes = set(case_df["gene1"]) | set(case_df["gene2"])
    control_genes = set(control_df["gene1"]) | set(control_df["gene2"])
    n_gene_overlap = len(case_genes & control_genes)

    # universe = every gene that could possibly appear (the RareComb input matrix's own gene set)
    with open(c.INPUT_MATRIX, "r", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
    n_universe = len(header) - 2  # drop Sample_Name, Output_1
    # one-sided: is the observed gene overlap >= what's expected by chance?
    p_hyper = sstats.hypergeom.sf(n_gene_overlap - 1, n_universe, len(control_genes), len(case_genes))
    expected_overlap = len(case_genes) * len(control_genes) / n_universe

    print(f"Gene-pair overlap: {n_pair_overlap} / {len(case_pairs)} case pairs, {len(control_pairs)} control pairs")
    print(f"Gene overlap: {n_gene_overlap} / {len(case_genes)} case genes, {len(control_genes)} control genes "
          f"(expected by chance ~{expected_overlap:.1f} of {n_universe} total genes, hypergeometric p={p_hyper:.3g})")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3), constrained_layout=True)

    draw_two_circles(
        axes[0], r1=1.0, r2=1.0, overlap_frac=0.0,
        label1="Case gene pairs", label2="Control gene pairs",
        n1_only=len(case_pairs), n_both=n_pair_overlap, n2_only=len(control_pairs),
        title="Gene pair overlap",
    )

    r1_g = 1.0
    r2_g = r1_g * np.sqrt(len(control_genes) / len(case_genes))
    overlap_frac_g = n_gene_overlap / min(len(case_genes), len(control_genes))
    draw_two_circles(
        axes[1], r1=r1_g, r2=r2_g, overlap_frac=min(overlap_frac_g, 0.85),
        label1="Case genes", label2="Control genes",
        n1_only=len(case_genes) - n_gene_overlap, n_both=n_gene_overlap,
        n2_only=len(control_genes) - n_gene_overlap,
        title="Individual gene overlap",
    )
    axes[1].text(0, -max(r1_g, r2_g) * 1.75,
                 f"hypergeometric p={p_hyper:.2g}  (expected ~{expected_overlap:.0f} by chance,\n"
                 f"of {n_universe:,} genes in the RareComb matrix)",
                 ha="center", va="top", fontsize=7.8)

    c.panel_label(axes[0], "A")
    c.panel_label(axes[1], "B")
    fig.suptitle("Overlap between case and control gene pairs", fontsize=12)
    c.savefig(fig, "Fig4_overlap")
    c.save_panels(fig, axes, "Fig4_overlap")
    plt.close(fig)

    summary = pd.DataFrame([
        {"level": "gene_pair", "n_case": len(case_pairs), "n_control": len(control_pairs),
         "n_overlap": n_pair_overlap, "n_universe": np.nan, "expected_by_chance": np.nan,
         "hypergeometric_p": np.nan},
        {"level": "gene", "n_case": len(case_genes), "n_control": len(control_genes),
         "n_overlap": n_gene_overlap, "n_universe": n_universe,
         "expected_by_chance": expected_overlap, "hypergeometric_p": p_hyper},
    ])
    out_path = c.FIGURES_DIR / "Fig4_overlap_data.tsv"
    summary.to_csv(out_path, sep="\t", index=False)
    print(f"  saved {out_path}")

    # ── explicit gene membership list, incl. the case-only / control-only sets ──
    case_only = sorted(case_genes - control_genes)
    control_only = sorted(control_genes - case_genes)
    shared = sorted(case_genes & control_genes)
    print(f"Case-only genes (in a case gene pair, absent from every control gene pair): "
          f"{len(case_only)}")
    print(f"Control-only genes (in a control gene pair, absent from every case gene pair): "
          f"{len(control_only)}")

    membership = pd.DataFrame({
        "gene": sorted(case_genes | control_genes),
    })
    membership["in_case_gene_pairs"] = membership["gene"].isin(case_genes)
    membership["in_control_gene_pairs"] = membership["gene"].isin(control_genes)
    membership["group"] = np.select(
        [membership["gene"].isin(case_only), membership["gene"].isin(control_only)],
        ["case_only", "control_only"], default="shared",
    )
    genes_path = c.FIGURES_DIR / "Fig4_gene_membership.tsv"
    membership.sort_values(["group", "gene"]).to_csv(genes_path, sep="\t", index=False)
    print(f"  saved {genes_path}  ({len(case_only)} case_only, {len(control_only)} control_only, "
          f"{len(shared)} shared)")

    # ── case gene pairs that involve a case-only gene ───────────────────────
    case_only_set = set(case_only)
    involves_case_only = case_df[case_df["gene1"].isin(case_only_set) | case_df["gene2"].isin(case_only_set)].copy()
    involves_case_only["gene1_case_only"] = involves_case_only["gene1"].isin(case_only_set)
    involves_case_only["gene2_case_only"] = involves_case_only["gene2"].isin(case_only_set)
    keep_cols = ["gene1", "gene2", "gene1_case_only", "gene2_case_only",
                 "Case_Obs_Count_Combo", "n_fam", "n_pro", "Case_Adj_Pval_BH"]
    involves_case_only = involves_case_only[[col for col in keep_cols if col in involves_case_only.columns]]
    pairs_path = c.FIGURES_DIR / "Case_gene_pairs_with_case_only_gene.tsv"
    involves_case_only.to_csv(pairs_path, sep="\t", index=False)
    print(f"  saved {pairs_path}  ({len(involves_case_only)} of {len(case_df)} case gene pairs "
          f"involve at least one case-only gene)")


if __name__ == "__main__":
    main()
