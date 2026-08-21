"""
Figure 2 -- effect of gene length on the number of rare-variant gene pairs
a gene participates in. Three panels:
  A. gene_size_variants -- CDS length vs. proband carrier count, full gene
     universe, genes in case gene pairs highlighted blue on a grey background
     (style precedent: scripts_upd/plot/plot_length_ocurrance.py, which made
     the still-present results_221/gene_occurrence_scatter.png).
  B. n_comb_vs_length -- among genes in case gene pairs, CDS length vs.
     number of distinct gene pairs that gene participates in.
  C. "very pertinent 3rd analysis" -- reuses simulation/simulation_results_
     empirical.tsv (already finished, no recompute) to show whether the
     length effect in A/B survives a rigorous length-matched empirical null.

Usage: python fig2_gene_length.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sstats
import matplotlib.pyplot as plt

import _common as c


def _log_trendline(ax, x, y, color, **kw):
    """Fit y ~ log10(x) and draw it straight across a log-x axis."""
    logx = np.log10(x)
    slope, intercept = np.polyfit(logx, y, 1)
    xs = np.linspace(logx.min(), logx.max(), 100)
    ax.plot(10 ** xs, slope * xs + intercept, **kw)


def panel_a(ax, occurrence_df, gene_table_all, case_genes):
    df = occurrence_df.merge(gene_table_all, on="gene", how="left")
    df = df[df["CDS_length"].notna() & (df["CDS_length"] > 0)]
    fg = df[df["gene"].isin(case_genes)]
    bg = df[~df["gene"].isin(case_genes)]

    r_bg, _ = sstats.spearmanr(bg["CDS_length"], bg["n_occurrence_case"])
    r_fg, _ = sstats.spearmanr(fg["CDS_length"], fg["n_occurrence_case"])

    ax.scatter(bg["CDS_length"], bg["n_occurrence_case"], s=9, alpha=0.3,
               color=c.CONTROL_COLOR, edgecolors="none", zorder=1,
               label=f"Other genes (n={len(bg):,}), r={r_bg:.2f}")
    ax.scatter(fg["CDS_length"], fg["n_occurrence_case"], s=16, alpha=0.65,
               color=c.CASE_COLOR, edgecolors="none", zorder=3,
               label=f"Genes in case gene pairs (n={len(fg):,}), r={r_fg:.2f}")
    _log_trendline(ax, bg["CDS_length"], bg["n_occurrence_case"], c.CONTROL_COLOR,
                    linestyle="--", linewidth=1.2, zorder=2)
    _log_trendline(ax, fg["CDS_length"], fg["n_occurrence_case"], c.CASE_COLOR,
                    linestyle="--", linewidth=1.2, zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel("CDS length (bp, log scale)")
    ax.set_ylabel("Proband carriers (n_occurrence)")
    ax.legend(fontsize=6.3, frameon=False, loc="upper left", handletextpad=0.3)
    ax.set_title("Gene size vs. carriers\n(gene_size_variants)", fontsize=9.5)
    return df


def panel_b(ax, case_df, gene_table_case):
    counts = pd.concat([case_df["gene1"], case_df["gene2"]]).value_counts()
    df = gene_table_case.copy()
    df["n_gene_pairs"] = df["gene"].map(counts).fillna(0).astype(int)
    df = df[df["CDS_length"].notna() & (df["CDS_length"] > 0)]

    r, p = sstats.spearmanr(df["CDS_length"], df["n_gene_pairs"])
    ax.scatter(df["CDS_length"], df["n_gene_pairs"], s=18, alpha=0.65,
               color=c.CASE_COLOR, edgecolors="none")
    _log_trendline(ax, df["CDS_length"], df["n_gene_pairs"], "black",
                    linestyle="--", linewidth=1.2)

    top = df.nlargest(3, "n_gene_pairs")
    for _, row in top.iterrows():
        ax.annotate(row["gene"], (row["CDS_length"], row["n_gene_pairs"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=7)

    ax.set_xscale("log")
    ax.set_xlabel("CDS length (bp, log scale)")
    ax.set_ylabel("Number of gene pairs")
    ax.set_title(f"Gene size vs. gene pairs\n(n_comb_vs_length; n={len(df)} genes, "
                 f"Spearman r={r:.2f}, p={p:.2g})", fontsize=9.5)
    return df, r, p


def panel_c(ax):
    sim = pd.read_csv(c.SIMULATION_DIR / "simulation_results_empirical.tsv", sep="\t")
    sim["combined_length"] = sim["length1"] + sim["length2"]
    sig = sim["q_bh_two_sided"] < 0.05  # two-tailed q, per request -- flags both enrichment and depletion

    ax.scatter(sim.loc[~sig, "combined_length"], sim.loc[~sig, "z"], s=15, alpha=0.5,
               color=c.CONTROL_COLOR, edgecolors="none",
               label=f"n.s. after length-null (n={int((~sig).sum())})")
    ax.scatter(sim.loc[sig, "combined_length"], sim.loc[sig, "z"], s=15, alpha=0.8,
               color=c.CASE_COLOR, edgecolors="none",
               label=f"two-tailed q<0.05 (n={int(sig.sum())})")
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel("Combined gene length (bp, log scale)")
    ax.set_ylabel("Permutation z-score\n(observed vs. length-matched null)")
    ax.legend(fontsize=6.3, frameon=False, loc="upper right", handletextpad=0.3)
    ax.set_title(f"Length-matched null model (two-tailed q)\n(simulation/, {len(sim)}/477 "
                 f"case pairs simulated)", fontsize=9.5)
    return sim


def main():
    c.set_style()
    case_df = c.load_stats(str(c.CASE_OUTPUT))
    case_genes = set(case_df["gene1"]) | set(case_df["gene2"])

    _burden_df, occurrence_df = c.parse_input_matrix()
    all_genes = occurrence_df["gene"].tolist()

    print(f"Resolving CDS length/pLI/mis_z for {len(all_genes):,} genes ...")
    gene_table_all = c.build_gene_table(all_genes)
    gene_table_case = gene_table_all[gene_table_all["gene"].isin(case_genes)].copy()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    panel_a_df = panel_a(axes[0], occurrence_df, gene_table_all, case_genes)
    c.panel_label(axes[0], "A")
    panel_b_df, r_b, p_b = panel_b(axes[1], case_df, gene_table_case)
    c.panel_label(axes[1], "B")
    panel_c_df = panel_c(axes[2])
    c.panel_label(axes[2], "C")

    fig.suptitle("Effect of gene length on rare-variant gene pairs", fontsize=11, y=1.02)
    fig.tight_layout()
    c.savefig(fig, "Fig2_gene_length_effect")
    c.save_panels(fig, axes, "Fig2_gene_length_effect")
    plt.close(fig)

    occ_out = occurrence_df.merge(gene_table_all, on="gene", how="left")
    occ_out["in_case_gene_pair"] = occ_out["gene"].isin(case_genes)
    occ_out["n_gene_pairs"] = occ_out["gene"].map(
        pd.concat([case_df["gene1"], case_df["gene2"]]).value_counts()).fillna(0).astype(int)
    out_path = c.FIGURES_DIR / "Fig2_gene_length_effect_data.tsv"
    occ_out.to_csv(out_path, sep="\t", index=False)
    print(f"  saved {out_path}")
    print(f"  (panel C source data is simulation/simulation_results_empirical.tsv, "
          f"unchanged -- not duplicated here)")


if __name__ == "__main__":
    main()
