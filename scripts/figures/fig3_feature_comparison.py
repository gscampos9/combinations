"""
Figure 3 -- feature comparison between case gene pairs (CADD20_2_2_1_output.txt,
477 pairs) and control gene pairs (CADD20_2_2_1_output_control.txt, 1573 pairs,
a reversed-direction RareComb run enriched in siblings).

Laid out in two blocks, gene-level features first, then gene-pair-level
features (row 1 then row 2):
  Gene-level   A. missense z-score   B. pLI   C. brain expression (expressed/enriched)
  Pair-level   D. GO term Jaccard    E. mean STRING PPI   F. PPI at 0.4/0.7/0.9   G. coexpression

Gene-level panels deduplicate to one row per unique gene per group -- a gene
in 5 gene pairs counts once -- but a gene can legitimately appear in BOTH
groups (252 of 1846 genes do), so group membership is not mutually
exclusive; that overlap is quantified in Figure 4, not here.

GO Jaccard: all 3 ontologies (BP+MF+CC) combined into one gene->GO-id set,
terms restricted to 3-500 annotated genes.
PPI: STRING combined_score, missing edge imputed as 0 (matches
common/db_loaders.py::ppi_score) -- stated on the axis since it conflates
"tested, no interaction" with "never tested".
Coexpression: are the two genes of a pair in the same PsychENCODE INT-09
WGCNA module -- a plain pair-level %, deliberately simpler than the earlier
BrainSpan-module-size metric.

Stats: two-sided Mann-Whitney U for continuous features, Fisher's exact for
proportions, BH-adjusted across all 10 tests in this figure. Every
significant bracket shows both the star tier and the actual p-value; every
box plot shows both the median/quartiles and the mean (diamond marker);
every panel's x-tick labels carry that panel's own N.

Usage: python fig3_feature_comparison.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import _common as c


def main():
    c.set_style()

    case_df = c.load_stats(str(c.CASE_OUTPUT))
    control_df = c.load_stats(str(c.CONTROL_OUTPUT))
    case_genes = set(case_df["gene1"]) | set(case_df["gene2"])
    control_genes = set(control_df["gene1"]) | set(control_df["gene2"])
    all_genes = case_genes | control_genes
    print(f"Case pairs: {len(case_df)}  Control pairs: {len(control_df)}")
    print(f"Case genes: {len(case_genes)}  Control genes: {len(control_genes)}")

    # ── gene-level data ──────────────────────────────────────────────────
    gene_table = c.build_gene_table(all_genes)
    db_hits = c.build_database_hits(all_genes)
    gene_table = gene_table.merge(db_hits, on="gene")
    case_gf = gene_table[gene_table["gene"].isin(case_genes)].assign(group="case")
    control_gf = gene_table[gene_table["gene"].isin(control_genes)].assign(group="control")

    # ── pair-level data ──────────────────────────────────────────────────
    go_index = c.build_go_term_index(min_genes=3, max_genes=500)
    ppi = c.load_ppi_for_genes(all_genes)
    modules = c.load_coexpression_modules()
    case_pf = c.pair_features(case_df, go_index, ppi, modules)
    control_pf = c.pair_features(control_df, go_index, ppi, modules)
    case_coexpr = case_pf["coexpressed"].dropna().astype(bool)
    control_coexpr = control_pf["coexpressed"].dropna().astype(bool)

    # ── stats: gather every raw p first, BH-adjust together ────────────────
    t_mis_z = c.mannwhitney(case_gf["mis_z"], control_gf["mis_z"])
    t_pli = c.mannwhitney(case_gf["pLI"], control_gf["pLI"])
    exp_case, exp_control, _, p_exp = c.fisher_fraction(case_gf["brain_expressed"], control_gf["brain_expressed"])
    enr_case, enr_control, _, p_enr = c.fisher_fraction(case_gf["brain_enriched"], control_gf["brain_enriched"])
    t_go = c.mannwhitney(case_pf["go_jaccard"], control_pf["go_jaccard"])
    t_ppi = c.mannwhitney(case_pf["ppi_score"], control_pf["ppi_score"])
    ppi_rows = [c.fisher_fraction(case_pf["ppi_score"] >= t, control_pf["ppi_score"] >= t) for t in (0.4, 0.7, 0.9)]
    coex_case_pct, coex_control_pct, _, p_coex = c.fisher_fraction(case_coexpr, control_coexpr)

    raw_p = [t_mis_z[1], t_pli[1], p_exp, p_enr, t_go[1], t_ppi[1],
             ppi_rows[0][3], ppi_rows[1][3], ppi_rows[2][3], p_coex]
    names = ["mis_z", "pLI", "brain_expressed", "brain_enriched", "go_jaccard", "ppi_score",
             "ppi_ge_0.4", "ppi_ge_0.7", "ppi_ge_0.9", "coexpression"]
    adj_p = c.bh_adjust(raw_p)
    p_bh = dict(zip(names, adj_p))

    # ── plot: row 1 = gene-level, row 2 = pair-level ────────────────────────
    fig = plt.figure(figsize=(15.5, 9.0))
    gs = fig.add_gridspec(2, 12, hspace=0.7, wspace=1.4)
    ax_mis_z = fig.add_subplot(gs[0, 0:4])
    ax_pli = fig.add_subplot(gs[0, 4:8])
    ax_brain = fig.add_subplot(gs[0, 8:12])
    ax_go = fig.add_subplot(gs[1, 0:3])
    ax_ppi = fig.add_subplot(gs[1, 3:6])
    ax_ppi_thresh = fig.add_subplot(gs[1, 6:9])
    ax_coex = fig.add_subplot(gs[1, 9:12])

    fig.text(0.06, 0.94, "Gene-level features", fontsize=11, fontweight="bold")
    fig.text(0.06, 0.47, "Gene-pair-level features", fontsize=11, fontweight="bold")

    c.violin(ax_mis_z, case_gf["mis_z"].dropna(), control_gf["mis_z"].dropna(),
             "Missense z-score", "Missense constraint")
    c.annotate_bracket(ax_mis_z, case_gf["mis_z"].dropna().values, control_gf["mis_z"].dropna().values,
                        c.stars(p_bh["mis_z"]), p_bh["mis_z"])
    c.panel_label(ax_mis_z, "A")

    c.violin(ax_pli, case_gf["pLI"].dropna(), control_gf["pLI"].dropna(), "pLI", "LoF intolerance")
    c.annotate_bracket(ax_pli, case_gf["pLI"].dropna().values, control_gf["pLI"].dropna().values,
                        c.stars(p_bh["pLI"]), p_bh["pLI"])
    c.panel_label(ax_pli, "B")

    c.grouped_bar(ax_brain, ["Expressed", "Enriched"], [exp_case, enr_case], [exp_control, enr_control],
                  len(case_gf), len(control_gf), "% of genes", "Brain expression (HPA)",
                  [c.stars(p_bh["brain_expressed"]), c.stars(p_bh["brain_enriched"])],
                  [p_bh["brain_expressed"], p_bh["brain_enriched"]],
                  counts_a=[int(case_gf["brain_expressed"].sum()), int(case_gf["brain_enriched"].sum())],
                  counts_b=[int(control_gf["brain_expressed"].sum()), int(control_gf["brain_enriched"].sum())])
    c.panel_label(ax_brain, "C")

    c.violin(ax_go, case_pf["go_jaccard"], control_pf["go_jaccard"],
             "GO term Jaccard\n(BP+MF+CC, 3–500 genes/term)", "GO term overlap")
    c.annotate_bracket(ax_go, case_pf["go_jaccard"].values, control_pf["go_jaccard"].values,
                        c.stars(p_bh["go_jaccard"]), p_bh["go_jaccard"])
    c.panel_label(ax_go, "D")

    c.violin(ax_ppi, case_pf["ppi_score"], control_pf["ppi_score"],
             "STRING PPI score\n(missing edge = 0)", "Protein-protein interaction")
    c.annotate_bracket(ax_ppi, case_pf["ppi_score"].values, control_pf["ppi_score"].values,
                        c.stars(p_bh["ppi_score"]), p_bh["ppi_score"])
    c.panel_label(ax_ppi, "E")

    thresh_labels = ["≥ 0.4", "≥ 0.7", "≥ 0.9"]
    thresh_case = [r[0] for r in ppi_rows]
    thresh_control = [r[1] for r in ppi_rows]
    thresh_stars = [c.stars(p_bh[k]) for k in ("ppi_ge_0.4", "ppi_ge_0.7", "ppi_ge_0.9")]
    thresh_p = [p_bh[k] for k in ("ppi_ge_0.4", "ppi_ge_0.7", "ppi_ge_0.9")]
    thresh_counts_case = [int((case_pf["ppi_score"] >= t).sum()) for t in (0.4, 0.7, 0.9)]
    thresh_counts_control = [int((control_pf["ppi_score"] >= t).sum()) for t in (0.4, 0.7, 0.9)]
    c.grouped_bar(ax_ppi_thresh, thresh_labels, thresh_case, thresh_control,
                  len(case_pf), len(control_pf), "% gene pairs", "PPI score thresholds",
                  thresh_stars, thresh_p,
                  counts_a=thresh_counts_case, counts_b=thresh_counts_control)
    c.panel_label(ax_ppi_thresh, "F")

    c.grouped_bar(ax_coex, ["Same module"], [coex_case_pct], [coex_control_pct],
                  len(case_coexpr), len(control_coexpr), "% gene pairs",
                  "Coexpression\n(PsychENCODE INT-09)",
                  [c.stars(p_bh["coexpression"])], [p_bh["coexpression"]],
                  counts_a=[int(case_coexpr.sum())], counts_b=[int(control_coexpr.sum())])
    c.panel_label(ax_coex, "G")

    fig.suptitle("Case vs. control gene pair features", fontsize=12, y=0.985)
    c.savefig(fig, "Fig3_feature_comparison")
    c.save_panels(fig, [
        (ax_mis_z, "A"), (ax_pli, "B"), (ax_brain, "C"), (ax_go, "D"),
        (ax_ppi, "E"), (ax_ppi_thresh, "F"), (ax_coex, "G"),
    ], "Fig3_feature_comparison")
    plt.close(fig)

    # ── source data ──────────────────────────────────────────────────────
    gene_wide = pd.concat([case_gf, control_gf], ignore_index=True)
    gene_long = gene_wide.melt(id_vars=["gene", "group"],
                                value_vars=["mis_z", "pLI", "brain_expressed", "brain_enriched"],
                                var_name="feature", value_name="value")
    gene_long.insert(0, "level", "gene")

    pair_wide = pd.concat([case_pf.assign(group="case"), control_pf.assign(group="control")], ignore_index=True)
    pair_wide["id"] = pair_wide["gene1"] + "," + pair_wide["gene2"]
    pair_long = pair_wide.melt(id_vars=["id", "group"], value_vars=["go_jaccard", "ppi_score", "coexpressed"],
                                var_name="feature", value_name="value")
    pair_long.insert(0, "level", "pair")
    pair_long = pair_long.rename(columns={"id": "gene"})

    full_long = pd.concat([gene_long, pair_long], ignore_index=True)
    data_path = c.FIGURES_DIR / "Fig3_feature_comparison_data.tsv"
    full_long.to_csv(data_path, sep="\t", index=False)
    print(f"  saved {data_path}")

    stats_rows = [
        {"feature": "mis_z", "test": "mannwhitney_u", "statistic": t_mis_z[0],
         "n_case": t_mis_z[3], "n_control": t_mis_z[4], "p_raw": t_mis_z[1],
         "p_bh": p_bh["mis_z"], "effect_size": t_mis_z[2]},
        {"feature": "pLI", "test": "mannwhitney_u", "statistic": t_pli[0],
         "n_case": t_pli[3], "n_control": t_pli[4], "p_raw": t_pli[1],
         "p_bh": p_bh["pLI"], "effect_size": t_pli[2]},
        {"feature": "brain_expressed", "test": "fisher_exact", "statistic": np.nan,
         "n_case": len(case_gf), "n_control": len(control_gf), "p_raw": p_exp,
         "p_bh": p_bh["brain_expressed"], "effect_size": np.nan},
        {"feature": "brain_enriched", "test": "fisher_exact", "statistic": np.nan,
         "n_case": len(case_gf), "n_control": len(control_gf), "p_raw": p_enr,
         "p_bh": p_bh["brain_enriched"], "effect_size": np.nan},
        {"feature": "go_jaccard", "test": "mannwhitney_u", "statistic": t_go[0],
         "n_case": t_go[3], "n_control": t_go[4], "p_raw": t_go[1],
         "p_bh": p_bh["go_jaccard"], "effect_size": t_go[2]},
        {"feature": "ppi_score", "test": "mannwhitney_u", "statistic": t_ppi[0],
         "n_case": t_ppi[3], "n_control": t_ppi[4], "p_raw": t_ppi[1],
         "p_bh": p_bh["ppi_score"], "effect_size": t_ppi[2]},
        {"feature": "ppi_ge_0.4", "test": "fisher_exact", "statistic": np.nan,
         "n_case": len(case_pf), "n_control": len(control_pf), "p_raw": ppi_rows[0][3],
         "p_bh": p_bh["ppi_ge_0.4"], "effect_size": np.nan},
        {"feature": "ppi_ge_0.7", "test": "fisher_exact", "statistic": np.nan,
         "n_case": len(case_pf), "n_control": len(control_pf), "p_raw": ppi_rows[1][3],
         "p_bh": p_bh["ppi_ge_0.7"], "effect_size": np.nan},
        {"feature": "ppi_ge_0.9", "test": "fisher_exact", "statistic": np.nan,
         "n_case": len(case_pf), "n_control": len(control_pf), "p_raw": ppi_rows[2][3],
         "p_bh": p_bh["ppi_ge_0.9"], "effect_size": np.nan},
        {"feature": "coexpression", "test": "fisher_exact", "statistic": np.nan,
         "n_case": len(case_coexpr), "n_control": len(control_coexpr), "p_raw": p_coex,
         "p_bh": p_bh["coexpression"], "effect_size": np.nan},
    ]
    for row in stats_rows:
        row["stars"] = c.stars(row["p_bh"])
    stats_df = pd.DataFrame(stats_rows)
    stats_path = c.FIGURES_DIR / "Fig3_feature_comparison_stats.tsv"
    stats_df.to_csv(stats_path, sep="\t", index=False)
    print(f"  saved {stats_path}")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
