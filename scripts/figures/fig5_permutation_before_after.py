"""
Figure 5 -- the full Fig3-style feature battery, but "before" (all 477 case
gene pairs) vs. "after" (the subset remaining after the length-matched null
simulation, two-tailed q_bh<0.05, 77 of the 464 simulated pairs -- see
Figure 2 panel C) instead of case-vs-control. This is how the null-
simulation result gets integrated into the rest of the analysis rather than
staying a single isolated panel.

Same two-block layout as Figure 3 (gene-level features, then gene-pair-level
features), same stats/annotation conventions (mean marker + median/
quartiles on every violin, p-value text on every significant bracket, N in
every panel's tick labels, BH-adjustment across this figure's own test set).

"Before" and "after" are both still the CASE group throughout (a within-case
comparison, not case-vs-control) -- shaded as two blues so colour keeps
meaning "case" consistently across every figure: light blue = all case gene
pairs, the established case-blue = gene pairs after null simulation.

Usage: python fig5_permutation_before_after.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import _common as c

BEFORE_COLOR = "#a9c6e8"     # light blue -- all case gene pairs
AFTER_COLOR = c.CASE_COLOR   # established case-blue -- gene pairs after null simulation
LABEL_BEFORE = "All case gene pairs"
LABEL_AFTER = "After null simulation"


def main():
    c.set_style()

    case_df = c.load_stats(str(c.CASE_OUTPUT))
    perm = c.load_permutation_results()
    perm["combination"] = [",".join(sorted([a, b])) for a, b in zip(perm.gene1, perm.gene2)]
    after_null_combos = set(perm.loc[perm["q_bh_two_sided"] < 0.05, "combination"])

    case_df = case_df.copy()
    case_df["combination"] = [",".join(sorted([a, b])) for a, b in zip(case_df.gene1, case_df.gene2)]
    after_df = case_df[case_df["combination"].isin(after_null_combos)]
    n_before, n_after = len(case_df), len(after_df)
    print(f"All case gene pairs: {n_before}  Simulated: {perm['combination'].nunique()}  "
          f"Remain after null simulation (two-tailed q<0.05): {n_after}")

    before_genes = set(case_df.gene1) | set(case_df.gene2)
    after_genes = set(after_df.gene1) | set(after_df.gene2)
    print(f"Genes involved: {len(before_genes)} before, {len(after_genes)} after")

    # ── gene-level data ──────────────────────────────────────────────────
    gene_table = c.build_gene_table(sorted(before_genes))
    db_hits = c.build_database_hits(sorted(before_genes))
    gene_table = gene_table.merge(db_hits, on="gene")
    before_gf = gene_table.assign(group="before")
    after_gf = gene_table[gene_table["gene"].isin(after_genes)].assign(group="after")

    # ── pair-level data ──────────────────────────────────────────────────
    go_index = c.build_go_term_index(min_genes=3, max_genes=500)
    ppi = c.load_ppi_for_genes(before_genes)
    modules = c.load_coexpression_modules()
    before_pf = c.pair_features(case_df, go_index, ppi, modules)
    after_pf = c.pair_features(after_df, go_index, ppi, modules)
    before_coexpr = before_pf["coexpressed"].dropna().astype(bool)
    after_coexpr = after_pf["coexpressed"].dropna().astype(bool)

    # ── stats ────────────────────────────────────────────────────────────
    t_mis_z = c.mannwhitney(before_gf["mis_z"], after_gf["mis_z"])
    t_pli = c.mannwhitney(before_gf["pLI"], after_gf["pLI"])
    exp_b, exp_a, _, p_exp = c.fisher_fraction(before_gf["brain_expressed"], after_gf["brain_expressed"])
    enr_b, enr_a, _, p_enr = c.fisher_fraction(before_gf["brain_enriched"], after_gf["brain_enriched"])
    t_go = c.mannwhitney(before_pf["go_jaccard"], after_pf["go_jaccard"])
    t_ppi = c.mannwhitney(before_pf["ppi_score"], after_pf["ppi_score"])
    ppi_rows = [c.fisher_fraction(before_pf["ppi_score"] >= t, after_pf["ppi_score"] >= t) for t in (0.4, 0.7, 0.9)]
    coex_b_pct, coex_a_pct, _, p_coex = c.fisher_fraction(before_coexpr, after_coexpr)

    raw_p = [t_mis_z[1], t_pli[1], p_exp, p_enr, t_go[1], t_ppi[1],
             ppi_rows[0][3], ppi_rows[1][3], ppi_rows[2][3], p_coex]
    names = ["mis_z", "pLI", "brain_expressed", "brain_enriched", "go_jaccard", "ppi_score",
             "ppi_ge_0.4", "ppi_ge_0.7", "ppi_ge_0.9", "coexpression"]
    p_bh = dict(zip(names, c.bh_adjust(raw_p)))

    # ── plot: row 1 = gene-level, row 2 = pair-level, same layout as Fig3 ──
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

    kw = dict(label_a=LABEL_BEFORE, label_b=LABEL_AFTER, color_a=BEFORE_COLOR, color_b=AFTER_COLOR)

    c.violin(ax_mis_z, before_gf["mis_z"].dropna(), after_gf["mis_z"].dropna(),
             "Missense z-score", "Missense constraint", **kw)
    c.annotate_bracket(ax_mis_z, before_gf["mis_z"].dropna().values, after_gf["mis_z"].dropna().values,
                        c.stars(p_bh["mis_z"]), p_bh["mis_z"])
    c.panel_label(ax_mis_z, "A")

    c.violin(ax_pli, before_gf["pLI"].dropna(), after_gf["pLI"].dropna(), "pLI", "LoF intolerance", **kw)
    c.annotate_bracket(ax_pli, before_gf["pLI"].dropna().values, after_gf["pLI"].dropna().values,
                        c.stars(p_bh["pLI"]), p_bh["pLI"])
    c.panel_label(ax_pli, "B")

    c.grouped_bar(ax_brain, ["Expressed", "Enriched"], [exp_b, enr_b], [exp_a, enr_a],
                  len(before_gf), len(after_gf), "% of genes", "Brain expression (HPA)",
                  [c.stars(p_bh["brain_expressed"]), c.stars(p_bh["brain_enriched"])],
                  [p_bh["brain_expressed"], p_bh["brain_enriched"]],
                  counts_a=[int(before_gf["brain_expressed"].sum()), int(before_gf["brain_enriched"].sum())],
                  counts_b=[int(after_gf["brain_expressed"].sum()), int(after_gf["brain_enriched"].sum())], **kw)
    c.panel_label(ax_brain, "C")

    c.violin(ax_go, before_pf["go_jaccard"], after_pf["go_jaccard"],
             "GO term Jaccard\n(BP+MF+CC, 3–500 genes/term)", "GO term overlap", **kw)
    c.annotate_bracket(ax_go, before_pf["go_jaccard"].values, after_pf["go_jaccard"].values,
                        c.stars(p_bh["go_jaccard"]), p_bh["go_jaccard"])
    c.panel_label(ax_go, "D")

    c.violin(ax_ppi, before_pf["ppi_score"], after_pf["ppi_score"],
             "STRING PPI score\n(missing edge = 0)", "Protein-protein interaction", **kw)
    c.annotate_bracket(ax_ppi, before_pf["ppi_score"].values, after_pf["ppi_score"].values,
                        c.stars(p_bh["ppi_score"]), p_bh["ppi_score"])
    c.panel_label(ax_ppi, "E")

    thresh_labels = ["≥ 0.4", "≥ 0.7", "≥ 0.9"]
    thresh_b = [r[0] for r in ppi_rows]
    thresh_a = [r[1] for r in ppi_rows]
    thresh_stars = [c.stars(p_bh[k]) for k in ("ppi_ge_0.4", "ppi_ge_0.7", "ppi_ge_0.9")]
    thresh_p = [p_bh[k] for k in ("ppi_ge_0.4", "ppi_ge_0.7", "ppi_ge_0.9")]
    thresh_counts_b = [int((before_pf["ppi_score"] >= t).sum()) for t in (0.4, 0.7, 0.9)]
    thresh_counts_a = [int((after_pf["ppi_score"] >= t).sum()) for t in (0.4, 0.7, 0.9)]
    c.grouped_bar(ax_ppi_thresh, thresh_labels, thresh_b, thresh_a,
                  len(before_pf), len(after_pf), "% gene pairs", "PPI score thresholds",
                  thresh_stars, thresh_p,
                  counts_a=thresh_counts_b, counts_b=thresh_counts_a, **kw)
    c.panel_label(ax_ppi_thresh, "F")

    c.grouped_bar(ax_coex, ["Same module"], [coex_b_pct], [coex_a_pct],
                  len(before_coexpr), len(after_coexpr), "% gene pairs",
                  "Coexpression\n(PsychENCODE INT-09)",
                  [c.stars(p_bh["coexpression"])], [p_bh["coexpression"]],
                  counts_a=[int(before_coexpr.sum())], counts_b=[int(after_coexpr.sum())], **kw)
    c.panel_label(ax_coex, "G")

    fig.suptitle(f"Case gene pairs before vs. after length-permutation correction "
                 f"({n_before} -> {n_after}, {100 * n_after / n_before:.0f}% retained)", fontsize=12, y=0.985)
    c.savefig(fig, "Fig5_permutation_before_after")
    c.save_panels(fig, [
        (ax_mis_z, "A"), (ax_pli, "B"), (ax_brain, "C"), (ax_go, "D"),
        (ax_ppi, "E"), (ax_ppi_thresh, "F"), (ax_coex, "G"),
    ], "Fig5_permutation_before_after")
    plt.close(fig)

    # ── source data ──────────────────────────────────────────────────────
    gene_wide = pd.concat([before_gf, after_gf], ignore_index=True)
    gene_long = gene_wide.melt(id_vars=["gene", "group"],
                                value_vars=["mis_z", "pLI", "brain_expressed", "brain_enriched"],
                                var_name="feature", value_name="value")
    gene_long.insert(0, "level", "gene")

    pair_wide = pd.concat([before_pf.assign(group="before"), after_pf.assign(group="after")], ignore_index=True)
    pair_wide["id"] = pair_wide["gene1"] + "," + pair_wide["gene2"]
    pair_long = pair_wide.melt(id_vars=["id", "group"], value_vars=["go_jaccard", "ppi_score", "coexpressed"],
                                var_name="feature", value_name="value")
    pair_long.insert(0, "level", "pair")
    pair_long = pair_long.rename(columns={"id": "gene"})

    full_long = pd.concat([gene_long, pair_long], ignore_index=True)
    data_path = c.FIGURES_DIR / "Fig5_permutation_before_after_data.tsv"
    full_long.to_csv(data_path, sep="\t", index=False)
    print(f"  saved {data_path}")

    stats_rows = [
        {"feature": "mis_z", "test": "mannwhitney_u", "statistic": t_mis_z[0],
         "n_before": t_mis_z[3], "n_after": t_mis_z[4], "p_raw": t_mis_z[1],
         "p_bh": p_bh["mis_z"], "effect_size": t_mis_z[2]},
        {"feature": "pLI", "test": "mannwhitney_u", "statistic": t_pli[0],
         "n_before": t_pli[3], "n_after": t_pli[4], "p_raw": t_pli[1],
         "p_bh": p_bh["pLI"], "effect_size": t_pli[2]},
        {"feature": "brain_expressed", "test": "fisher_exact", "statistic": np.nan,
         "n_before": len(before_gf), "n_after": len(after_gf), "p_raw": p_exp,
         "p_bh": p_bh["brain_expressed"], "effect_size": np.nan},
        {"feature": "brain_enriched", "test": "fisher_exact", "statistic": np.nan,
         "n_before": len(before_gf), "n_after": len(after_gf), "p_raw": p_enr,
         "p_bh": p_bh["brain_enriched"], "effect_size": np.nan},
        {"feature": "go_jaccard", "test": "mannwhitney_u", "statistic": t_go[0],
         "n_before": t_go[3], "n_after": t_go[4], "p_raw": t_go[1],
         "p_bh": p_bh["go_jaccard"], "effect_size": t_go[2]},
        {"feature": "ppi_score", "test": "mannwhitney_u", "statistic": t_ppi[0],
         "n_before": t_ppi[3], "n_after": t_ppi[4], "p_raw": t_ppi[1],
         "p_bh": p_bh["ppi_score"], "effect_size": t_ppi[2]},
        {"feature": "ppi_ge_0.4", "test": "fisher_exact", "statistic": np.nan,
         "n_before": len(before_pf), "n_after": len(after_pf), "p_raw": ppi_rows[0][3],
         "p_bh": p_bh["ppi_ge_0.4"], "effect_size": np.nan},
        {"feature": "ppi_ge_0.7", "test": "fisher_exact", "statistic": np.nan,
         "n_before": len(before_pf), "n_after": len(after_pf), "p_raw": ppi_rows[1][3],
         "p_bh": p_bh["ppi_ge_0.7"], "effect_size": np.nan},
        {"feature": "ppi_ge_0.9", "test": "fisher_exact", "statistic": np.nan,
         "n_before": len(before_pf), "n_after": len(after_pf), "p_raw": ppi_rows[2][3],
         "p_bh": p_bh["ppi_ge_0.9"], "effect_size": np.nan},
        {"feature": "coexpression", "test": "fisher_exact", "statistic": np.nan,
         "n_before": len(before_coexpr), "n_after": len(after_coexpr), "p_raw": p_coex,
         "p_bh": p_bh["coexpression"], "effect_size": np.nan},
    ]
    for row in stats_rows:
        row["stars"] = c.stars(row["p_bh"])
    stats_df = pd.DataFrame(stats_rows)
    stats_path = c.FIGURES_DIR / "Fig5_permutation_before_after_stats.tsv"
    stats_df.to_csv(stats_path, sep="\t", index=False)
    print(f"  saved {stats_path}")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
