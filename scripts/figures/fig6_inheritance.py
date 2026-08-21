"""
Figure 6 -- inheritance-category contribution per gene pair, cases vs.
controls. For each of the 6 origin categories a RareComb gene pair can be
resolved to:
  DD  both genes' qualifying variant arose de novo
  DM  one de novo, one maternally inherited
  DP  one de novo, one paternally inherited
  MP  one maternally inherited, one paternally inherited (i.e. a true
      compound-het-style trans configuration across the two genes)
  MM  both maternally inherited
  PP  both paternally inherited

Panel A shows why a naive raw-count comparison is misleading here: case
gene pairs simply have more TOTAL supporting individuals per gene pair
than control gene pairs (baseline, summed across all 6 categories) --
so *every* category's raw count comes out higher for cases even if the
relative mix of inheritance types is identical. Panel B is that naive raw
comparison, included for transparency. Panel C is the corrected comparison:
each gene pair's supporting individuals are turned into a PROPORTION
across the 6 categories first (so a gene pair's own baseline is divided
out), then proportions are compared, cases vs. controls -- this is what
actually answers "does the inheritance-pattern *composition* differ",
independent of panel A's baseline gap.

Uses the raw origin-category assignment from annotate_inheritance.py's
build_origin_instances() for BOTH directions (case_value="asd" for the case
output, "sib" for the reversed control output) -- not its downstream
flag_selection() step, which hardcodes a proband-only regex and an
"affected-parent" validity rule that has no clean sibling/control analogue
(see _common.py::build_inheritance_origin_table's docstring). So this is the
raw category distribution, not the case pipeline's "valid support" subset.

Usage: python fig6_inheritance.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import _common as c

CATEGORY_LABELS = {"DD": "DD\n(both de novo)", "DM": "DM\n(de novo + mat.)",
                    "DP": "DP\n(de novo + pat.)", "MP": "MP\n(mat. + pat.)",
                    "MM": "MM\n(both maternal)", "PP": "PP\n(both paternal)"}


def per_combination_counts(stats_df: pd.DataFrame, rarecomb_output_path, case_value: str) -> pd.DataFrame:
    instances = c.build_inheritance_origin_table(rarecomb_output_path, case_value)
    counts = instances.groupby("combination")[c.ORIGIN_COLS].sum()
    combos = pd.DataFrame({"combination": [",".join(sorted([a, b]))
                                            for a, b in zip(stats_df.gene1, stats_df.gene2)]})
    out = combos.merge(counts, on="combination", how="left").fillna(0)
    out[c.ORIGIN_COLS] = out[c.ORIGIN_COLS].astype(int)
    out["total_support"] = out[c.ORIGIN_COLS].sum(axis=1)
    return out


def category_tests(case_counts: pd.DataFrame, control_counts: pd.DataFrame, columns) -> pd.DataFrame:
    rows = []
    for cat in c.ORIGIN_COLS:
        a, b = case_counts[cat].values, control_counts[cat].values
        U, p, r, n_a, n_b = c.mannwhitney(a, b)
        rows.append({"category": cat, f"mean_case_{columns}": a.mean(),
                     f"sem_case_{columns}": a.std(ddof=1) / np.sqrt(len(a)),
                     f"mean_control_{columns}": b.mean(),
                     f"sem_control_{columns}": b.std(ddof=1) / np.sqrt(len(b)),
                     "mannwhitney_U": U, "p_raw": p, "effect_size": r, "n_case": n_a, "n_control": n_b})
    return pd.DataFrame(rows)


def bar_with_error(ax, stats_df, mean_col_case, sem_col_case, mean_col_control, sem_col_control,
                    ylabel, title, n_case, n_control):
    x = np.arange(len(c.ORIGIN_COLS))
    w = 0.36
    ax.bar(x - w / 2, stats_df[mean_col_case], width=w, yerr=stats_df[sem_col_case],
           color=c.CASE_COLOR, label=f"Cases (n={n_case:,} gene pairs)", capsize=3)
    ax.bar(x + w / 2, stats_df[mean_col_control], width=w, yerr=stats_df[sem_col_control],
           color=c.CONTROL_COLOR, label=f"Controls (n={n_control:,} gene pairs)", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABELS[cat] for cat in c.ORIGIN_COLS], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10.5)
    ax.legend(fontsize=7.3, frameon=False, loc="upper left")
    y_top = (stats_df[mean_col_case] + stats_df[sem_col_case]).combine(
        stats_df[mean_col_control] + stats_df[sem_col_control], max)
    ax.set_ylim(0, y_top.max() * 1.85)
    return x, w, y_top


def main():
    c.set_style()

    case_df = c.load_stats(str(c.CASE_OUTPUT))
    control_df = c.load_stats(str(c.CONTROL_OUTPUT))

    print("Case gene pairs ...")
    case_counts = per_combination_counts(case_df, c.CASE_OUTPUT, "asd")
    print("Control gene pairs ...")
    control_counts = per_combination_counts(control_df, c.CONTROL_OUTPUT, "sib")

    n_case, n_control = len(case_counts), len(control_counts)

    # ── Panel A: the baseline confound, made explicit ───────────────────────
    t_baseline = c.mannwhitney(case_counts["total_support"], control_counts["total_support"])
    print(f"Baseline total support -- cases mean={case_counts['total_support'].mean():.2f}, "
          f"controls mean={control_counts['total_support'].mean():.2f}, "
          f"Mann-Whitney p={t_baseline[1]:.3g}")

    # ── Panel B: naive raw counts (kept for transparency) ───────────────────
    raw_stats = category_tests(case_counts, control_counts, "raw")
    raw_stats["p_bh"] = c.bh_adjust(raw_stats["p_raw"].values)
    raw_stats["stars"] = raw_stats["p_bh"].apply(c.stars)

    # ── Panel C: proportion of each gene pair's OWN support per category --
    # divides out the baseline gap from panel A, so this is the comparison
    # that actually speaks to whether the inheritance-pattern *mix* differs ──
    case_props = case_counts[c.ORIGIN_COLS].div(case_counts["total_support"], axis=0)
    control_props = control_counts[c.ORIGIN_COLS].div(control_counts["total_support"], axis=0)
    prop_stats = category_tests(case_props, control_props, "prop")
    prop_stats["p_bh"] = c.bh_adjust(prop_stats["p_raw"].values)
    prop_stats["stars"] = prop_stats["p_bh"].apply(c.stars)

    print("\n-- raw counts --")
    print(raw_stats[["category", "mean_case_raw", "mean_control_raw", "p_bh", "stars"]].to_string(index=False))
    print("\n-- proportions (baseline-corrected) --")
    print(prop_stats[["category", "mean_case_prop", "mean_control_prop", "p_bh", "stars"]].to_string(index=False))

    # ── plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw={"width_ratios": [1, 2, 2]},
                              constrained_layout=True)

    c.violin(axes[0], case_counts["total_support"].values, control_counts["total_support"].values,
             "Total supporting individuals\nper gene pair (all 6 categories summed)",
             "Baseline support")
    c.annotate_bracket(axes[0], case_counts["total_support"].values, control_counts["total_support"].values,
                        c.stars(t_baseline[1]), t_baseline[1])
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    c.panel_label(axes[0], "A")

    x, w, y_top = bar_with_error(axes[1], raw_stats, "mean_case_raw", "sem_case_raw",
                                  "mean_control_raw", "sem_control_raw",
                                  "Mean supporting individuals\nper gene pair",
                                  "Raw count per category", n_case, n_control)
    for xi, row, top in zip(x, raw_stats.itertuples(), y_top):
        c.draw_sig_bracket(axes[1], xi - w / 2, xi + w / 2, top * 1.08, row.stars, pvalue=row.p_bh,
                            delta=row.mean_case_raw - row.mean_control_raw, test_name="Mann-Whitney U")
    c.panel_label(axes[1], "B")

    x, w, y_top = bar_with_error(axes[2], prop_stats, "mean_case_prop", "sem_case_prop",
                                  "mean_control_prop", "sem_control_prop",
                                  "Mean % of a gene pair's own\nsupport in this category",
                                  "Proportion per category", n_case, n_control)
    for xi, row, top in zip(x, prop_stats.itertuples(), y_top):
        c.draw_sig_bracket(axes[2], xi - w / 2, xi + w / 2, top * 1.08, row.stars, pvalue=row.p_bh,
                            delta=row.mean_case_prop - row.mean_control_prop, test_name="Mann-Whitney U")
    c.panel_label(axes[2], "C")

    fig.suptitle("Inheritance-category contribution per gene pair", fontsize=12)
    c.savefig(fig, "Fig6_inheritance_contribution")
    c.save_panels(fig, axes, "Fig6_inheritance_contribution")
    plt.close(fig)

    raw_stats["level"] = "raw_count"
    prop_stats["level"] = "proportion"
    baseline_row = pd.DataFrame([{"category": "TOTAL_SUPPORT", "level": "baseline",
                                   "mean_case": case_counts["total_support"].mean(),
                                   "mean_control": control_counts["total_support"].mean(),
                                   "mannwhitney_U": t_baseline[0], "p_raw": t_baseline[1],
                                   "p_bh": t_baseline[1], "effect_size": t_baseline[2],
                                   "n_case": t_baseline[3], "n_control": t_baseline[4],
                                   "stars": c.stars(t_baseline[1])}])
    out_cols = ["level", "category", "mean_case", "mean_control",
                "mannwhitney_U", "p_raw", "p_bh", "effect_size", "n_case", "n_control", "stars"]
    combined = pd.concat([
        baseline_row[out_cols],
        raw_stats.rename(columns={"mean_case_raw": "mean_case", "mean_control_raw": "mean_control"})[out_cols],
        prop_stats.rename(columns={"mean_case_prop": "mean_case", "mean_control_prop": "mean_control"})[out_cols],
    ], ignore_index=True)
    out_path = c.FIGURES_DIR / "Fig6_inheritance_contribution_stats.tsv"
    combined.to_csv(out_path, sep="\t", index=False)
    print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
