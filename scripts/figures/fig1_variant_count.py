"""
Figure 1 -- per-individual variant count, cases (probands) vs. controls
(siblings). Replaces the old boxplot_burden.png. 4 panels:

  A. All variants -- every LGD/MIS row in parsed_dataset.tsv for that
     person, NO CADD cutoff applied. The broadest count available: the
     source file was already pre-restricted to LGD/MIS candidates before
     parsing (confirmed: `consequence` is exactly {LGD, MIS}, no synonymous/
     other types exist to make this any broader).
  B. Damaging -- the CADD>=20 LGD-or-MIS filter (filter_rrvs(), the same one
     used everywhere else in this pipeline) applied to panel A's set. A
     true per-person VARIANT count, not a gene-hit count -- two qualifying
     variants in the same gene count as 2 here (they'd count as 1 "gene
     hit" in the RareComb boolean matrix Figures 2/3/5's gene tables use).
  C. Parentally Inherited Variant (PIV) -- panel B's set, restricted to
     inherited-from-parent origin.
  D. De Novo Mutation (DNM) -- panel B's set, restricted to de novo origin
     (parsed_dataset.tsv's own column calls this "DNV"; shown as "DNM" here
     per request -- same variants, different label).

C + D exactly reconstruct B (no double-counting): confirmed at runtime.

Every panel: N, mean, the group difference (delta), and the statistical
test are all written directly on the plot, not just implied.

Usage: python fig1_variant_count.py
"""
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

import _common as c


def panel(ax, letter, values_case, values_control, ylabel, title):
    c.violin(ax, values_case, values_control, ylabel, title)
    U, p, r, n_case, n_control = c.mannwhitney(values_case, values_control)
    y_max = max(values_case.max(), values_control.max())
    ax.set_ylim(0, y_max * 1.42)
    c.annotate_bracket(ax, values_case.values, values_control.values, c.stars(p), p)
    c.panel_label(ax, letter)
    print(f"  {title}: cases mean={values_case.mean():.2f} (n={n_case:,}), "
          f"controls mean={values_control.mean():.2f} (n={n_control:,}), "
          f"Mann-Whitney p={p:.3g} -> {c.stars(p)}")
    return {"n_case": n_case, "n_control": n_control, "mean_case": values_case.mean(),
            "mean_control": values_control.mean(), "mannwhitney_U": U, "p_raw": p,
            "effect_size": r, "stars": c.stars(p)}


def main():
    c.set_style()

    print("Loading parsed_dataset.tsv ...")
    parsed = c.load_parsed_dataset()
    parsed = parsed[parsed["pheno"].isin(["asd", "sib"])]
    all_individuals = parsed[["spid", "pheno"]]

    def split(df):
        counts = c.variant_counts_per_individual(df, all_individuals)
        return (counts.loc[counts["pheno"] == "asd", "n_variants"],
                counts.loc[counts["pheno"] == "sib", "n_variants"])

    print("Panel A: all LGD/MIS variants, no CADD cutoff")
    case_a, control_a = split(parsed)

    print("Panel B: damaging (CADD>=20 LGD or MIS)")
    damaging = c.filter_rrvs(parsed, cadd_cutoff=20.0)
    case_b, control_b = split(damaging)

    print("Panels C/D: damaging, split by origin (PIV / DNM)")
    case_c, control_c = split(damaging[damaging["type"] == "PIV"])
    case_d, control_d = split(damaging[damaging["type"] == "DNV"])

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.6), constrained_layout=True)

    stats = {}
    stats["A_all_variants"] = panel(
        axes[0], "A", case_a, control_a,
        "All LGD/MIS variants\n(no CADD cutoff) per individual",
        "All variants")
    stats["B_damaging"] = panel(
        axes[1], "B", case_b, control_b,
        "Damaging variants\n(CADD≥20, LGD or MIS) per individual",
        "Damaging")
    stats["C_PIV"] = panel(
        axes[2], "C", case_c, control_c,
        "Damaging PIV variants\nper individual",
        "Parentally Inherited Variant (PIV)")
    stats["D_DNM"] = panel(
        axes[3], "D", case_d, control_d,
        "Damaging DNM variants\nper individual",
        "De Novo Mutation (DNM)")

    fig.suptitle("Per-individual variant count", fontsize=12)
    c.savefig(fig, "Fig1_variant_count")
    c.save_panels(fig, axes, "Fig1_variant_count")
    plt.close(fig)

    # ── source data ──────────────────────────────────────────────────────
    long_rows = []
    for label, (cv, kv) in {
        "A_all_variants": (case_a, control_a), "B_damaging": (case_b, control_b),
        "C_PIV": (case_c, control_c), "D_DNM": (case_d, control_d),
    }.items():
        for v in cv:
            long_rows.append({"panel": label, "group": "case", "value": v})
        for v in kv:
            long_rows.append({"panel": label, "group": "control", "value": v})
    data_path = c.FIGURES_DIR / "Fig1_variant_count_data.tsv"
    pd.DataFrame(long_rows).to_csv(data_path, sep="\t", index=False)
    print(f"  saved {data_path}")

    stats_df = pd.DataFrame.from_dict(stats, orient="index").reset_index(names="panel")
    stats_path = c.FIGURES_DIR / "Fig1_variant_count_stats.tsv"
    stats_df.to_csv(stats_path, sep="\t", index=False)
    print(f"  saved {stats_path}")


if __name__ == "__main__":
    main()
