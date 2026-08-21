"""
Figure 7 -- pathway/GO enrichment of the 3 gene membership groups from
Figure 4: genes found only in case gene pairs, genes shared between case and
control gene pairs, and genes found only in control gene pairs. Biological
interpretation of what distinguishes (or doesn't) these 3 populations.

Hypergeometric over-representation test, GO (BP+MF+CC) + Reactome combined
into one term universe (3-500 genes/term, same convention as Figure 3's
Jaccard), each group tested against the whole RareComb gene universe
(18,256 genes) as background, BH-adjusted separately per panel.

Usage: python fig7_pathway_enrichment.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import _common as c

TOP_N = 15


def _short_name(name: str, width: int = 55) -> str:
    return name if len(name) <= width else name[:width - 1] + "…"


def enrichment_bar(ax, result: pd.DataFrame, title: str):
    top = result.head(TOP_N).iloc[::-1]  # smallest q at the top of the plot
    y = np.arange(len(top))
    nlq = -np.log10(top["p_bh"].clip(lower=1e-300)) if len(top) else np.array([])
    ax.barh(y, nlq, color=c.CASE_COLOR)
    ax.set_yticks(y)
    labels = [f"{_short_name(t)}  [{s.replace('GO:', '')}]" for t, s in zip(top["term_name"], top["source"])]
    ax.set_yticklabels(labels, fontsize=7.5)
    for yi, q, k in zip(y, nlq, top["n_query_hits"]):
        ax.text(q + (nlq.max() * 0.015 if len(nlq) else 0.02), yi, f"n={k}", va="center", fontsize=7)
    ax.axvline(-np.log10(0.05), color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("-log$_{10}$(BH-adjusted p)")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, nlq.max() * 1.18 if len(nlq) else 1)
    if not len(top):
        ax.text(0.5, 0.5, "no terms reach q<0.05", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, style="italic")


def main():
    c.set_style()

    case_df = c.load_stats(str(c.CASE_OUTPUT))
    control_df = c.load_stats(str(c.CONTROL_OUTPUT))
    case_genes = set(case_df.gene1) | set(case_df.gene2)
    control_genes = set(control_df.gene1) | set(control_df.gene2)

    case_only = case_genes - control_genes
    control_only = control_genes - case_genes
    shared = case_genes & control_genes
    print(f"Case-only genes: {len(case_only)}  Shared: {len(shared)}  Control-only: {len(control_only)}")

    with open(c.INPUT_MATRIX, "r", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
    genome_genes = {g[len("Input_"):] for g in header[1:-1]}
    print(f"Genome (RareComb matrix) background: {len(genome_genes)} genes")

    groups = [("case_only", "Case-only genes", case_only),
              ("shared", "Shared genes", shared),
              ("control_only", "Control-only genes", control_only)]

    results = {}
    for key, label, genes in groups:
        print(f"Running enrichment: {label} (n={len(genes)}) vs. genome background ...")
        res = c.pathway_enrichment(genes, genome_genes)
        n_sig = int((res["p_bh"] < 0.05).sum()) if len(res) else 0
        print(f"  {len(res)} terms testable, {n_sig} significant at q<0.05")
        results[key] = res

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.4), constrained_layout=True)
    for ax, letter, (key, label, genes) in zip(axes, "ABC", groups):
        enrichment_bar(ax, results[key], f"{label} (n={len(genes)})\nvs. genome background")
        c.panel_label(ax, letter)

    fig.suptitle("Pathway/GO enrichment by gene-pair membership group", fontsize=12)
    c.savefig(fig, "Fig7_pathway_enrichment")
    c.save_panels(fig, zip(axes, "ABC"), "Fig7_pathway_enrichment")
    plt.close(fig)

    for key, label, genes in groups:
        out_path = c.FIGURES_DIR / f"Fig7_pathway_enrichment_{key}.tsv"
        results[key].to_csv(out_path, sep="\t", index=False)
        print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
