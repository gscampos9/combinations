"""
Single entry point: regenerates every publication figure plus the
comprehensive annotation table in one process, so the expensive shared
loads (877MB RareComb input matrix, gnomAD constraint metrics, STRING PPI,
parsed_dataset.tsv for inheritance) are each fetched/parsed once and reused
across scripts instead of once per script.

Run build_final_annotated_table.py separately for the prioritized-gene
supplementary workbook -- it's a standalone script by design (meant to run
on a bare results_221/+databases/ checkout, e.g. the cluster, without
depending on this script having run first).

Usage: python run_all.py
"""
from __future__ import annotations

import time

import fig1_variant_count
import fig2_gene_length
import fig3_feature_comparison
import fig4_overlap
import fig5_permutation_before_after
import fig6_inheritance
import fig7_pathway_enrichment
import fig8_ndd_genes
import build_annotations


def main():
    t0 = time.time()

    print("\n=== Figure 1: variant count ===")
    fig1_variant_count.main()

    print("\n=== Figure 2: gene length effect ===")
    fig2_gene_length.main()

    print("\n=== Figure 3: feature comparison ===")
    fig3_feature_comparison.main()

    print("\n=== Figure 4: case/control overlap ===")
    fig4_overlap.main()

    print("\n=== Figure 5: permutation before/after ===")
    fig5_permutation_before_after.main()

    print("\n=== Figure 6: inheritance contribution ===")
    fig6_inheritance.main()

    print("\n=== Figure 7: pathway enrichment (case-only / shared / control-only genes) ===")
    fig7_pathway_enrichment.main()

    print("\n=== Figure 8: NDD genes per pair ===")
    fig8_ndd_genes.main()

    print("\n=== Comprehensive annotation tables ===")
    build_annotations.main()

    print(f"\nAll figures and tables regenerated in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
