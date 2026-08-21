"""
Comprehensive per-gene-pair annotation table, cases and controls: RareComb
stats + gene-pair features (GO Jaccard, STRING PPI, PsychENCODE coexpression)
+ per-gene database hits (SFARI, NDD, G2P_DD, ClinGen, HPA brain
expressed/enriched, gnomAD CDS length/pLI/missense z) + inheritance origin
category counts (DD/DM/DP/MP/MM/PP).

Length-permutation results are deliberately NOT annotated onto these tables
-- they're specific to Fig5 (permutation_before_after), which loads them
itself via _common.load_permutation_results(); baking perm_z/p/q into the
general-purpose annotated_gene_pairs_cases.tsv isn't needed and just adds a
simulation/ dependency this table otherwise doesn't have.

This is a supplementary/reference table, not something the other figure
scripts read back in -- each figure pulls the specific pieces it needs
directly from _common.py so no figure depends on this script having run
first. Run standalone or via run_all.py.

Usage: python build_annotations.py
"""
from __future__ import annotations

import pandas as pd

import _common as c


def _combo_key(g1, g2) -> str:
    return ",".join(sorted([g1, g2]))


def _gene_side_columns(df: pd.DataFrame, gene_col: str, db: pd.DataFrame, gene_table: pd.DataFrame, prefix: str) -> pd.DataFrame:
    genes = df[gene_col]
    db_side = db.reindex(genes).reset_index(drop=True).add_prefix(f"{prefix}_")
    gt_side = gene_table.reindex(genes).reset_index(drop=True).add_prefix(f"{prefix}_")
    return pd.concat([db_side, gt_side], axis=1)


def annotate_group(stats_df: pd.DataFrame, rarecomb_output_path, case_value: str,
                    go_index: dict, ppi: dict, modules: dict) -> pd.DataFrame:
    df = stats_df.reset_index(drop=True).copy()
    df["combination"] = [_combo_key(a, b) for a, b in zip(df.gene1, df.gene2)]

    # pair-level features
    jacc, ppi_val, coexpr = [], [], []
    for g1, g2 in zip(df.gene1, df.gene2):
        _shared, j = c._shared_and_jaccard(g1, g2, go_index)
        jacc.append(j)
        ppi_val.append(c.ppi_score(g1, g2, ppi) / 1000.0)
        coexpr.append(c.coexpressed_pair(g1, g2, modules))
    df["go_jaccard"] = jacc
    df["ppi_score"] = ppi_val
    df["coexpressed_module"] = coexpr

    # per-gene database + constraint annotations, for both sides of the pair
    all_genes = sorted(set(df.gene1) | set(df.gene2))
    db = c.build_database_hits(all_genes).set_index("gene")
    gene_table = c.build_gene_table(all_genes).set_index("gene")
    df = pd.concat([
        df,
        _gene_side_columns(df, "gene1", db, gene_table, "gene1"),
        _gene_side_columns(df, "gene2", db, gene_table, "gene2"),
    ], axis=1)

    # inheritance: per-gene-pair sum of raw origin-category instances
    print(f"  Building inheritance origin table (case_value={case_value!r}) ...")
    inh = c.build_inheritance_origin_table(rarecomb_output_path, case_value)
    inh_counts = inh.groupby("combination")[c.ORIGIN_COLS].sum().add_prefix("inh_")
    df = df.merge(inh_counts, on="combination", how="left")
    for col in c.ORIGIN_COLS:
        df[f"inh_{col}"] = df[f"inh_{col}"].fillna(0).astype(int)

    return df


def main():
    case_df = c.load_stats(str(c.CASE_OUTPUT))
    control_df = c.load_stats(str(c.CONTROL_OUTPUT))

    all_genes = sorted(set(case_df.gene1) | set(case_df.gene2) | set(control_df.gene1) | set(control_df.gene2))
    go_index = c.build_go_term_index(min_genes=3, max_genes=500)
    ppi = c.load_ppi_for_genes(all_genes)
    modules = c.load_coexpression_modules()

    print("Annotating case gene pairs ...")
    case_ann = annotate_group(case_df, c.CASE_OUTPUT, "asd", go_index, ppi, modules)
    case_path = c.FIGURES_DIR / "annotated_gene_pairs_cases.tsv"
    case_ann.to_csv(case_path, sep="\t", index=False)
    print(f"  saved {case_path}  ({len(case_ann)} gene pairs, {len(case_ann.columns)} columns)")

    print("Annotating control gene pairs ...")
    control_ann = annotate_group(control_df, c.CONTROL_OUTPUT, "sib", go_index, ppi, modules)
    control_path = c.FIGURES_DIR / "annotated_gene_pairs_controls.tsv"
    control_ann.to_csv(control_path, sep="\t", index=False)
    print(f"  saved {control_path}  ({len(control_ann)} gene pairs, {len(control_ann.columns)} columns)")

    return case_ann, control_ann


if __name__ == "__main__":
    main()
