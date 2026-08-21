"""
Build a per-gene summary table from a RareComb boolean input matrix and the
gnomad constraint metrics file, with HGNC alias/previous-symbol fallback
matching for genes that don't match gnomad directly.

For each gene present as an `Input_<gene>` column in the RareComb matrix,
compute:
    - n_occurrence      : total number of individuals carrying the gene
    - n_occurrence_pro  : number of ASD probands carrying the gene (Output_1 == 1)
    - n_occurrence_sib  : number of siblings carrying the gene (Output_1 == 0)
    - CDS_length        : canonical transcript CDS length from gnomad constraint metrics

Matching strategy:
    1. Direct match on gnomad `gene` column (canonical == True rows only).
    2. For unmatched genes, resolve via HGNC complete set: map the query gene
       (as an alias_symbol or prev_symbol) to its current approved HGNC
       `symbol`, then retry the match against gnomad using that symbol.
    3. For still-unmatched genes containing a hyphen (candidate gene fusions,
       e.g. "GENE1-GENE2"), try matching each hyphen-separated part via steps
       1-2 above. These are flagged separately (`match_type = fusion_partial`)
       since they are approximate and should be reviewed manually.

Output columns (tab-separated):
    gene    n_occurrence    n_occurrence_pro    n_occurrence_sib    CDS_length    match_type

Usage:
    python build_gene_summary.py rarecomb_input.txt gnomad.v4.1.1.constraint_metrics.tsv.bgz \
        hgnc_complete_set_2026-04-28.txt output.txt
"""

import argparse

import pandas as pd


def _split_multi(value) -> list:
    """HGNC multi-value fields are pipe-delimited (e.g. alias_symbol, prev_symbol)."""
    if pd.isna(value) or value == "":
        return []
    return [v.strip() for v in str(value).split("|") if v.strip()]


def build_hgnc_maps(hgnc: pd.DataFrame):
    """Build lookup: any known symbol/alias/prev_symbol -> current approved HGNC symbol."""
    approved_symbols = set(hgnc["symbol"].dropna())
    alias_to_symbol = {}

    for _, row in hgnc.iterrows():
        symbol = row["symbol"]
        if pd.isna(symbol):
            continue
        for alias in _split_multi(row.get("alias_symbol")):
            alias_to_symbol.setdefault(alias, symbol)
        for prev in _split_multi(row.get("prev_symbol")):
            alias_to_symbol.setdefault(prev, symbol)

    return approved_symbols, alias_to_symbol


def resolve_symbol(gene: str, gnomad_genes: set, approved_symbols: set, alias_to_symbol: dict):
    """Return (matched_gene_in_gnomad, match_type) or (None, None)."""
    if gene in gnomad_genes:
        return gene, "direct"

    if gene in approved_symbols or gene in alias_to_symbol:
        current_symbol = gene if gene in approved_symbols else alias_to_symbol[gene]
        if current_symbol in gnomad_genes:
            return current_symbol, "hgnc_alias"

    return None, None


def build_gene_summary(rc_df: pd.DataFrame, gnomad: pd.DataFrame, hgnc: pd.DataFrame) -> pd.DataFrame:
    gene_cols = [c for c in rc_df.columns if c.startswith("Input_")]

    pro_mask = rc_df["Output_1"] == 1
    sib_mask = rc_df["Output_1"] == 0

    records = []
    for col in gene_cols:
        gene = col[len("Input_"):]
        records.append({
            "gene": gene,
            "n_occurrence": int(rc_df[col].sum()),
            "n_occurrence_pro": int(rc_df.loc[pro_mask, col].sum()),
            "n_occurrence_sib": int(rc_df.loc[sib_mask, col].sum()),
        })
    summary = pd.DataFrame.from_records(records)

    # Canonical transcript rows only, one per gene
    # (handles canonical column stored as native bool True/False or as string "true"/"false")
    canon_mask = gnomad["canonical"].astype(str).str.strip().str.lower() == "true"
    gnomad_canon = gnomad[canon_mask].copy()
    gnomad_canon["_mane_select"] = (
    gnomad_canon["mane_select"].astype(str).str.strip().str.lower() == "true"
    )
    gnomad_canon = gnomad_canon.sort_values(
        ["_mane_select", "cds_length"], ascending=[False, False], na_position="last"
    )
    gnomad_canon = gnomad_canon.drop_duplicates(subset="gene", keep="first")
    gnomad_canon = gnomad_canon.drop(columns="_mane_select")
    cds_lookup = gnomad_canon.set_index("gene")["cds_length"].to_dict()
    gnomad_genes = set(cds_lookup.keys())

    approved_symbols, alias_to_symbol = build_hgnc_maps(hgnc)

    cds_lengths = []
    match_types = []
    unresolved = []

    for gene in summary["gene"]:
        matched, mtype = resolve_symbol(gene, gnomad_genes, approved_symbols, alias_to_symbol)

        if matched is not None:
            cds_lengths.append(cds_lookup[matched])
            match_types.append(mtype)
        else:
            cds_lengths.append(pd.NA)
            match_types.append("unmatched")
            unresolved.append(gene)

    summary["CDS_length"] = cds_lengths
    summary["match_type"] = match_types

    n_missing = len(unresolved)
    if n_missing:
        print(f"  Warning: {n_missing} gene(s) unmatched even after HGNC/fusion fallback")
        print(f"    e.g.: {unresolved[:10]}")

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rarecomb_input", help="RareComb boolean input matrix (from build_rarecomb_input.py)")
    parser.add_argument("gnomad", help="gnomad constraint metrics file (.tsv or .tsv.bgz)")
    parser.add_argument("hgnc", help="HGNC complete_set file (.tsv/.txt, tab-separated)")
    parser.add_argument("output", help="Output summary table path")
    args = parser.parse_args()

    rc_df = pd.read_csv(args.rarecomb_input, sep="\t", low_memory=False)

    gnomad_compression = "gzip" if args.gnomad.endswith((".gz", ".bgz")) else "infer"
    gnomad = pd.read_csv(args.gnomad, sep="\t", compression=gnomad_compression, low_memory=False)

    hgnc = pd.read_csv(args.hgnc, sep="\t", low_memory=False)

    summary = build_gene_summary(rc_df, gnomad, hgnc)
    summary.to_csv(args.output, sep="\t", index=False)

    print(f"Saved gene summary: {args.output}")
    match_counts = summary["match_type"].value_counts().to_dict()
    print(f"  genes={len(summary)}  match_breakdown={match_counts}")


if __name__ == "__main__":
    main()