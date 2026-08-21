"""
Load a raw variant dataset, add the columns the rest of the
pipeline depends on (spid, consequence, pheno), and save:

  1. the parsed dataset itself (same rows, with the new columns), this is the file every later step is built from;
  2. a one-row summary of the initial input: n_individuals, n_cases,
     n_controls, n_genes, n_variants, n_lgd, n_mis, n_piv, n_dnv

Usage:
    python core/parse_dataset.py CADD_anno/LGDMIS_MZmark_DNM_PIV_com_noMZ_CADD.txt \
        --dataset-out results_221/parsed_dataset.tsv --summary-out results_221/input_summary.tsv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io_utils import load_spark, assign_phenotype


def build_dataset(raw_path: str):
    df = load_spark(raw_path)
    df = assign_phenotype(df)
    return df


def summarize(df) -> dict:
    return {
        "n_variants":    len(df),
        "n_individuals": df["spid"].nunique(),
        "n_cases":       df.loc[df["pheno"] == "asd", "spid"].nunique(),
        "n_controls":    df.loc[df["pheno"] == "sib", "spid"].nunique(),
        "n_genes":       df["gene"].nunique(),
        "n_lgd":         int((df["consequence"] == "LGD").sum()),
        "n_mis":         int((df["consequence"] == "MIS").sum()),
        "n_piv":         int((df["type"] == "PIV").sum()) if "type" in df.columns else 0,
        "n_dnv":         int((df["type"] == "DNV").sum()) if "type" in df.columns else 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw_file", help="Raw SPARK-style variant TSV")
    parser.add_argument("--dataset-out", default="parsed_dataset.tsv")
    parser.add_argument("--summary-out", default="input_summary.tsv")
    args = parser.parse_args()

    print(f"Loading {args.raw_file} ...")
    df = build_dataset(args.raw_file)
    print(f"  {len(df):,} variants  |  {df['spid'].nunique():,} individuals")

    df.to_csv(args.dataset_out, sep="\t", index=False)
    print(f"Saved parsed dataset: {args.dataset_out}")

    summary = summarize(df)
    import pandas as pd
    pd.DataFrame([summary]).to_csv(args.summary_out, sep="\t", index=False)
    print(f"Saved input summary: {args.summary_out}")
    for k, v in summary.items():
        print(f"  {k}: {v:,}")


if __name__ == "__main__":
    main()
