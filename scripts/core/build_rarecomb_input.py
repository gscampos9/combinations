"""
Build the RareComb boolean input matrix from the parsed dataset: filter to rare damaging variants (LGD, or MIS with
CADD >= --cadd), drop genes carried by too few individuals to ever appear
in a combination, and pivot to one row per individual with `Input_<gene>`
0/1 columns and an `Output_1` phenotype column (1 = ASD proband, 0 = sibling).

Usage:
    python core/build_rarecomb_input.py results_221/parsed_dataset.tsv results_221/CADD20_2_2_1_input.txt \
        --cadd 20 --min-carriers 2
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io_utils import filter_rrvs  # noqa: E402


def build_rarecomb_input(df: pd.DataFrame, cadd_cutoff: float, min_carriers: int) -> pd.DataFrame:
    rrv = filter_rrvs(df, cadd_cutoff)
    print(f"  RRV filter (LGD or MIS CADD>={cadd_cutoff}): {len(df):,} -> {len(rrv):,} variants")

    rrv = rrv[rrv["pheno"].isin(["asd", "sib"])].copy()

    rrv["present"] = 1
    gene_matrix = (
        rrv[["spid", "gene", "present"]]
        .pivot_table(index="spid", columns="gene", values="present", aggfunc="max", fill_value=0)
    )

    before = gene_matrix.shape[1]
    freq = gene_matrix.sum(axis=0)
    gene_matrix = gene_matrix.loc[:, freq >= min_carriers]
    after = gene_matrix.shape[1]
    print(f"  Gene filter (min_carriers={min_carriers}): {before} -> {after} genes retained")

    gene_matrix = gene_matrix.add_prefix("Input_")

    pheno_map = rrv[["spid", "pheno"]].drop_duplicates().set_index("spid")
    pheno_map["Output_1"] = pheno_map["pheno"].map({"asd": 1, "sib": 0})

    rc_df = gene_matrix.join(pheno_map[["Output_1"]], how="inner").fillna(0)
    rc_df = rc_df.reset_index().rename(columns={"spid": "Sample_Name"})
    return rc_df


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="parsed_dataset.tsv from 01_build_dataset.py")
    parser.add_argument("output")
    parser.add_argument("--cadd", type=float, default=30.0, help="CADD cutoff for MIS variants (default: 30)")
    parser.add_argument("--min-carriers", type=int, default=3,
                        help="Drop genes carried by fewer than this many individuals (default: 3)")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset, sep="\t", low_memory=False)
    rc_df = build_rarecomb_input(df, args.cadd, args.min_carriers)

    n_asd = int((rc_df["Output_1"] == 1).sum())
    n_sib = int((rc_df["Output_1"] == 0).sum())
    n_genes = rc_df.shape[1] - 2

    rc_df.to_csv(args.output, sep="\t", index=False)
    print(f"Saved RareComb input: {args.output}")
    print(f"  ASD={n_asd}  SIB={n_sib}  genes={n_genes}")


if __name__ == "__main__":
    main()
