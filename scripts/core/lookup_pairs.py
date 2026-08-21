"""
For a given gene pair (order doesn't matter -- gene1/gene2 in the
annotated tables are Item_1/Item_2 from RareComb, not alphabetical), find
every row matching that combination in the annotated case/control gene-pair
tables, and print each carrying family's full set of variants in the two
genes, pulled from results_221/parsed_dataset.tsv.

For every matching row, EVERY individual in that row's Case_Samples and
Control_Samples is included (not just siblings) -- so this shows the full
carrier table for the combination, cases and controls alike, plus every
other family member sharing that family_id (so you can see who in the
family does and doesn't carry each variant).

Source tables, in order of preference (whichever exists is used, no need to
pass anything if you're running from anywhere inside the repo checkout):
  1. figures/annotated_gene_pairs_cases.tsv / _controls.tsv
     (build_annotations.py's output)
  2. figures/Gene_pairs_SPARK_annotated.xlsx, sheets enriched_in_pro /
     enriched_in_sib (build_final_annotated_table.py's output)
Override either with --annotated/--annotated-controls (a .tsv) or
--annotated-xlsx (a single .xlsx with both sheets).

Usage:
    python scripts/core/lookup_pairs.py GENE_A GENE_B
    python scripts/core/lookup_pairs.py GENE_A GENE_B --annotated PATH --annotated-controls PATH --parsed PATH
    python scripts/core/lookup_pairs.py GENE_A GENE_B --annotated-xlsx PATH
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io_utils import family_id  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # scripts/core -> comb
FIGURES_DIR = PROJECT_ROOT / "figures"

_REQUIRED_COLS = ["gene1", "gene2", "Case_Samples", "Control_Samples"]


def _load_table(path, sheet_name=None) -> pd.DataFrame:
    if Path(path).suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=sheet_name, dtype=str)
    return pd.read_csv(path, sep="\t", dtype=str)


def load_pairs_table(explicit_path, xlsx_path, sheet_name, label: str):
    """explicit_path (a .tsv or .xlsx) if it exists, else sheet_name of
    xlsx_path if that exists, else None (caller skips this source)."""
    if explicit_path and Path(explicit_path).exists():
        return _load_table(explicit_path, sheet_name)
    if Path(xlsx_path).exists():
        return _load_table(xlsx_path, sheet_name)
    print(f"(skipping {label}: neither {explicit_path} nor {xlsx_path} [{sheet_name}] found)")
    return None


def find_combination_rows(df: pd.DataFrame, gene_a: str, gene_b: str) -> pd.DataFrame:
    target = frozenset((gene_a, gene_b))
    mask = [frozenset((g1, g2)) == target for g1, g2 in zip(df["gene1"], df["gene2"])]
    return df.loc[mask, _REQUIRED_COLS]


def _spids(cell) -> list:
    if pd.isna(cell) or str(cell).strip() == "":
        return []
    return [s.strip() for s in str(cell).split(",") if s.strip()]


def print_carrier_table(gene1: str, gene2: str, carriers: list, parsed: pd.DataFrame, source: str):
    """carriers: list of (spid, role) -- role is 'case' or 'control', the
    column that spid came from in this row."""
    families = sorted({family_id(spid) for spid, _ in carriers})

    for fam in families:
        fam_carriers = [f"{spid} ({role})" for spid, role in carriers if family_id(spid) == fam]
        print(f"\n=== Family {fam} | combination {gene1} + {gene2} [{source}] "
              f"| carriers: {', '.join(fam_carriers)} ===")

        fam_rows = parsed[
            parsed["spid"].str.startswith(fam + ".") &
            parsed["gene"].isin([gene1, gene2])
        ].sort_values(["gene", "spid"])

        cols = ["gene", "spid", "consequence", "hg38_ID", "carrier", "CADD_PHRED", "pheno"]
        print(fam_rows[cols].to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gene_a")
    ap.add_argument("gene_b")
    ap.add_argument("--annotated", default=str(FIGURES_DIR / "annotated_gene_pairs_cases.tsv"))
    ap.add_argument("--annotated-controls", default=str(FIGURES_DIR / "annotated_gene_pairs_controls.tsv"))
    ap.add_argument("--annotated-xlsx", default=str(FIGURES_DIR / "Gene_pairs_SPARK_annotated.xlsx"),
                    help="fallback source when --annotated/--annotated-controls don't exist: a single "
                         ".xlsx with enriched_in_pro/enriched_in_sib sheets (build_final_annotated_table.py's output)")
    ap.add_argument("--parsed", default=str(PROJECT_ROOT / "results_221" / "parsed_dataset.tsv"))
    args = ap.parse_args()

    parsed = pd.read_csv(args.parsed, sep="\t", dtype=str)

    sources = [
        ("case_pairs_table", args.annotated, "enriched_in_pro"),
        ("control_pairs_table", args.annotated_controls, "enriched_in_sib"),
    ]

    found_any = False
    for label, explicit_path, sheet_name in sources:
        df = load_pairs_table(explicit_path, args.annotated_xlsx, sheet_name, label)
        if df is None:
            continue
        rows = find_combination_rows(df, args.gene_a, args.gene_b)
        if rows.empty:
            continue
        found_any = True
        for _, row in rows.iterrows():
            gene1, gene2 = row["gene1"], row["gene2"]
            carriers = ([(s, "case") for s in _spids(row["Case_Samples"])] +
                        [(s, "control") for s in _spids(row["Control_Samples"])])
            print_carrier_table(gene1, gene2, carriers, parsed, label)

    if not found_any:
        print(f"No row found for combination {args.gene_a} + {args.gene_b} in any available source "
              f"({args.annotated}, {args.annotated_controls}, {args.annotated_xlsx}).")


if __name__ == "__main__":
    main()
