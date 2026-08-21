"""
Resolve the inheritance origin of every RareComb 2-gene combination, for
every supporting individual, and flag which of those (individual,
combination) instances are valid support

Usage:
    python 05_annotate_inheritance.py parsed_dataset.tsv rarecomb_output.txt \
        affected_parents.txt inheritance_flag.tsv --cadd 30 \
        --exclude-bed exclude.bed --excluded-pairs excluded_pairs.tsv
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io_utils import (  # noqa: E402
    ORIGIN_COLS, CAT_TO_COL, DIFF_SOURCE_CATS,
    annotate_inheritance, build_consequence_source, build_proband_gene_source,
    combination_key, family_id, filter_rrvs, find_tag_column,
    flag_excluded_variants, load_bed_intervals, load_excluded_pairs,
    origin_category, parse_rarecomb,
)

_PROBAND_RE = re.compile(r"\.p\d*$")


def load_affected_parents(path: str) -> dict:
    affected = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        fam, _, parent = line.rpartition(".")
        affected.setdefault(fam, set()).add(parent)
    return affected


# ── Step 0: the optional exclusion layer ─────────────────────────────────────

def build_region_exclusion_counts(case_df: pd.DataFrame, exclude_bed: str,
                                  tag_col: str, tag_base: int) -> dict:
    """`{(spid, gene): n_variants_in_excluded_region}`, or `{}` if no BED
    was given. One pass over the case variant table."""
    if not exclude_bed:
        return {}

    index = load_bed_intervals(exclude_bed)
    n_regions = sum(len(starts) for starts, _ in index.values())
    print(f"  exclusion BED: {n_regions:,} merged regions over {len(index)} chromosomes")

    resolved_tag_col = find_tag_column(case_df, tag_col)
    if resolved_tag_col is None:
        raise KeyError(
            "--exclude-bed needs a variant coordinate column; none of "
            "hg38_ID/hg38_tag/tag is present. Pass --tag-col explicitly."
        )

    in_excluded = flag_excluded_variants(case_df, index, resolved_tag_col, tag_base)
    print(f"  coordinate column: {resolved_tag_col} ({tag_base}-based); "
          f"variants inside excluded regions: {int(in_excluded.sum()):,} "
          f"/ {len(case_df):,} ({100.0 * in_excluded.mean():.2f}%)")

    return case_df[in_excluded].groupby(["spid", "gene"]).size().to_dict()


# ── Step 1: build the raw origin instances table ─────────────────────────────

def build_origin_instances(
    df: pd.DataFrame,
    rarecomb_output: str,
    affected_parents_path: str,
    category_col: str,
    case_value: str,
    cadd_cutoff: float,
    exclude_bed: str = None,
    tag_col: str = None,
    tag_base: int = 1,
    excluded_pairs: dict = None,
) -> pd.DataFrame:
    rrv = filter_rrvs(df, cadd_cutoff)
    rrv = annotate_inheritance(rrv)
    case_df = rrv[rrv[category_col] == case_value]
    print(f"  {len(case_df):,} variant rows, {category_col}=='{case_value}'")

    gene_source = build_proband_gene_source(case_df)
    consequence_source = build_consequence_source(case_df)
    region_excl_counts = build_region_exclusion_counts(case_df, exclude_bed, tag_col, tag_base)
    excluded_pairs = excluded_pairs or {}

    known_genes = set(df["gene"].dropna().astype(str))
    combos, case_samples = parse_rarecomb(rarecomb_output, return_samples=True, known_genes=known_genes)
    affected_parents = load_affected_parents(affected_parents_path)

    rows = []
    n_skipped_size = n_skipped_unclassified = n_skipped_missing = 0
    for combo, samples_cell in zip(combos, case_samples):
        if len(combo) != 2:
            n_skipped_size += 1
            continue
        gene_a, gene_b = sorted(combo)
        combination = combination_key(gene_a, gene_b)
        pair_reason = excluded_pairs.get(combination, "")
        spids = [s for s in str(samples_cell).strip().strip('"').split(",") if s.strip()]

        for spid in spids:
            src_a = gene_source.get((spid, gene_a))
            src_b = gene_source.get((spid, gene_b))
            if src_a is None or src_b is None:
                n_skipped_missing += 1
                continue
            cat = origin_category(src_a, src_b)
            if cat is None:
                n_skipped_unclassified += 1

            fam = family_id(spid)
            affected_parent = ",".join(sorted(affected_parents.get(fam, set())))
            cons_a = consequence_source.get((spid, gene_a), "")
            cons_b = consequence_source.get((spid, gene_b), "")

            reasons = [pair_reason] if pair_reason else []
            if region_excl_counts.get((spid, gene_a)) or region_excl_counts.get((spid, gene_b)):
                reasons.append("region_exclusion")

            row = {"spid": spid, "combination": combination}
            for col in ORIGIN_COLS:
                row[col] = 0
            if cat is not None:
                row[CAT_TO_COL[cat]] = 1
            row["affected_parent"] = affected_parent
            row["consequence"] = f"{cons_a};{cons_b}"
            row["excluded"] = "yes" if reasons else "no"
            row["exclusion_reason"] = ";".join(reasons)
            rows.append(row)

    print(f"  {len(rows):,} (spid, combination) origin instances")
    print(f"    combos skipped (size != 2)                    : {n_skipped_size}")
    print(f"    instances skipped (gene not found)            : {n_skipped_missing}")
    print(f"    instances kept as unclassified (select = no)  : {n_skipped_unclassified}")

    return pd.DataFrame(rows, columns=["spid", "combination"] + ORIGIN_COLS
                         + ["affected_parent", "consequence", "excluded", "exclusion_reason"])


# ── Step 2: select flag + family dedup (vectorized) ──────────────────────────

def flag_selection(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n_total_in = len(df)
    df = df[df["spid"].str.contains(_PROBAND_RE)].copy()
    n_dropped_non_proband = n_total_in - len(df)

    # Each row is one-hot across ORIGIN_COLS (or all-zero if unclassified).
    one_hot = df[ORIGIN_COLS]
    df["origin_cat"] = one_hot.idxmax(axis=1).where(one_hot.sum(axis=1) > 0)
    df["family_id"] = df["spid"].map(family_id)
    df["affected_parent"] = df["affected_parent"].fillna("")

    # Values are sorted-joined from {mo, fa} only ("", "fa", "mo", "fa,mo"),
    # so a plain substring check is an exact membership test.
    mo_affected = df["affected_parent"].str.contains("mo")
    fa_affected = df["affected_parent"].str.contains("fa")
    is_diff_source = df["origin_cat"].isin(DIFF_SOURCE_CATS)
    is_affected_match = ((df["origin_cat"] == "MM") & mo_affected) | ((df["origin_cat"] == "PP") & fa_affected)

    df["select"] = np.where(is_diff_source | is_affected_match, "yes", "no")
    df["select_reason"] = np.select(
        [df["origin_cat"].isna(), is_diff_source, is_affected_match],
        ["unclassified", "diff_source", "affected_parent"],
        default="same_source_unaffected",
    )

    # Family dedup: among selected rows, same family + combination + origin
    # category means the same transmission event counted twice -- keep the
    # first proband (lowest spid), downgrade the rest.
    ordered = df[df["select"] == "yes"].sort_values("spid")
    dup_idx = ordered.index[ordered.duplicated(subset=["family_id", "combination", "origin_cat"])]
    df.loc[dup_idx, "select"] = "no"
    df.loc[dup_idx, "select_reason"] = "duplicate_proband_same_origin"

    n_rows = len(df)
    n_final_yes = (df["select"] == "yes").sum()
    n_excluded_of_yes = ((df["select"] == "yes") & (df["excluded"] == "yes")).sum()
    print(f"\n  Dropped {n_dropped_non_proband:,} non-proband rows")
    print(f"  select = yes : {n_final_yes:,} / {n_rows:,} ({n_final_yes/n_rows:.1%})" if n_rows else "  no rows")
    print(f"  Family dedup downgraded {len(dup_idx):,} rows to select=no")
    if n_final_yes:
        print(f"  of which excluded = yes : {n_excluded_of_yes:,} "
              f"({n_excluded_of_yes/n_final_yes:.1%}) -- filter these out downstream if needed")

    return df[["spid", "family_id", "combination"] + ORIGIN_COLS
              + ["affected_parent", "consequence", "origin_cat", "select", "select_reason",
                 "excluded", "exclusion_reason"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="parsed_dataset.tsv (core) or a carrier cohort table")
    parser.add_argument("rarecomb_output", help="RareComb compare_enrichment() output")
    parser.add_argument("affected_parents", help="One '<family>.<mo|fa>' per line")
    parser.add_argument("output", help="Where to save the inheritance flag table")
    parser.add_argument("--cadd", type=float, default=30.0,
                        help="CADD cutoff for MIS variants — must match what built the RareComb input (default: 30)")
    parser.add_argument("--category-col", default="pheno")
    parser.add_argument("--case-value", default="asd")
    parser.add_argument("--exclude-bed", default=None, metavar="FILE",
                        help="Excluded-region BED (the same one build_pairs_table.py "
                             "subtracts from the CDS). Feeds the excluded/exclusion_reason columns")
    parser.add_argument("--tag-col", default=None, metavar="COL",
                        help="Variant coordinate column for --exclude-bed "
                             "(default: first of hg38_ID/hg38_tag/tag present)")
    parser.add_argument("--tag-base", type=int, default=1, choices=(0, 1),
                        help="Coordinate convention of --tag-col: 1 = VCF-style "
                             "1-based (default), 0 = already BED-style")
    parser.add_argument("--excluded-pairs", default=None, metavar="FILE",
                        help="excluded_pairs.tsv (comb + reason), or "
                             "simulation_results_BH.tsv with --excluded-pairs-q-col. "
                             "Feeds the excluded/exclusion_reason columns")
    parser.add_argument("--excluded-pairs-q-col", default=None, metavar="COL",
                        help="Read --excluded-pairs as a results table instead, "
                             "excluding every pair whose q in this column is >= "
                             "--excluded-pairs-q-threshold (e.g. q_bh_enrich)")
    parser.add_argument("--excluded-pairs-q-threshold", type=float, default=0.05)
    args = parser.parse_args()

    print(f"Loading {args.dataset} ...")
    df = pd.read_csv(args.dataset, sep="\t", low_memory=False)

    excluded_pairs = {}
    if args.excluded_pairs:
        print(f"Loading excluded pairs from {args.excluded_pairs} ...")
        excluded_pairs = load_excluded_pairs(
            args.excluded_pairs,
            known_genes=set(df["gene"].dropna().astype(str)),
            q_col=args.excluded_pairs_q_col,
            q_threshold=args.excluded_pairs_q_threshold,
        )
        print(f"  {len(excluded_pairs):,} excluded pairs")
        for why, n in Counter(excluded_pairs.values()).most_common():
            print(f"      {why}: {n:,}")

    print("Building origin instances ...")
    instances = build_origin_instances(
        df, args.rarecomb_output, args.affected_parents,
        args.category_col, args.case_value, args.cadd,
        exclude_bed=args.exclude_bed, tag_col=args.tag_col,
        tag_base=args.tag_base, excluded_pairs=excluded_pairs,
    )

    print("Flagging valid support instances ...")
    flagged = flag_selection(instances)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    flagged.to_csv(args.output, sep="\t", index=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
