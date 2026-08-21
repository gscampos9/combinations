"""
Build the final per-combination annotated table from RareComb case/control
statistics, with exactly the column set in colunas.txt.

gene1/gene2 always follow Item_1/Item_2 (never re-sorted alphabetically) --
every gene1-keyed column (n_combo_*_gene1, etc.) refers to whichever gene
is Item_1 in that row.

Required inputs:
  - rarecomb_stats : RareComb compare_enrichment() case-vs-control stats
                      table (Item_1, Item_2, Case_Obs_Count_I1, ...,
                      Case_Samples, Control_Samples)
  - db_dir          : databases/ directory (SFARI, NDD, G2P_DD, Clingen,
                      HPA brain files, BrainSpan coexpression, Rgene2go,
                      ReactomePathways)
  - output          : where to write the annotated TSV

Optional inputs:
  --inheritance-flag       : output of 05_annotate_inheritance.py. If
                             omitted (e.g. that step hasn't been run yet),
                             origin_n_*/n_combo_*_geneN columns are filled
                             with 0 so the rest of the table is still usable.
  --ys-file / --ys-sheet   : ys_prioritized workbook, for LGD_sel/MIS_sel/
                             sel_category (default: combination_support_HC_
                             coexp_ys.xlsx / ys_prioritized)
  --ecm-genes              : plain gene list (one per line) for ecm_genes
  --ppi-file               : local STRING interactions TSV; if omitted,
                             fetched live from the STRING API and cached
  --connectors             : network/networkx_connector.py output CSV, for
                             networkx_n_hops / networkx_n_path. If omitted,
                             those two columns are left empty.
  --sfari-glob             : override which SFARI release file to use.
                             Default: auto-detected (case-insensitive, any
                             of csv/tsv/xlsx; several releases in db_dir
                             resolve to max() on the filename)
  --excluded-pairs         : the gene-level CDS-length prioritization's
                             verdict -- excluded_pairs.tsv (comb + reason),
                             or simulation_results_BH.tsv together with
                             --excluded-pairs-q-col. Adds `pair_excluded` /
                             `pair_exclusion_reason` columns. Rows are NEVER
                             dropped: excluded pairs stay in the table,
                             flagged, so the filtering is a column
                             comparison done downstream
  --term-max-genes         : GO/Reactome terms with >= this many genes are
                             dropped before computing shared terms / Jaccard
                             (default: 300)

`origin_n_excluded` reports how many of a combination's origin_n_total
support instances 05_annotate_inheritance.py flagged via its `excluded`
column (region and/or pair exclusion) -- 0 throughout if 05 ran without
--exclude-bed/--excluded-pairs. Filtering on it is left to downstream
analysis (e.g. `origin_n_total - origin_n_excluded`).

`ndd_genes`/`Sui_HC` both come from the same NDDgenes.txt gene-evidence
matrix (one row per gene, one column per source -- a gene has evidence
from a source iff that cell is non-empty; see
common/db_loaders.py:load_ndd_evidence). `ndd_genes` is any source hit,
`Sui_HC` specifically the Sui_NDD686 column.

Supersedes core/06_annotate_combinations.py as the core pipeline's final
annotation step (06 is kept only for the carrier pipeline's generic
03-06 reuse -- see README.txt). Everything 06 had that this script didn't
is folded in below:
  --carrier FILE           gene list -> carrier_genes column (06's
                           carrier-highlighting, now available here too)
  --length-correction FILE core/length_correction.py's output TSV ->
                           merges combined_length/z_raw/z_corrected/
                           pvalue_corrected/qvalue_bh/significant
  diff_source_prioritized / origin_prop_diff_source -- 06's
                           proportion-based flag (fraction of this
                           combination's supporting instances that came
                           from two different parental sources), derived
                           from the origin_n_* counts already computed
                           here rather than recomputed from the dataset

Usage:
    python 08_annotate_final.py rarecomb_stats.txt ../databases \
        annotated_combinations.tsv \
        --inheritance-flag inheritance_flag.tsv --ecm-genes ecm_genes.txt \
        --connectors gene_pair_connectors.csv
"""

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io_utils import (  # noqa: E402
    DIFF_SOURCE_CATS, combination_key, family_id, fix_dot_mangled_gene,
    load_excluded_pairs, load_gene_list,
)
from common.db_loaders import (  # noqa: E402
    PPI_BANDS as _PPI_BANDS, PPI_MIN as _PPI_MIN,
    coexpressed as _coexpressed, load_brain_enriched, load_brain_expressed,
    load_coexpression, load_go_mf, load_hgnc_symbols, load_ndd, load_ndd_evidence,
    load_ppi, load_reactome, load_sfari_scores, load_sui, ppi_score as _ppi_score,
)

_ORIGIN_COLS = ["DD", "DM", "DP", "MP", "MM", "PP"]
_VALID_CONF = {"moderate", "strong", "definitive"}
_SFARI_NDD_SCORES = ("1",)
_STRING_PPI_THRESHOLDS = (400, 700, 900)
_PROPORTION_CUTOFF = 0.6  # matches core/06_annotate_combinations.py's diff_source_prioritized cutoff


def _clean(x) -> str:
    return "" if pd.isna(x) else str(x).strip()


def _find(db_dir, pattern: str) -> str:
    matches = glob.glob(str(Path(db_dir) / pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} in {db_dir}")
    return max(matches)


def _read_table(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python", dtype=str, encoding="utf-8-sig")


def _col(df: pd.DataFrame, *candidates: str):
    norm = lambda s: re.sub(r"[\s_-]", "", s.strip().lower())
    lookup = {norm(c): c for c in df.columns}
    for cand in candidates:
        hit = lookup.get(norm(cand))
        if hit is not None:
            return hit
    raise KeyError(f"None of {candidates} found in columns {list(df.columns)}")


# ── Base stats table: Item_1/Item_2 -> gene1/gene2, n_fam/n_pro/n_sib ────────

def _samples(cell) -> list:
    if pd.isna(cell) or str(cell).strip().upper() in ("", "NA"):
        return []
    return [s.strip() for s in str(cell).strip().strip('"').split(",") if s.strip()]


def load_stats(path: str, known_genes: set = None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df["gene1"] = df["Item_1"].str.replace("Input_", "", regex=False)
    df["gene2"] = df["Item_2"].str.replace("Input_", "", regex=False)
    if known_genes:
        df["gene1"] = df["gene1"].apply(lambda g: fix_dot_mangled_gene(g, known_genes))
        df["gene2"] = df["gene2"].apply(lambda g: fix_dot_mangled_gene(g, known_genes))

    pro = df["Case_Samples"].apply(_samples)
    sib = df["Control_Samples"].apply(_samples)
    df["n_pro"] = pro.apply(len)
    df["n_sib"] = sib.apply(len)
    df["n_fam"] = [
        len({family_id(s) for s in p} | {family_id(s) for s in c})
        for p, c in zip(pro, sib)
    ]
    return df


# ── Origin support + per-gene LGD/MIS combo counts, from inheritance_flag ────

def _side_type(side: str) -> str:
    side = str(side)
    if "LGD" in side:
        return "LGD"
    if "MIS" in side:
        return "MIS"
    return ""


ORIGIN_COUNT_COLS = [
    "origin_n_total", "origin_n_DD", "origin_n_DM", "origin_n_DP", "origin_n_MP",
    "origin_n_MM_affected", "origin_n_PP_affected",
    "origin_n_MM_unaffected", "origin_n_PP_unaffected", "origin_n_excluded",
    "n_fam_diff_source",
    "n_combo_LGD_LGD_gene1", "n_combo_LGD_MIS_gene1", "n_combo_MIS_MIS_gene1",
    "n_combo_LGD_LGD_gene2", "n_combo_LGD_MIS_gene2", "n_combo_MIS_MIS_gene2",
]


def build_origin_and_combo_counts(flag_path: str, gene1_of: dict) -> pd.DataFrame:
    """gene1_of: combination key ("geneA,geneB", sorted) -> gene1 (Item_1,
    prefix-stripped) for that row, so LGD/MIS counts can be attributed to
    the gene1/gene2 roles used in the stats table rather than the flag
    file's alphabetical gene_a/gene_b.

    `origin_n_excluded` counts select=="yes" instances that
    05_annotate_inheritance.py's `excluded` column flagged (region and/or
    pair exclusion) -- 0 for every combination if 05 ran without
    --exclude-bed/--excluded-pairs.

    `n_fam_diff_source` is families, not instances: distinct family_id
    values with select=="yes" support in one of the diff-source categories
    (DD/DM/DP/MP) -- how n_fam (total families carrying both genes,
    counted upstream from RareComb's Case/Control_Samples before any
    inheritance filtering) narrows once you require diff-source support."""
    df = pd.read_csv(flag_path, sep="\t", dtype=str)
    for col in _ORIGIN_COLS:
        df[col] = df[col].astype(int)
    has_excluded_col = "excluded" in df.columns

    counts: dict = {}
    diff_source_families: dict = {}

    def _row(combo):
        return counts.setdefault(combo, {c: 0 for c in ORIGIN_COUNT_COLS})

    for _, r in df.iterrows():
        combo = r["combination"]
        cat = _clean(r.get("origin_cat", ""))
        select = r.get("select", "")
        reason = r.get("select_reason", "")
        row = _row(combo)

        if select == "yes":
            row["origin_n_total"] += 1
            if cat in DIFF_SOURCE_CATS:
                row[f"origin_n_{cat}"] += 1
                diff_source_families.setdefault(combo, set()).add(r.get("family_id", ""))
            elif cat == "MM":
                row["origin_n_MM_affected"] += 1
            elif cat == "PP":
                row["origin_n_PP_affected"] += 1
            if has_excluded_col and r.get("excluded") == "yes":
                row["origin_n_excluded"] += 1
        elif reason == "same_source_unaffected":
            if cat == "MM":
                row["origin_n_MM_unaffected"] += 1
            elif cat == "PP":
                row["origin_n_PP_unaffected"] += 1

        if select != "yes":
            continue

        genes = sorted(combo.split(","))
        if len(genes) != 2:
            continue
        gene_a, gene_b = genes
        parts = str(r.get("consequence", "")).split(";")
        if len(parts) != 2:
            continue
        type_a, type_b = _side_type(parts[0]), _side_type(parts[1])
        if not type_a or not type_b:
            continue

        g1 = gene1_of.get(combo)
        if g1 == gene_a:
            type_g1, type_g2 = type_a, type_b
        elif g1 == gene_b:
            type_g1, type_g2 = type_b, type_a
        else:
            continue  # combination not present in the stats table

        if type_g1 == "LGD" and type_g2 == "LGD":
            row["n_combo_LGD_LGD_gene1"] += 1
            row["n_combo_LGD_LGD_gene2"] += 1
        elif type_g1 == "MIS" and type_g2 == "MIS":
            row["n_combo_MIS_MIS_gene1"] += 1
            row["n_combo_MIS_MIS_gene2"] += 1
        elif type_g1 == "LGD" and type_g2 == "MIS":
            row["n_combo_LGD_MIS_gene1"] += 1
        elif type_g1 == "MIS" and type_g2 == "LGD":
            row["n_combo_LGD_MIS_gene2"] += 1

    for combo, fams in diff_source_families.items():
        counts[combo]["n_fam_diff_source"] = len(fams)

    return pd.DataFrame.from_dict(counts, orient="index")


# ── Gene-list / database loaders not already in common/db_loaders.py ────────

def load_g2p_dd(db_dir: str) -> tuple:
    df = _read_table(_find(db_dir, "G2P_DD_*"))
    gene_col = _col(df, "gene symbol")
    conf_col = _col(df, "confidence")
    allelic_col = _col(df, "allelic requirement")
    mechanism_col = _col(df, "molecular mechanism")
    variant_col = _col(df, "variant types")
    conf, allelic, mech, var = {}, {}, {}, {}
    for _, r in df.iterrows():
        gene = _clean(r[gene_col])
        if not gene:
            continue
        conf[gene] = _clean(r[conf_col])
        allelic[gene] = _clean(r[allelic_col])
        mech[gene] = _clean(r[mechanism_col])
        var[gene] = _clean(r[variant_col])
    return conf, allelic, mech, var


def load_clingen(db_dir: str) -> tuple:
    df = _read_table(_find(db_dir, "Clingen_Gene-Disease_Validity_*"))
    gene_col = _col(df, "gene")
    class_col = _col(df, "classification")
    panel_col = _col(df, "expert panel", "expert_panel")
    classification, panel = {}, {}
    for _, r in df.iterrows():
        raw_gene = _clean(r[gene_col])
        gene = raw_gene.split("HGNC")[0].strip() if "HGNC" in raw_gene else raw_gene
        cls = _clean(r[class_col]).lower()
        if gene and cls in _VALID_CONF:
            classification[gene] = cls
            panel[gene] = _clean(r[panel_col])
    return classification, panel


def load_ys_prioritized(path: str, sheet: str) -> tuple:
    df = pd.read_excel(path, sheet_name=sheet, dtype=str)
    gene_col = _col(df, "gene")
    lgd_col = _col(df, "LGD_sel")
    mis_col = _col(df, "MIS_sel")
    lgd_sel, mis_sel = {}, {}
    for _, r in df.iterrows():
        gene = _clean(r[gene_col])
        if not gene:
            continue
        if _clean(r[lgd_col]).lower() == "y":
            lgd_sel[gene] = "y"
        if _clean(r[mis_col]).lower() == "y":
            mis_sel[gene] = "y"
    return lgd_sel, mis_sel


_NUM_WORDS = {1: "one", 2: "two"}


def _sel_label(genes: list, lgd_sel: dict, mis_sel: dict) -> str:
    counts = {"LGD": 0, "MIS": 0, "LGD_MIS": 0}
    for g in genes:
        in_lgd, in_mis = g in lgd_sel, g in mis_sel
        if in_lgd and in_mis:
            counts["LGD_MIS"] += 1
        elif in_lgd:
            counts["LGD"] += 1
        elif in_mis:
            counts["MIS"] += 1
    parts = [f"{_NUM_WORDS.get(counts[c], str(counts[c]))}_{c}" for c in ("LGD", "MIS", "LGD_MIS") if counts[c] > 0]
    return "_".join(parts)


def load_connectors(path: str) -> dict:
    df = pd.read_csv(path, dtype=str)
    out = {}
    for _, r in df.iterrows():
        g1, g2 = _clean(r.get("Gene1", "")), _clean(r.get("Gene2", ""))
        if not g1 or not g2 or not _clean(r.get("Path", "")):
            continue
        n_hops = _clean(r.get("N_hops", ""))
        try:
            n_hops = str(int(float(n_hops)))
        except ValueError:
            pass
        out[frozenset((g1, g2))] = {"N_hops": n_hops, "Path": _clean(r.get("Path", ""))}
    return out


# ── Term indices for GO BP / GO MF / Reactome shared-terms + Jaccard ────────

def _gene_term_index(term_to_genes: dict, max_genes: int) -> dict:
    """{term: set(genes)} -> {gene: set(terms)}, dropping terms with >= max_genes."""
    gene_terms: dict = {}
    for term, genes in term_to_genes.items():
        if len(genes) >= max_genes:
            continue
        for g in genes:
            gene_terms.setdefault(g, set()).add(term)
    return gene_terms


def load_go_bp(db_dir, max_genes: int) -> dict:
    f = _find(db_dir, "Rgene2go_v2*")
    go = pd.read_csv(f, sep="\t", dtype=str)
    go = go[go["ONTOLOGY"].str.strip().str.upper() == "BP"]
    idx = {}
    for term, grp in go.groupby("TERM"):
        idx[f"GO:BP - {term.strip()}"] = frozenset(grp["SYMBOL"].str.strip())
    return idx


def _shared_and_jaccard(gene_a, gene_b, gene_terms: dict):
    a, b = gene_terms.get(gene_a, set()), gene_terms.get(gene_b, set())
    union = a | b
    shared = sorted(a & b)
    jacc = len(shared) / len(union) if union else float("nan")
    return ";".join(shared), jacc


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rarecomb_stats")
    parser.add_argument("db_dir")
    parser.add_argument("output")
    parser.add_argument("--inheritance-flag", default=None,
                        help="Output of 05_annotate_inheritance.py. If omitted, "
                             "origin_n_*/n_combo_*_geneN columns are filled with 0.")
    parser.add_argument("--ys-file", default="combination_support_HC_coexp_ys.xlsx")
    parser.add_argument("--ys-sheet", default="ys_prioritized")
    parser.add_argument("--ecm-genes", default=None)
    parser.add_argument("--ppi-file", default=None)
    parser.add_argument("--connectors", default=None)
    parser.add_argument("--sfari-glob", default=None,
                        help="Override which SFARI release file to use (default: "
                             "auto-detected, case-insensitive, csv/tsv/xlsx)")
    parser.add_argument("--excluded-pairs", default=None, metavar="FILE",
                        help="excluded_pairs.tsv (comb + reason), or "
                             "simulation_results_BH.tsv with --excluded-pairs-q-col. "
                             "Adds pair_excluded / pair_exclusion_reason; no row is dropped")
    parser.add_argument("--excluded-pairs-q-col", default=None, metavar="COL",
                        help="Read --excluded-pairs as a results table instead, "
                             "excluding every pair whose q in this column is >= "
                             "--excluded-pairs-q-threshold (e.g. q_bh_enrich)")
    parser.add_argument("--excluded-pairs-q-threshold", type=float, default=0.05)
    parser.add_argument("--carrier", default=None, metavar="FILE",
                        help="Gene list for carrier highlighting (carrier_genes column)")
    parser.add_argument("--length-correction", default=None, metavar="FILE",
                        help="core/length_correction.py output TSV -- merges combined_length/"
                             "z_raw/z_corrected/pvalue_corrected/qvalue_bh/significant, "
                             "matched by gene pair")
    parser.add_argument("--term-max-genes", type=int, default=300)
    args = parser.parse_args()

    db_dir = args.db_dir

    hgnc_symbols = load_hgnc_symbols(db_dir)

    print(f"Loading {args.rarecomb_stats}")
    df = load_stats(args.rarecomb_stats, known_genes=hgnc_symbols).reset_index(drop=True)
    gene1_of = dict(zip(
        (",".join(sorted([g1, g2])) for g1, g2 in zip(df["gene1"], df["gene2"])),
        df["gene1"],
    ))

    combo_key = [combination_key(g1, g2) for g1, g2 in zip(df["gene1"], df["gene2"])]
    if args.inheritance_flag:
        print(f"Loading {args.inheritance_flag}")
        origin = build_origin_and_combo_counts(args.inheritance_flag, gene1_of)
        df = df.join(origin.reindex(columns=ORIGIN_COUNT_COLS, index=combo_key).reset_index(drop=True))
    else:
        print("No --inheritance-flag given: origin_n_*/n_combo_*_geneN columns filled with 0")
        for c in ORIGIN_COUNT_COLS:
            df[c] = 0
    df[ORIGIN_COUNT_COLS] = df[ORIGIN_COUNT_COLS].fillna(0).astype(int)

    # diff_source_prioritized / origin_prop_diff_source -- 06's proportion
    # flag, derived from the origin_n_* counts above. DD/DM/DP/MP are
    # diff-source by construction; MM_affected/PP_affected are the
    # same-source instances that still passed selection. Note this is
    # scored over select=="yes" instances only (what's already in
    # origin_n_total here), not every resolved instance regardless of
    # selection like 06's version was -- the more relevant population for
    # a "should this pair be prioritized" flag.
    diff_n = df["origin_n_DD"] + df["origin_n_DM"] + df["origin_n_DP"] + df["origin_n_MP"]
    has_support = df["origin_n_total"] > 0
    df["origin_prop_diff_source"] = np.where(has_support, diff_n / df["origin_n_total"].replace(0, np.nan), np.nan)
    df["diff_source_prioritized"] = has_support & (df["origin_prop_diff_source"] > _PROPORTION_CUTOFF)

    if args.carrier:
        carrier = load_gene_list(args.carrier)
        df["carrier_genes"] = [",".join(g for g in (g1, g2) if g in carrier)
                               for g1, g2 in zip(df["gene1"], df["gene2"])]

    if args.length_correction:
        print(f"Merging length correction results from {args.length_correction}")
        lc = pd.read_csv(args.length_correction, sep="\t")
        lc["combination"] = [combination_key(a, b) for a, b in zip(lc["Item_1"], lc["Item_2"])]
        lc = lc.drop(columns=["Item_1", "Item_2"]).drop_duplicates(subset="combination")
        df["combination"] = combo_key
        df = df.merge(lc, on="combination", how="left")
        print(f"  {df['qvalue_bh'].notna().sum()}/{len(df)} combinations matched"
              if "qvalue_bh" in df.columns else "  no qvalue_bh column in the correction file")

    if args.excluded_pairs:
        print(f"Loading excluded pairs from {args.excluded_pairs}")
        excluded = load_excluded_pairs(
            args.excluded_pairs,
            known_genes=set(df["gene1"]) | set(df["gene2"]),
            q_col=args.excluded_pairs_q_col,
            q_threshold=args.excluded_pairs_q_threshold,
        )
        df["pair_exclusion_reason"] = [excluded.get(k, "") for k in combo_key]
        df["pair_excluded"] = (df["pair_exclusion_reason"] != "").map({True: "yes", False: "no"})
        n_hit = int((df["pair_excluded"] == "yes").sum())
        print(f"  {len(excluded):,} pairs in the exclusion file; "
              f"{n_hit:,}/{len(df):,} rows of this table flagged (kept, not dropped)")

    print(f"Loading gene-list databases from {db_dir}")
    sfari_scores = load_sfari_scores(db_dir, args.sfari_glob)
    sfari_1 = {g for g, v in sfari_scores.items() if v.startswith(_SFARI_NDD_SCORES)}
    ndd_evidence = load_ndd_evidence(db_dir)
    ndd = load_ndd(evidence=ndd_evidence)
    sui = load_sui(evidence=ndd_evidence)
    g2p_conf, g2p_allelic, g2p_mech, g2p_var = load_g2p_dd(db_dir)
    clingen, clingen_panel = load_clingen(db_dir)
    brain_enr = load_brain_enriched(db_dir)
    brain_exp = load_brain_expressed(db_dir)
    coexpr = load_coexpression(db_dir)
    ecm = load_gene_list(args.ecm_genes) if args.ecm_genes else set()

    print(f"Loading {args.ys_file} [{args.ys_sheet}]")
    lgd_sel, mis_sel = load_ys_prioritized(args.ys_file, args.ys_sheet)

    all_genes = set(df["gene1"]) | set(df["gene2"])
    print(f"Loading STRING PPI for {len(all_genes)} genes")
    ppi = load_ppi(all_genes, args.ppi_file)

    print(f"Building GO BP / GO MF / Reactome term indices (max {args.term_max_genes} genes/term)")
    go_bp_genes = _gene_term_index(load_go_bp(db_dir, args.term_max_genes), args.term_max_genes)
    go_mf_genes = _gene_term_index(load_go_mf(db_dir, max_genes=args.term_max_genes), args.term_max_genes)
    reactome_genes = _gene_term_index(
        {p["pathway"]: p["genes"] for p in load_reactome(db_dir)}, args.term_max_genes)

    connectors = load_connectors(args.connectors) if args.connectors else {}

    print("Annotating ...")
    rows = []
    for _, r in df.iterrows():
        g1, g2 = r["gene1"], r["gene2"]
        genes = [g1, g2]

        sfari_hits = [g for g in genes if g in sfari_scores]
        sfari_1_hits = [g for g in genes if g in sfari_1]
        ndd_hits = [g for g in genes if g in ndd]
        sui_hits = [g for g in genes if g in sui]
        brain_exp_hits = [g for g in genes if g in brain_exp]

        s = _ppi_score(g1, g2, ppi)
        bp_shared, bp_jacc = _shared_and_jaccard(g1, g2, go_bp_genes)
        mf_shared, mf_jacc = _shared_and_jaccard(g1, g2, go_mf_genes)
        rx_shared, rx_jacc = _shared_and_jaccard(g1, g2, reactome_genes)
        has_coexpr = _coexpressed(g1, g2, coexpr)
        conn = connectors.get(frozenset(genes), {})

        row = {
            "sfari_genes": ",".join(sfari_hits),
            "sfari_n": len(sfari_hits),
            "sfari_1_genes": ",".join(sfari_1_hits),
            "sfari_1_n": len(sfari_1_hits),
            "ndd_genes": ",".join(ndd_hits),
            "ndd_n": len(ndd_hits),
            "Sui_HC": ",".join(sui_hits),
            "G2P_DD": ";".join(f"{g}_{g2p_conf[g]}" for g in genes if g in g2p_conf),
            "G2P_DD_allelic": ";".join(f"{g}_{g2p_allelic[g]}" for g in genes if g in g2p_allelic),
            "G2P_DD_mechanism": ";".join(f"{g}_{g2p_mech[g]}" for g in genes if g in g2p_mech),
            "G2P_DD_variant": ";".join(f"{g}_{g2p_var[g]}" for g in genes if g in g2p_var),
            "Clingen": ";".join(f"{g}_{clingen[g]}" for g in genes if g in clingen),
            "Clingen_panel": ";".join(f"{g}_{clingen_panel[g]}" for g in genes if g in clingen_panel),
            "LGD_sel": ",".join(g for g in genes if g in lgd_sel),
            "MIS_sel": ",".join(g for g in genes if g in mis_sel),
            "sel_category": _sel_label(genes, lgd_sel, mis_sel),
            "ecm_genes": ",".join(g for g in genes if g in ecm),
            "brain_expressed_genes": ",".join(brain_exp_hits),
            "brain_expressed_n": len(brain_exp_hits),
            "brain_enriched_genes": ",".join(g for g in genes if g in brain_enr),
            "STRING_ppi_value": s,
            "STRING_ppi_400": int(s >= 400),
            "STRING_ppi_700": int(s >= 700),
            "STRING_ppi_900": int(s >= 900),
            "go_bp_shared_terms": bp_shared,
            "go_bp_jaccard": bp_jacc,
            "go_mf_shared_terms": mf_shared,
            "go_mf_jaccard": mf_jacc,
            "reactome_shared_pathways": rx_shared,
            "reactome_shared_jaccard": rx_jacc,
            "coexpression": f"{g1}-{g2}:module={coexpr[g1]}" if has_coexpr else "",
            "functional_evidence": "y" if (s >= _PPI_MIN or has_coexpr) else "",
            "networkx_n_hops": conn.get("N_hops", ""),
            "networkx_n_path": conn.get("Path", ""),
        }
        rows.append(row)

    ann = pd.DataFrame(rows)
    out = pd.concat([df.reset_index(drop=True), ann], axis=1)

    final_cols = [
        "Item_1", "Item_2", "Case_Obs_Count_I1", "Case_Obs_Count_I2",
        "Case_Exp_Prob_Combo", "Case_Obs_Prob_Combo", "Case_Exp_Count_Combo",
        "Case_Obs_Count_Combo", "Case_pvalue_more", "Cont_Obs_Count_I1",
        "Cont_Obs_Count_I2", "Cont_Exp_Prob_Combo", "Cont_Obs_Prob_Combo",
        "Cont_Exp_Count_Combo", "Cont_Obs_Count_Combo", "Control_pvalue_more",
        "Case_Adj_Pval_BH", "Case_Adj_Pval_bonf", "Effect_Size",
        "Power_One_Pct", "Power_Five_Pct", "Case_Samples", "Control_Samples",
        "gene1", "gene2", "n_fam", "n_pro", "n_sib", "n_fam_diff_source",
        "origin_n_total", "origin_n_DD", "origin_n_DM", "origin_n_DP", "origin_n_MP",
        "origin_n_MM_affected", "origin_n_PP_affected",
        "origin_n_MM_unaffected", "origin_n_PP_unaffected", "origin_n_excluded",
        "origin_prop_diff_source", "diff_source_prioritized", "carrier_genes",
        "n_combo_LGD_LGD_gene1", "n_combo_LGD_MIS_gene1", "n_combo_MIS_MIS_gene1",
        "n_combo_LGD_LGD_gene2", "n_combo_LGD_MIS_gene2", "n_combo_MIS_MIS_gene2",
        "sfari_genes", "sfari_n", "sfari_1_genes", "sfari_1_n", "ndd_genes", "ndd_n",
        "Sui_HC", "G2P_DD", "G2P_DD_allelic", "G2P_DD_mechanism", "G2P_DD_variant",
        "Clingen", "Clingen_panel", "LGD_sel", "MIS_sel", "sel_category",
        "ecm_genes", "brain_expressed_genes", "brain_expressed_n", "brain_enriched_genes",
        "STRING_ppi_value", "STRING_ppi_400", "STRING_ppi_700", "STRING_ppi_900",
        "go_bp_shared_terms", "go_bp_jaccard", "go_mf_shared_terms", "go_mf_jaccard",
        "reactome_shared_pathways", "reactome_shared_jaccard",
        "coexpression", "functional_evidence", "networkx_n_hops", "networkx_n_path",
    ]
    # length_correction / exclusion layer last, so colunas.txt's column
    # order is untouched for every consumer that doesn't care about them.
    final_cols += ["combined_length", "z_raw", "z_corrected", "pvalue_corrected",
                   "qvalue_bh", "significant"]
    final_cols += ["pair_excluded", "pair_exclusion_reason"]
    out = out[[c for c in final_cols if c in out.columns]]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, sep="\t", index=False)
    print(f"\nSaved: {args.output}  ({len(out):,} combinations, {len(out.columns)} columns)")


if __name__ == "__main__":
    main()