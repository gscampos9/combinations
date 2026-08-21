"""
Shared helpers
"""

from __future__ import annotations

import bisect
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

# ── Individual id / consequence ──────────────────────────────────────────────

def add_spid(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `spid` column: `person` if non-empty, else `EID`."""
    df = df.copy()
    person_ok = df["person"].notna() & (df["person"].astype(str).str.strip() != "")
    df["spid"] = df["person"].where(person_ok, df["EID"])
    return df


def load_spark(data_file: str) -> pd.DataFrame:
    """Load a raw SPARK-style variant TSV and add `spid` + `consequence`.

    `consequence` is `CQ` when present, else `DNV_type` (de novo variants
    are only annotated in `DNV_type`). The `tag` column, when present, is
    renamed to `hg38_ID`.
    """
    spark = pd.read_csv(data_file, sep="\t", low_memory=False)

    cq_ok = spark["CQ"].notna() & (spark["CQ"].astype(str).str.strip() != "")

    spark = add_spid(spark)
    spark["consequence"] = spark["CQ"].where(cq_ok, spark["DNV_type"])
    if "tag" in spark.columns:
        spark = spark.rename(columns={"tag": "hg38_ID"})

    return spark


# ── Phenotype ─────────────────────────────────────────────────────────────────

ASD_SUFFIXES = (".p", ".p1", ".p2", ".p3", ".p4", ".p5", ".p6")
SIB_SUFFIXES = (".s", ".s1", ".s2", ".s3", ".s4", ".s5")


def assign_phenotype(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `pheno` column (`asd` / `sib`) derived from the `spid` suffix."""
    import numpy as np

    df = df.copy()
    df["pheno"] = np.select(
        [
            df["spid"].str.endswith(tuple(ASD_SUFFIXES)),
            df["spid"].str.endswith(tuple(SIB_SUFFIXES)),
        ],
        ["asd", "sib"],
        default=None,
    )
    return df


# ── Rare damaging variant filter ─────────────────────────────────────────────

def filter_rrvs(spark: pd.DataFrame, cadd_cutoff: float) -> pd.DataFrame:
    """Keep only LGD variants, or MIS variants with CADD >= cadd_cutoff."""
    mask = (
        (spark["consequence"] == "LGD") |
        (
            (spark["consequence"] == "MIS") &
            (pd.to_numeric(spark["CADD_PHRED"], errors="coerce") >= cadd_cutoff)
        )
    )
    return spark[mask].copy()


# ── Gene lists ────────────────────────────────────────────────────────────────

def load_gene_list(path: str) -> set:
    genes = pd.read_csv(path, header=None, names=["gene"])
    return set(genes["gene"].astype(str).str.strip())


# ── Inheritance ───────────────────────────────────────────────────────────────

INHERITANCE_MAP = {"mo": "maternal", "fa": "paternal"}


def annotate_inheritance(df: pd.DataFrame, carrier_col: str = "carrier") -> pd.DataFrame:
    """Add an `inheritance` column: "de_novo" for type == "DNV", else the
    transmitting parent (maternal/paternal) from the `carrier` column for
    type == "PIV", else "unresolved"."""
    df = df.copy()
    inh = pd.Series(index=df.index, dtype=object)
    inh[df["type"] == "DNV"] = "de_novo"
    piv_mask = df["type"] == "PIV"
    inh[piv_mask] = df.loc[piv_mask, carrier_col].map(INHERITANCE_MAP)
    df["inheritance"] = inh.fillna("unresolved")
    return df


_FAMILY_SUFFIX_RE = re.compile(r"\.(?:p|s)\d*$")


def family_id(spid: str) -> str:
    """Strip the proband/sibling suffix (.p, .p1, .s2, ...) from an spid."""
    return _FAMILY_SUFFIX_RE.sub("", str(spid).strip().strip('"'))


# ── Per-gene / per-pair inheritance source resolution ────────────────────────
# Shared by core/05_annotate_inheritance.py (per-instance origin + selection)
# and core/06_annotate_combinations.py (per-combination origin proportions +
# pair-level diff_source scoring).

ORIGIN_COLS = ["DD", "DM", "DP", "MP", "MM", "PP"]
CAT_TO_COL = {
    "both_denovo": "DD", "denovo_maternal": "DM", "denovo_paternal": "DP",
    "maternal_paternal": "MP", "both_maternal": "MM", "both_paternal": "PP",
}
DIFF_SOURCE_CATS = {"DD", "DM", "DP", "MP"}
SAME_SOURCE_CATS = {"MM", "PP"}


def resolve_gene_source(inheritances: pd.Series) -> str:
    """Collapse one gene's per-variant inheritance values (for one
    individual) into a single source: de_novo > mixed (both parents) >
    paternal/maternal/ambiguous/unresolved."""
    vals = set(inheritances.dropna())
    if "de_novo" in vals:
        return "de_novo"
    if "paternal" in vals and "maternal" in vals:
        return "mixed"
    for s in ("paternal", "maternal", "ambiguous", "unresolved"):
        if s in vals:
            return s
    return "unresolved"


def build_proband_gene_source(df: pd.DataFrame) -> dict:
    """(spid, gene) -> resolved inheritance source, from a variant table
    with an `inheritance` column (see `annotate_inheritance`)."""
    grouped = df.groupby(["spid", "gene"])["inheritance"]
    return {key: resolve_gene_source(s) for key, s in grouped}


def build_consequence_source(df: pd.DataFrame) -> dict:
    """(spid, gene) -> comma-joined sorted unique consequences."""
    grouped = df.groupby(["spid", "gene"])["consequence"]
    return {key: ",".join(sorted(set(s.dropna()))) for key, s in grouped}


def origin_category(src_a: str, src_b: str):
    """Classify a gene pair's combined inheritance origin. Returns None if
    either source isn't a clean de_novo/maternal/paternal/mixed call.

    A "mixed" source (one gene carrying two-or-more qualifying variants
    inherited in trans, one maternal + one paternal) already represents
    both a maternal and a paternal contribution by itself, so any pair
    involving a mixed gene is classified as maternal_paternal (MP) --
    confirmed by manual review (diagnostics one-liner over
    05_annotate_inheritance.py's `unclassified` rows) that these are
    consistently trans compound-het cases, not genuine ambiguity."""
    if src_a == "mixed" or src_b == "mixed":
        return "maternal_paternal"
    valid = {"de_novo", "maternal", "paternal"}
    if src_a not in valid or src_b not in valid:
        return None
    pair = frozenset((src_a, src_b))
    if pair == frozenset({"de_novo"}):
        return "both_denovo"
    if pair == frozenset({"de_novo", "maternal"}):
        return "denovo_maternal"
    if pair == frozenset({"de_novo", "paternal"}):
        return "denovo_paternal"
    if pair == frozenset({"maternal", "paternal"}):
        return "maternal_paternal"
    if pair == frozenset({"maternal"}):
        return "both_maternal"
    if pair == frozenset({"paternal"}):
        return "both_paternal"
    return None


# ── RareComb output parsing ───────────────────────────────────────────────────

def _clean(x) -> str:
    return "" if pd.isna(x) else str(x).strip()


def sniff_sep(path: str) -> str:
    """Peek the header line to tell a comma- from a tab-separated file,
    avoiding pandas' `sep=None, engine="python"` sniffer -- on wide
    matrices that combination is dramatically slower than the default C
    engine for no benefit here (RareComb output is always comma or tab)."""
    with open(path) as f:
        header = f.readline()
    return "," if header.count(",") >= header.count("\t") else "\t"


def fix_dot_mangled_gene(gene: str, known_genes: set) -> str:
    """Undo R's make.names() "-" -> "." mangling of Input_<gene> column
    headers. run_rarecomb.R reads the boolean input matrix with
    data.table::fread when available (check.names=FALSE, hyphens kept),
    but falls back to base read.table (check.names=TRUE by default) when
    data.table isn't installed -- that silently turns e.g. "Input_HLA-A"
    into "Input_HLA.A" in every Item_1/Item_2 value RareComb writes out.

    Only swap when it actually resolves an unrecognized name into a known
    one, so genuine gene symbols are never touched."""
    if not known_genes or gene in known_genes or "." not in gene:
        return gene
    fixed = gene.replace(".", "-")
    return fixed if fixed in known_genes else gene


def parse_rarecomb(path: str, return_samples: bool = False, known_genes: set = None):
    """Parse a RareComb `compare_enrichment` output table into a list of
    gene combinations (each a sorted list of gene symbols, `Input_` prefix
    stripped). If `return_samples`, also return the `Case_samples` column
    (one comma-joined spid string per combination).

    `known_genes`, if given, is used to repair any dot-mangled gene name
    (see `fix_dot_mangled_gene`) -- pass the set of gene symbols actually
    present in your dataset/reference so the fix only fires on real
    mismatches."""
    df = pd.read_csv(path, sep=sniff_sep(path), dtype=str)
    cols = [c for c in df.columns if c.startswith("Item_")]

    def _genes_of(row):
        genes = {_clean(row[c]).replace("Input_", "") for c in cols} - {""}
        if known_genes:
            genes = {fix_dot_mangled_gene(g, known_genes) for g in genes}
        return sorted(genes)

    combos = [_genes_of(row) for _, row in df.iterrows()]
    if not return_samples:
        return combos
    samples_col = next((c for c in df.columns if c.strip().strip('"').lower() == "case_samples"), None)
    case_samples = df[samples_col].fillna("").tolist() if samples_col else [""] * len(df)
    return combos, case_samples


# ── Combination keys ─────────────────────────────────────────────────────────
# Everything downstream of RareComb keys a gene pair on the sorted,
# comma-joined symbol pair ("GENEA,GENEB"). The gene-level prioritization
# scripts (gene_level_prioritization/, build_pairs_table.py) instead write
# "GENEA_GENEB", so anything crossing that boundary goes through
# `split_pair` + `combination_key` rather than assuming one of the two.

_PAIR_SEPS = (",", ";", "|", "\t")


def combination_key(gene_a: str, gene_b: str) -> str:
    """The canonical sorted comma-joined key for a gene pair."""
    return ",".join(sorted([str(gene_a).strip(), str(gene_b).strip()]))


def split_pair(comb, known_genes: set = None):
    """Split a pair label back into its two gene symbols. Accepts the
    comma/semicolon/pipe-joined form used across the RareComb pipeline and
    the underscore-joined "GENEA_GENEB" form written by
    build_pairs_table.py. Returns None when it can't be split in two.

    Underscore splitting is ambiguous for symbols that contain an
    underscore themselves; pass `known_genes` (e.g. the HGNC symbol set or
    the genes present in the dataset) and the split that resolves both
    sides to known symbols wins. Without it, the first underscore is used.
    """
    s = str(comb).strip().strip('"')
    if not s:
        return None

    for sep in _PAIR_SEPS:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            return (parts[0], parts[1]) if len(parts) == 2 else None

    if "_" not in s:
        return None
    parts = s.split("_")
    if len(parts) == 2:
        return parts[0], parts[1]
    if known_genes:
        for i in range(1, len(parts)):
            gene_a, gene_b = "_".join(parts[:i]), "_".join(parts[i:])
            if gene_a in known_genes and gene_b in known_genes:
                return gene_a, gene_b
    return parts[0], "_".join(parts[1:])


_COMB_COL_CANDIDATES = ("comb", "combination", "pair", "gene_pair", "genes")
_REASON_COL_CANDIDATES = ("reason", "exclusion_reason", "excluded_reason",
                          "motivo", "why")
_GENE_COL_PAIRS = (("gene1", "gene2"), ("gene_a", "gene_b"),
                   ("item_1", "item_2"), ("geneA", "geneB"))


def _lookup(df: pd.DataFrame) -> dict:
    return {str(c).strip().lower(): c for c in df.columns}


def load_excluded_pairs(path, known_genes: set = None,
                        q_col: str = None, q_threshold: float = None) -> dict:
    """`{combination key -> exclusion reason}` for the gene-level CDS-length
    prioritization's verdict on each pair.

    Reads either shape:

    * `excluded_pairs.tsv` -- a `comb` column (either join style) plus a
      reason column (`gene_not_in_refseq`, `q_ge_0.05`, `null_sd_zero`,
      `no_result_in_table`). Used as-is.
    * `simulation_results_BH.tsv` -- the full results table; pass
      `q_col`/`q_threshold` and every row with q >= threshold (or a missing
      q) is treated as excluded, with the reason derived on the spot.

    Never raises on an unrecognised row: rows that can't be split into two
    genes are skipped and counted in the caller-visible return of
    `describe_excluded_pairs`.
    """
    df = pd.read_csv(path, sep=sniff_sep(path), dtype=str)
    lookup = _lookup(df)

    comb_col = next((lookup[c] for c in _COMB_COL_CANDIDATES if c in lookup), None)
    gene_cols = next(((lookup[a.lower()], lookup[b.lower()]) for a, b in _GENE_COL_PAIRS
                      if a.lower() in lookup and b.lower() in lookup), None)
    if comb_col is None and gene_cols is None:
        raise KeyError(f"{path}: no pair column found (looked for "
                       f"{_COMB_COL_CANDIDATES} or {_GENE_COL_PAIRS})")

    reason_col = next((lookup[c] for c in _REASON_COL_CANDIDATES if c in lookup), None)
    if q_col is not None:
        q_col = lookup.get(q_col.strip().lower(), q_col)
        if q_col not in df.columns:
            raise KeyError(f"{path}: q-value column {q_col!r} not found")

    excluded = {}
    for _, row in df.iterrows():
        if gene_cols is not None:
            gene_a = _clean(row[gene_cols[0]]).replace("Input_", "")
            gene_b = _clean(row[gene_cols[1]]).replace("Input_", "")
            pair = (gene_a, gene_b) if gene_a and gene_b else None
        else:
            pair = split_pair(_clean(row[comb_col]), known_genes)
        if pair is None:
            continue

        if q_col is not None:
            q = pd.to_numeric(row[q_col], errors="coerce")
            if pd.isna(q):
                reason = "no_qvalue"
            elif q >= q_threshold:
                reason = f"q_ge_{q_threshold:g}"
            else:
                continue  # kept by the length analysis, not an exclusion
        else:
            reason = _clean(row[reason_col]) if reason_col else "excluded"

        excluded[combination_key(*pair)] = reason or "excluded"

    return excluded


# ── Excluded genomic regions (the exclusion BED) ─────────────────────────────
# The same regions build_pairs_table.py subtracts from the CDS before
# measuring combined_length and counting DNMs. A variant sitting inside one
# of them contributed no length to the null, so it shouldn't contribute
# support either -- annotated, never silently dropped.

def norm_chrom(chrom) -> str:
    """`chr1`, `1`, `CHR1` -> `1`; `chrM`/`chrMT` -> `MT`. Lets a BED and a
    variant table that disagree on the `chr` prefix still line up."""
    c = str(chrom).strip()
    if c[:3].lower() == "chr":
        c = c[3:]
    c = c.upper()
    return "MT" if c == "M" else c


def load_bed_intervals(path) -> dict:
    """Read a BED into `{chrom: (starts, ends)}`, merged and sorted, kept in
    the BED's own 0-based half-open coordinates."""
    by_chrom = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.split()
            if len(f) < 3:
                continue
            try:
                start, end = int(f[1]), int(f[2])
            except ValueError:
                continue
            if end > start:
                by_chrom[norm_chrom(f[0])].append((start, end))

    index = {}
    for chrom, ivs in by_chrom.items():
        merged = []
        for s, e in sorted(ivs):
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        index[chrom] = ([s for s, _ in merged], [e for _, e in merged])
    return index


def in_intervals(index: dict, chrom, start0: int) -> bool:
    """Is the 0-based position `start0` inside any interval of `index`?"""
    got = index.get(norm_chrom(chrom))
    if not got:
        return False
    starts, ends = got
    i = bisect.bisect_right(starts, start0) - 1
    return i >= 0 and start0 < ends[i]


VARIANT_TAG_COLS = ("hg38_ID", "hg38_tag", "hg38_id", "tag")


def find_tag_column(df: pd.DataFrame, explicit: str = None):
    """The column holding each variant's `chr:pos:REF:ALT` hg38 coordinate.
    `load_spark` renames the raw `tag` column to `hg38_ID`; datasets built
    through add_hg38_tag.py instead carry `hg38_tag`. Returns None when
    none of them is present."""
    if explicit:
        if explicit not in df.columns:
            raise KeyError(f"--tag-col {explicit!r} not in {list(df.columns)}")
        return explicit
    lookup = _lookup(df)
    return next((lookup[c.lower()] for c in VARIANT_TAG_COLS if c.lower() in lookup), None)


def parse_variant_tag(tag):
    """`chr1:12345:A:G` -> `("chr1", 12345)`. Also accepts `chr1:12345`,
    `1-12345-A-G` and `chr1_12345`. Returns `(None, None)` for a missing or
    unparseable tag (rows with no hg38 coordinate, `NA`, alt contigs whose
    name eats the separator)."""
    if tag is None:
        return None, None
    s = str(tag).strip().strip('"')
    if not s or s.upper() in ("NA", "NAN", "NONE", "."):
        return None, None
    for parts in (s.split(":"), re.split(r"[:\-_]", s)):
        if len(parts) >= 2:
            try:
                return parts[0], int(parts[1])
            except ValueError:
                continue
    return None, None


def flag_excluded_variants(df: pd.DataFrame, index: dict, tag_col: str,
                           tag_base: int = 1) -> pd.Series:
    """Boolean Series, True where the variant's coordinate falls inside an
    excluded region.

    `tag_base` is the coordinate convention of `tag_col`: 1 (default) for
    VCF-style 1-based positions, which is what both the PIV tags and the
    lifted-over DNV tags carry -- a 1-based position p sits in the 0-based
    half-open interval [s, e) iff s <= p-1 < e. Pass 0 if the tags in a
    given dataset are already BED-style.
    """
    offset = 1 if int(tag_base) == 1 else 0

    def _hit(tag) -> bool:
        chrom, pos = parse_variant_tag(tag)
        return False if chrom is None else in_intervals(index, chrom, pos - offset)

    return df[tag_col].map(_hit).fillna(False).astype(bool)
