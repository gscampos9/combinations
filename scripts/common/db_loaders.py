"""
Shared external-database loaders (SFARI, NDD, HPA brain expression,
BrainSpan coexpression, STRING PPI) — used by core/06_annotate_combinations.py
and by the plotting/ scripts that fall back to re-loading these databases
for older annotated TSVs that don't already embed the gene-list columns.
"""

from __future__ import annotations

import fnmatch
import glob
import re
from io import StringIO
from pathlib import Path

import pandas as pd

try:
    import requests as _requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

PPI_BANDS = [
    (900, "highest (>=0.9)"), (700, "high (>=0.7)"),
    (400, "medium (>=0.4)"), (150, "low (>=0.15)"),
]
PPI_MIN = 150
_STRING_MAP = "https://string-db.org/api/tsv/get_string_ids"
_STRING_NET = "https://string-db.org/api/tsv/network"


def _clean(x) -> str:
    return "" if pd.isna(x) else str(x).strip()


def _find(db_dir, pattern: str):
    matches = glob.glob(str(Path(db_dir) / pattern))
    return matches[0] if matches else None


def load_hgnc_symbols(db_dir) -> set:
    """All current HGNC-approved gene symbols -- used as the reference set
    for fix_dot_mangled_gene (common/io_utils.py), to repair Item_1/Item_2
    gene names that R's read.table fallback mangled ("-" -> ".")."""
    f = _find(db_dir, "hgnc_complete_set*")
    if not f:
        print("  HGNC symbols: file not found - skipping dot-mangling fix")
        return set()
    df = pd.read_csv(f, sep="\t", dtype=str, low_memory=False)
    symbols = {_clean(g) for g in df.get("symbol", []) if _clean(g)}
    print(f"  HGNC symbols: {len(symbols)}")
    return symbols


# ── SFARI Gene ────────────────────────────────────────────────────────────────
# Release files vary a lot between downloads: the name can be
# SFARI-Gene_genes_05-01-2024release_06-14-2024export.csv, SFARI_Gene_...,
# or just sfari_genes.tsv; the export can be csv, tsv or xlsx; and the
# column headers alternate between `gene-symbol`/`Gene Symbol`/`gene_symbol`
# and `gene-score`/`Gene Score`/`score`. Everything here is matched on a
# normalised key (lowercased, `_ -` and spaces stripped) and every loader
# degrades to an empty result with a printed reason instead of raising, so
# one missing or renamed database never takes the whole annotation down.

_SFARI_PATTERNS = ("sfari-gene*", "sfari_gene*", "sfari*gene*", "sfari*")
_TABLE_EXTS = (".csv", ".tsv", ".txt", ".xlsx", ".xls")
_SFARI_GENE_COLS = ("gene-symbol", "gene symbol", "gene_symbol", "genesymbol",
                    "symbol", "gene")
_SFARI_SCORE_COLS = ("gene-score", "gene score", "gene_score", "genescore", "score")
_SFARI_SYNDROMIC_COLS = ("syndromic", "is-syndromic", "syndromic-status")


def _norm(s) -> str:
    return re.sub(r"[\s_-]", "", str(s).strip().lower())


def _find_ci(db_dir, patterns) -> str:
    """First file in `db_dir` matching any of `patterns` case-insensitively.
    Ties broken by max() on the filename, so a directory holding several
    releases resolves to the same one every run (matching what
    core/08_annotate_final.py already did with an explicit --sfari-glob)."""
    listing = sorted(p for p in Path(db_dir).glob("*") if p.is_file())
    for pattern in ([patterns] if isinstance(patterns, str) else patterns):
        rx = re.compile(fnmatch.translate(pattern), re.IGNORECASE)
        hits = [p for p in listing if rx.match(p.name)]
        if hits:
            return str(max(hits, key=lambda p: p.name))
    return None


def _read_table(path: str) -> pd.DataFrame:
    """Read a database export regardless of csv/tsv/xlsx, with a BOM-safe
    encoding (SFARI's csv export ships one) and everything as str."""
    suffix = Path(path).suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, dtype=str)
    return pd.read_csv(path, sep=None, engine="python", dtype=str, encoding="utf-8-sig")


def _pick_col(df: pd.DataFrame, candidates):
    lookup = {_norm(c): c for c in df.columns}
    return next((lookup[_norm(c)] for c in candidates if _norm(c) in lookup), None)


def load_sfari_table(db_dir, pattern=None) -> pd.DataFrame:
    """The SFARI Gene release as a tidy DataFrame with `gene`, `score` and
    `syndromic` columns (`score` and `syndromic` empty/False when the
    release doesn't carry them). Empty DataFrame if no file is found."""
    empty = pd.DataFrame(columns=["gene", "score", "syndromic"])

    f = _find_ci(db_dir, pattern or _SFARI_PATTERNS)
    if not f:
        print("  SFARI: file not found - skipping")
        return empty
    if Path(f).suffix.lower() not in _TABLE_EXTS:
        print(f"  SFARI: {Path(f).name} is not a readable table - skipping")
        return empty

    try:
        df = _read_table(f)
    except Exception as exc:  # unreadable/corrupt export shouldn't kill the run
        print(f"  SFARI: could not read {Path(f).name} ({exc}) - skipping")
        return empty

    gene_col = _pick_col(df, _SFARI_GENE_COLS)
    if gene_col is None:
        print(f"  SFARI: no gene-symbol column in {Path(f).name} "
              f"(columns: {list(df.columns)}) - skipping")
        return empty
    score_col = _pick_col(df, _SFARI_SCORE_COLS)
    synd_col = _pick_col(df, _SFARI_SYNDROMIC_COLS)

    out = pd.DataFrame({
        "gene": df[gene_col].map(_clean),
        "score": df[score_col].map(_clean) if score_col else "",
        "syndromic": (pd.to_numeric(df[synd_col], errors="coerce").fillna(0) == 1
                      if synd_col else False),
    })
    out = out[out["gene"] != ""].drop_duplicates(subset="gene")
    print(f"  SFARI: {len(out)} genes from {Path(f).name}"
          f" [gene={gene_col}"
          f"{f', score={score_col}' if score_col else ', no score column'}"
          f"{f', syndromic={synd_col}' if synd_col else ''}]")
    return out.reset_index(drop=True)


def load_sfari_scores(db_dir=None, pattern=None, table=None) -> dict:
    """`gene -> score label` ("1", "2", "3", with an "S" suffix for
    syndromic genes, and a bare "S" for syndromic genes carrying no
    category score -- those exist in every recent release and were silently
    lost when a non-empty score was required).

    Pass `table` (a `load_sfari_table` result) to reuse an already-loaded
    release instead of reading and re-announcing the file a second time."""
    if table is None:
        table = load_sfari_table(db_dir, pattern)
    scores = {}
    for gene, score, is_synd in zip(table["gene"], table["score"], table["syndromic"]):
        label = f"{score}S" if (score and is_synd) else (score or ("S" if is_synd else ""))
        if label:
            scores[gene] = label
    return scores


# ── NDD gene-evidence matrix ─────────────────────────────────────────────────
# One row per gene, one column per source (SFARI050126, Fu_NDD664,
# Satt_ASD102, Zhou_ASD404, Wang_NDD615, 2model_NDD_Wang, 2model_DD_Wang,
# 2model_ASD_Wang, Kaplanis_DD285, Sui_NDD686, ...): a gene has evidence
# from a source iff that cell is non-empty in its row. `gene` is the key
# (`hgnc_id`, when present, is not used); every other column is a source.

def load_ndd_evidence(db_dir, pattern="NDDgenes*", hgnc_id_to_symbol: dict = None) -> pd.DataFrame:
    """Boolean DataFrame indexed by gene (True = has evidence), columns =
    the file's own source column names. Empty DataFrame if the file or a
    `gene` column can't be found.

    `hgnc_id_to_symbol`, if given, backfills rows whose `gene`/`gene_all`
    cell is blank by resolving that row's `hgnc_id` instead -- some
    NDDgenes.txt releases (e.g. the version with an added Sui_NDD686
    column) only populate the gene-symbol cell for the newly-added source's
    rows, leaving every pre-existing row's symbol blank even though its
    evidence columns are populated; without the backfill those rows are
    silently dropped by the `gene_col != ""` filter below, undercounting
    every source except the new one. Pass e.g. gene_summary.build_hgnc_maps
    (hgnc.txt's hgnc_id -> approved symbol) to fix this."""
    f = _find_ci(db_dir, pattern)
    if not f:
        print("  NDD evidence: file not found - skipping")
        return pd.DataFrame()
    try:
        df = _read_table(f)
    except Exception as exc:  # unreadable/corrupt export shouldn't kill the run
        print(f"  NDD evidence: could not read {Path(f).name} ({exc}) - skipping")
        return pd.DataFrame()

    gene_col = _pick_col(df, ("gene", "gene_all"))
    if gene_col is None:
        print(f"  NDD evidence: no 'gene' column in {Path(f).name} "
              f"(columns: {list(df.columns)}) - skipping")
        return pd.DataFrame()

    resolved = df[gene_col].map(_clean)
    if hgnc_id_to_symbol:
        hgnc_col = _pick_col(df, ("hgnc_id",))
        if hgnc_col is not None:
            blank = resolved == ""
            n_before = blank.sum()
            backfilled = df.loc[blank, hgnc_col].map(_clean).map(lambda h: hgnc_id_to_symbol.get(h, ""))
            resolved.loc[blank] = backfilled
            n_fixed = (backfilled != "").sum()
            if n_before:
                print(f"  NDD evidence: backfilled {n_fixed}/{n_before} blank gene symbols via hgnc_id")
    df = df.assign(_resolved_gene=resolved)
    df = df[df["_resolved_gene"] != ""].drop_duplicates(subset="_resolved_gene")
    evidence_cols = [c for c in df.columns if c not in (gene_col, "_resolved_gene") and _norm(c) != "hgncid"]
    has_evidence = df[evidence_cols].apply(lambda s: s.map(_clean) != "")
    has_evidence.index = df["_resolved_gene"]
    print(f"  NDD evidence: {len(has_evidence)} genes x {len(evidence_cols)} "
          f"sources from {Path(f).name}")
    return has_evidence


def load_ndd(db_dir=None, pattern="NDDgenes*", evidence: pd.DataFrame = None) -> set:
    """Genes with >=1 hit across all sources in the NDD evidence matrix.

    Pass `evidence` (a `load_ndd_evidence` result) to reuse an
    already-loaded matrix instead of reading the file a second time."""
    ev = evidence if evidence is not None else load_ndd_evidence(db_dir, pattern)
    if ev.empty:
        return set()
    ndd = set(ev.index[ev.any(axis=1)])
    print(f"  NDD: {len(ndd)} genes")
    return ndd


def load_sui(db_dir=None, pattern="NDDgenes*", evidence: pd.DataFrame = None) -> set:
    """Genes with Sui_NDD686 evidence, straight from the same NDD evidence
    matrix as load_ndd -- no separate file, no gap.

    Pass `evidence` (a `load_ndd_evidence` result) to reuse an
    already-loaded matrix instead of reading the file a second time."""
    ev = evidence if evidence is not None else load_ndd_evidence(db_dir, pattern)
    if ev.empty:
        return set()
    col = next((c for c in ev.columns if "sui" in c.lower()), None)
    if col is None:
        print("  Sui: no Sui_* column in the NDD evidence spreadsheet - skipping")
        return set()
    sui = set(ev.index[ev[col]])
    print(f"  Sui: {len(sui)} genes")
    return sui


def load_sfari_ndd(db_dir, sfari_pattern=None) -> tuple:
    return set(load_sfari_table(db_dir, sfari_pattern)["gene"]), load_ndd(db_dir)


def _load_hpa_genes(db_dir, pattern: str, label: str) -> set:
    f = _find(db_dir, pattern)
    if not f:
        print(f"  {label} not found - skipping")
        return set()
    genes = {_clean(g) for g in pd.read_csv(f, sep="\t", dtype=str).get("Gene", []) if _clean(g)}
    print(f"  {label}: {len(genes)} genes")
    return genes


def load_brain_enriched(db_dir) -> set:
    return _load_hpa_genes(db_dir, "HPA_tissue_category_rna_brain_Tissue_brain_enriched*", "Brain-enriched (HPA)")


def load_brain_expressed(db_dir) -> set:
    return _load_hpa_genes(db_dir, "HPA*brain_detected*", "Brain-expressed (HPA)")


def load_coexpression(db_dir) -> dict:
    f = _find(db_dir, "Soto_2025_S2A_BrainSpan.xlsx")
    if not f:
        print("  Coexpression not found - skipping")
        return {}
    df = pd.read_excel(f, dtype=str)
    sym = next((c for c in df.columns if "symbol" in c.lower()), None)
    mod = next((c for c in df.columns if "module" in c.lower() and "membership" not in c.lower()), None)
    if not sym or not mod:
        print("  Coexpression: can't find symbol/module columns")
        return {}
    mapping = {_clean(r[sym]): _clean(r[mod]) for _, r in df.iterrows() if _clean(r[sym])}
    print(f"  Coexpression: {len(mapping)} genes")
    return mapping


def load_ppi(genes, ppi_file=None) -> dict:
    local = Path(ppi_file) if ppi_file else None
    if local and local.exists():
        df = pd.read_csv(local, sep="\t", dtype=str)
    elif not HAS_REQUESTS:
        print("  requests not installed - can't fetch STRING")
        return {}
    elif len(genes) >= 2000:
        print(f"  {len(genes)} genes - too many for STRING API")
        return {}
    else:
        glist = sorted(genes)
        r = _requests.post(_STRING_MAP, data={"identifiers": "\r".join(glist), "species": 9606,
                            "limit": 1, "echo_query": 1, "caller_identity": "combinations_pipeline"}, timeout=60)
        r.raise_for_status()
        id_df = pd.read_csv(StringIO(r.text), sep="\t")
        if id_df.empty or "stringId" not in id_df.columns:
            return {}
        ids = id_df["stringId"].dropna().tolist()
        r2 = _requests.post(_STRING_NET, data={"identifiers": "\r".join(ids), "species": 9606,
                             "required_score": PPI_MIN, "caller_identity": "combinations_pipeline"}, timeout=120)
        r2.raise_for_status()
        df = pd.read_csv(StringIO(r2.text), sep="\t")

    if df is None or df.empty:
        return {}

    rn = {}
    for col in df.columns:
        lo = col.lower().replace(" ", "")
        if lo in ("preferredname_a", "#node1", "gene_a"):
            rn[col] = "gene_a"
        elif lo in ("preferredname_b", "node2", "gene_b"):
            rn[col] = "gene_b"
        elif lo in ("score", "combined_score"):
            rn[col] = "score"
    df = df.rename(columns=rn)

    gs = {str(g) for g in genes}
    ppi = {}
    for _, row in df.iterrows():
        ga, gb = str(row.get("gene_a", "")), str(row.get("gene_b", ""))
        if not ga or not gb or ga not in gs or gb not in gs:
            continue
        try:
            s = float(row.get("score", 0))
        except ValueError:
            s = 0.0
        if s <= 1.0:
            s *= 1000
        ppi[(ga, gb)] = s
        ppi[(gb, ga)] = s

    if ppi and not (local and local.exists()):
        save = Path(ppi_file) if ppi_file else Path("string_interactions.tsv")
        rows = [{"gene_a": a, "gene_b": b, "score": s} for (a, b), s in ppi.items() if a < b]
        if rows:
            pd.DataFrame(rows).to_csv(save, sep="\t", index=False)
            print(f"  PPI saved: {save}")

    print(f"  {len(ppi)//2} interactions loaded")
    return ppi


def ppi_score(g1: str, g2: str, ppi: dict) -> float:
    return max(ppi.get((g1, g2), 0.0), ppi.get((g2, g1), 0.0))


def coexpressed(g1: str, g2: str, coexpr: dict) -> bool:
    return bool(coexpr) and bool(coexpr.get(g1)) and coexpr.get(g1) == coexpr.get(g2)


def load_reactome(db_dir) -> list:
    f = _find(db_dir, "ReactomePathways.gmt*")
    if not f:
        return []
    pathways = []
    for line in open(f):
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            pathways.append({"pathway": parts[0], "genes": frozenset(parts[2:])})
    print(f"  Reactome: {len(pathways)} pathways")
    return pathways


def load_go_mf(db_dir, max_genes: int = 500) -> dict:
    f = _find(db_dir, "Rgene2go_v2*")
    if not f:
        return {}
    go = pd.read_csv(f, sep="\t", dtype=str)
    go = go[go["ONTOLOGY"].str.strip().str.upper() == "MF"]
    idx = {}
    for (_, term), grp in go.groupby(["ONTOLOGY", "TERM"]):
        genes = frozenset(grp["SYMBOL"].str.strip())
        if len(genes) <= max_genes:
            idx[f"GO:MF - {term.strip()}"] = genes
    print(f"  GO:MF: {len(idx)} terms")
    return idx