"""
Shared style, stats helpers, and data loaders for the comb/figures/ publication
scripts (fig1_variant_count.py, fig2_gene_length.py, fig3_feature_comparison.py, ...).

Reuses existing, tested logic from scripts/core/ and scripts/common/ rather than reimplementing it:
  - core.annotate_combinations.load_stats          -- parses the quoted-CSV
    RareComb output tables (CADD20_2_2_1_output*.txt)
  - core.annotate_combinations._shared_and_jaccard  -- generic GO/Reactome
    shared-terms + Jaccard math
  - core.gene_summary.build_hgnc_maps / resolve_symbol -- HGNC alias/prev-symbol
    fallback matching for genes gnomAD doesn't match directly
  - common.db_loaders.load_ppi / ppi_score / load_coexpression -- STRING PPI
    (local-file-or-live-API) and BrainSpan coexpression-module loading

Only genuinely new logic lives here: the streaming parser for the 877MB
RareComb input matrix, the 3-500-gene GO term index (existing code only has a
one-sided <=300 bound), and small stats/style helpers.
"""
from __future__ import annotations

import glob as _glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as _stats
from statsmodels.stats.multitest import multipletests

# ── paths ────────────────────────────────────────────────────────────────────
# Priority: COMB_PROJECT_ROOT env var > the cluster's known fixed location (if
# this happens to be running there) > inferred from this file's own location
# (.../<project>/scripts/figures/_common.py -> parents[2] is <project>), which
# is right wherever else the repo is checked out.
_CLUSTER_PROJECT_ROOT = Path("/net/eichler/vol28/home/gscampos/nobackups/projects/combinations")
if os.environ.get("COMB_PROJECT_ROOT"):
    PROJECT_ROOT = Path(os.environ["COMB_PROJECT_ROOT"])
elif _CLUSTER_PROJECT_ROOT.is_dir():
    PROJECT_ROOT = _CLUSTER_PROJECT_ROOT
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
def _resolve_dir(env_var: str, name: str, *candidates: Path) -> Path:
    """env var override > first existing candidate dir > first candidate
    (so a missing dir still gets a sensible path in the error message)."""
    if os.environ.get(env_var):
        return Path(os.environ[env_var])
    for cand in candidates:
        if cand.is_dir():
            return cand
    return candidates[0]


FIGURES_DIR = PROJECT_ROOT / "figures"
CACHE_DIR = FIGURES_DIR / "_cache"                            # auto-generated, safe to delete
RESULTS_221 = PROJECT_ROOT / "results_221"
# simulation/ is a top-level sibling of results_221/ on this dev machine, but
# nested at results_221/simulation/ on at least one other checkout (the
# cluster) -- tried both, whichever exists wins. COMB_SIMULATION_DIR
# overrides.
SIMULATION_DIR = _resolve_dir("COMB_SIMULATION_DIR", "simulation",
                              PROJECT_ROOT / "simulation", RESULTS_221 / "simulation")
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# DB_DIR/STRING_DIR: tried inside PROJECT_ROOT first (e.g. combinations/databases/
# on the cluster), then as a sibling of PROJECT_ROOT (this dev machine's layout:
# ASD_project/{comb,databases,string_networks}) -- whichever actually exists on
# disk wins. COMB_DB_DIR/COMB_STRING_DIR env vars override both, no code edit
# needed: `COMB_DB_DIR=/path/to/databases python fig1_...py`.
DB_DIR = _resolve_dir("COMB_DB_DIR", "databases",
                      PROJECT_ROOT / "databases", PROJECT_ROOT.parent / "databases")
STRING_DIR = _resolve_dir("COMB_STRING_DIR", "string_networks",
                          PROJECT_ROOT / "string_networks", PROJECT_ROOT.parent / "string_networks")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CASE_OUTPUT = RESULTS_221 / "CADD20_2_2_1_output.txt"
CONTROL_OUTPUT = RESULTS_221 / "CADD20_2_2_1_output_control.txt"
INPUT_MATRIX = RESULTS_221 / "CADD20_2_2_1_input.txt"


def _find_db(pattern: str) -> str:
    matches = sorted(_glob.glob(str(DB_DIR / pattern)))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} in {DB_DIR}")
    return matches[-1]


GNOMAD_PATH = _find_db("gnomad*constraint_metrics*")  # version/compression-agnostic (v4.1.1.tsv.bgz here; other releases differ)
HGNC_PATH = _find_db("hgnc_complete_set*")
GO_PATH = _find_db("Rgene2go_v2*")

PARSED_DATASET = RESULTS_221 / "parsed_dataset.tsv"
AFFECTED_PARENTS = PROJECT_ROOT / "affected_parents.txt"
PSYCHENCODE_PATH = _find_db("INT-09_WGCNA_modules_hgnc_ids*")

sys.path.insert(0, str(SCRIPTS_DIR))
from core.annotate_combinations import (  # noqa: E402
    load_stats, _shared_and_jaccard, load_g2p_dd, load_clingen,
)
from core.gene_summary import build_hgnc_maps, resolve_symbol  # noqa: E402
from core.annotate_inheritance import build_origin_instances  # noqa: E402
from common.db_loaders import (  # noqa: E402
    load_ppi, ppi_score, load_sfari_scores, load_ndd_evidence, load_ndd, load_sui,
    load_brain_expressed, load_brain_enriched, load_reactome,
)
from common.io_utils import ORIGIN_COLS, filter_rrvs  # noqa: E402

__all__ = [
    "PROJECT_ROOT", "FIGURES_DIR", "RESULTS_221", "SIMULATION_DIR",
    "CASE_OUTPUT", "CONTROL_OUTPUT", "INPUT_MATRIX", "DB_DIR", "STRING_DIR",
    "PARSED_DATASET", "AFFECTED_PARENTS", "ORIGIN_COLS",
    "CASE_COLOR", "CONTROL_COLOR", "set_style", "panel_label",
    "stars", "mannwhitney", "bh_adjust", "draw_sig_bracket", "add_mean_markers", "savefig", "save_panels",
    "load_stats", "combo_pairs", "parse_input_matrix",
    "build_gene_table", "build_go_term_index",
    "load_ppi_for_genes", "ppi_score", "coexpressed_pair", "load_coexpression_modules",
    "build_inheritance_origin_table", "build_database_hits", "load_ndd_evidence", "load_permutation_results",
    "pair_features", "fisher_fraction", "violin", "annotate_bracket", "grouped_bar",
    "load_parsed_dataset", "variant_counts_per_individual", "filter_rrvs",
    "build_term_universe", "pathway_enrichment",
]

# ── style ────────────────────────────────────────────────────────────────────
CASE_COLOR = "#2a78d6"      # blue -- same hex already used for "LGD" in the archived plot_panel.py palette
CONTROL_COLOR = "#B0B0B0"   # matches the grey already used in plot_length_ocurrance.py


def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9, "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black", "axes.linewidth": 0.8,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def panel_label(ax, letter: str):
    ax.text(-0.15, 1.08, letter, transform=ax.transAxes, fontsize=13,
             fontweight="bold", va="top", ha="left")


def savefig(fig, name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    print(f"  saved {path}")


def save_panels(fig, panels, name: str):
    """Save each panel of an already-built multi-panel figure as its own
    standalone PDF (`{name}_A.pdf`, `{name}_B.pdf`, ...), in addition to
    the combined savefig(). Call after the figure is fully drawn (all
    panel_label/title/text calls done) and before plt.close(fig).

    `panels` is either an iterable of Axes (letters A, B, C, ... assigned
    in order) or an iterable of (ax, letter) pairs, e.g.:
        c.save_panels(fig, axes, "Fig1_variant_count")
        c.save_panels(fig, zip(axes, "ABC"), "Fig7_pathway_enrichment")
        c.save_panels(fig, [(ax_mis_z, "A"), (ax_pli, "B")], "Fig3_feature_comparison")

    Each panel's own artists (title, axis labels, panel_label letter,
    significance brackets, legends) are captured via get_tightbbox();
    figure-level artists (suptitle, fig.text() section headers) are not
    part of any single panel's bbox and so are correctly left out."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    panels = list(panels)
    if panels and not isinstance(panels[0], tuple):
        panels = list(zip(panels, (chr(ord("A") + i) for i in range(len(panels)))))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, letter in panels:
        bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        path = FIGURES_DIR / f"{name}_{letter}.pdf"
        fig.savefig(path, bbox_inches=bbox.padded(0.1))
        print(f"  saved {path}")


# ── stats ────────────────────────────────────────────────────────────────────
def stars(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    if p < 0.0001:
        return "****"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def mannwhitney(a, b):
    """Two-sided Mann-Whitney U on two 1-D arrays (NaNs dropped).
    Returns (U, p, rank_biserial_effect_size, n_a, n_b)."""
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float)
    b = b[~np.isnan(b)]
    U, p = _stats.mannwhitneyu(a, b, alternative="two-sided", method="auto")
    r = 1 - (2 * U) / (len(a) * len(b))
    return U, p, r, len(a), len(b)


def bh_adjust(pvals) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    adj = np.full_like(pvals, np.nan)
    mask = ~np.isnan(pvals)
    if mask.sum():
        adj[mask] = multipletests(pvals[mask], method="fdr_bh")[1]
    return adj


def _format_p(p: float) -> str:
    if p <= 0:  # underflowed to exactly 0 in floating point -- report a floor, not a false exact zero
        return "p<1e-300"
    return f"p={p:.2g}" if p >= 0.001 else f"p={p:.1e}"


def _format_delta(delta: float) -> str:
    sign = "+" if delta >= 0 else ""
    return f"Δ={sign}{_fmt_mean(delta)}"


def draw_sig_bracket(ax, x1: float, x2: float, y: float, text: str, height=None, pvalue=None,
                      delta=None, test_name=None):
    """Draw a horizontal significance bracket between x1/x2 at height y.
    `text` (the star tier) is always shown. `delta` (group A - group B) and
    `test_name` (which test produced `pvalue`), when given, are always shown
    too -- the actual p-value is only added as its own line when < 0.05
    (stars/delta/test name are shown regardless of significance; the exact
    p-value is reserved for when it's actually significant)."""
    if height is None:
        y0, y1 = ax.get_ylim()
        height = (y1 - y0) * 0.02
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y],
             color="black", linewidth=0.9)
    lines = [text]
    if delta is not None and not np.isnan(delta):
        lines.append(_format_delta(delta))
    if pvalue is not None and not np.isnan(pvalue) and pvalue < 0.05:
        lines.append(_format_p(pvalue))
    if test_name:
        lines.append(test_name)
    ax.text((x1 + x2) / 2, y + height, "\n".join(lines), ha="center", va="bottom",
             fontsize=7, linespacing=1.35)


def add_mean_markers(ax, groups: list, positions: list, color="white",
                     edgecolor="black", marker="D", size=22, zorder=5):
    """Overlay a diamond at the mean of each group on a box plot (the box
    itself already marks the median/quartiles -- the mean is drawn
    separately here so both are always visible)."""
    for pos, vals in zip(positions, groups):
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        ax.scatter([pos], [vals.mean()], marker=marker, s=size, color=color,
                   edgecolor=edgecolor, linewidth=0.9, zorder=zorder)


# ── Shared two-group feature-comparison plotting (Fig3 case/control, Fig5
# before/after-permutation) -- one set of helpers, reused by both so the
# visual language (mean marker, p-value text, N labelling) never drifts ────
def pair_features(df: pd.DataFrame, go_index: dict, ppi: dict, modules: dict) -> pd.DataFrame:
    """Per-pair GO Jaccard / STRING PPI / same-coexpression-module for every
    (gene1, gene2) row of a load_stats() result."""
    rows = []
    for g1, g2 in zip(df["gene1"], df["gene2"]):
        _shared, jacc = _shared_and_jaccard(g1, g2, go_index)
        ppi_val = ppi_score(g1, g2, ppi) / 1000.0  # STRING internal 0-1000 -> natural 0-1 scale
        coexpr = coexpressed_pair(g1, g2, modules)
        rows.append({"gene1": g1, "gene2": g2, "go_jaccard": jacc, "ppi_score": ppi_val,
                     "coexpressed": coexpr})
    return pd.DataFrame(rows)


def fisher_fraction(mask_a, mask_b):
    """% of each group meeting a boolean criterion, plus the Fisher's exact
    p-value that the two proportions differ."""
    a_hi, a_lo = int(mask_a.sum()), int((~mask_a).sum())
    b_hi, b_lo = int(mask_b.sum()), int((~mask_b).sum())
    odds, p = _stats.fisher_exact([[a_hi, a_lo], [b_hi, b_lo]])
    return 100 * a_hi / len(mask_a), 100 * b_hi / len(mask_b), odds, p


def _fmt_mean(x: float) -> str:
    return f"{x:,.3g}"


def violin(ax, values_a, values_b, ylabel, title, label_a="Cases", label_b="Controls",
           color_a=None, color_b=None):
    """Boxplot + jittered individual points for two groups (name kept as
    `violin` for call-site compatibility across fig1/3/5/6 -- the geometry
    itself is boxplot+dots, not a violin/KDE, so every raw data point is
    visible rather than a smoothed density)."""
    color_a = CASE_COLOR if color_a is None else color_a
    color_b = CONTROL_COLOR if color_b is None else color_b
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    values_a = values_a[~np.isnan(values_a)]
    values_b = values_b[~np.isnan(values_b)]

    bp = ax.boxplot([values_a, values_b], positions=[0, 1], widths=0.45,
                     showfliers=False, patch_artist=True, zorder=3)
    for patch, color in zip(bp["boxes"], (color_a, color_b)):
        patch.set_facecolor("white")
        patch.set_edgecolor(color)
        patch.set_linewidth(1.3)
    for part in ("whiskers", "caps"):
        for i, line in enumerate(bp[part]):
            line.set_color(color_a if i < 2 else color_b)
            line.set_linewidth(1.1)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.1)

    rng = np.random.default_rng(0)
    for x, values, color in ((0, values_a, color_a), (1, values_b, color_b)):
        jitter = rng.uniform(-0.16, 0.16, size=len(values))
        ax.scatter(x + jitter, values, s=6, color=color, alpha=0.35, linewidths=0, zorder=2)

    ax.set_xticks([0, 1])
    mean_a, mean_b = np.nanmean(values_a), np.nanmean(values_b)
    ax.set_xticklabels([
        f"{label_a}\n(n={len(values_a):,}, mean={_fmt_mean(mean_a)})",
        f"{label_b}\n(n={len(values_b):,}, mean={_fmt_mean(mean_b)})",
    ])
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9.5)
    add_mean_markers(ax, [values_a, values_b], [0, 1])


def annotate_bracket(ax, values_a, values_b, stars_text, pvalue, test_name="Mann-Whitney U"):
    y_max = np.nanmax(np.concatenate([values_a, values_b]))
    y0, _ = ax.get_ylim()
    ax.set_ylim(y0, y_max * 1.42 if y_max > 0 else y_max * 0.7 + 1)
    delta = np.nanmean(values_a) - np.nanmean(values_b)
    draw_sig_bracket(ax, 0, 1, y_max * 1.05, stars_text, pvalue=pvalue, delta=delta, test_name=test_name)


def grouped_bar(ax, series_labels, pct_a, pct_b, n_a, n_b, ylabel, title, stars_texts, pvalues,
                label_a="Cases", label_b="Controls", color_a=None, color_b=None,
                test_name="Fisher's exact", counts_a=None, counts_b=None):
    """Grouped bar with a distinct N (per category, "n=X vs Y") baked into
    each x-tick label -- n_a/n_b may be a single number (broadcast to every
    category) or one value per category (when a category's N genuinely
    differs, e.g. missing data dropped for just that feature).

    `counts_a`/`counts_b`, when given, are the raw (non-percentage) count
    behind each bar's height -- printed as a small label on top of that bar
    so the plot reads in both percentage and absolute-count terms at once."""
    color_a = CASE_COLOR if color_a is None else color_a
    color_b = CONTROL_COLOR if color_b is None else color_b
    if np.isscalar(n_a):
        n_a = [n_a] * len(series_labels)
    if np.isscalar(n_b):
        n_b = [n_b] * len(series_labels)
    x = np.arange(len(series_labels))
    w = 0.36
    ax.bar(x - w / 2, pct_a, width=w, color=color_a, label=label_a)
    ax.bar(x + w / 2, pct_b, width=w, color=color_b, label=label_b)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\n(n={na:,} vs {nb:,})" for lab, na, nb in zip(series_labels, n_a, n_b)],
                        fontsize=7.6)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9.5)
    y_max = max(max(pct_a), max(pct_b))
    ax.set_ylim(0, y_max * 1.75 + 3)
    if counts_a is not None and counts_b is not None:
        for xi, pa, pb, ca, cb in zip(x, pct_a, pct_b, counts_a, counts_b):
            ax.text(xi - w / 2, pa + y_max * 0.015, f"{ca:,}", ha="center", va="bottom", fontsize=7)
            ax.text(xi + w / 2, pb + y_max * 0.015, f"{cb:,}", ha="center", va="bottom", fontsize=7)
    for xi, pa, pb, st, pv in zip(x, pct_a, pct_b, stars_texts, pvalues):
        top = max(pa, pb)
        draw_sig_bracket(ax, xi - w / 2, xi + w / 2, top + y_max * 0.07, st, pvalue=pv,
                          delta=pa - pb, test_name=test_name)
    ax.legend(fontsize=7, frameon=False, loc="upper left")


# ── RareComb output (case/control gene pairs) ────────────────────────────────
def combo_pairs(df: pd.DataFrame) -> set:
    """Set of sorted (gene1, gene2) tuples from a load_stats() result."""
    return set(tuple(sorted((a, b))) for a, b in zip(df["gene1"], df["gene2"]))


# ── RareComb boolean input matrix: one streaming pass -> burden + occurrence ─
_input_matrix_cache: dict = {}


def parse_input_matrix(path: Path = INPUT_MATRIX):
    """Single streaming pass over the ~877MB RareComb boolean input matrix.

    Returns (burden_df, occurrence_df):
      burden_df:      sample, n_genes_hit, group ("case"/"control")   -- one row/individual
      occurrence_df:   gene, n_occurrence_case, n_occurrence_control  -- one row/gene

    "n_genes_hit" / occurrence counts = number of GENES carrying >=1 qualifying
    rare variant (CADD>=20 LGD/MIS), not a raw variant count -- this boolean
    matrix has no way to distinguish 1 vs 2+ qualifying variants in the same
    gene for the same person.
    """
    path = Path(path)
    key = str(path)
    if key in _input_matrix_cache:
        return _input_matrix_cache[key]

    print(f"Parsing {path.name} (one streaming pass, ~1-2 min) ...")
    t0 = time.time()
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        genes = [c[len("Input_"):] for c in header[1:-1]]
        n_genes = len(genes)
        case_counts = np.zeros(n_genes, dtype=np.int64)
        control_counts = np.zeros(n_genes, dtype=np.int64)
        samples, burdens, groups = [], [], []

        for i, line in enumerate(fh, 1):
            fields = line.rstrip("\n").split("\t")
            arr = np.array(fields[1:-1], dtype=np.int8)
            is_case = fields[-1] == "1"
            samples.append(fields[0])
            burdens.append(int(arr.sum()))
            groups.append("case" if is_case else "control")
            if is_case:
                case_counts += arr
            else:
                control_counts += arr
            if i % 5000 == 0:
                print(f"  ... {i:,} rows")

    burden_df = pd.DataFrame({"sample": samples, "n_genes_hit": burdens, "group": groups})
    occurrence_df = pd.DataFrame({
        "gene": genes,
        "n_occurrence_case": case_counts,
        "n_occurrence_control": control_counts,
    })
    print(f"  done in {time.time() - t0:.0f}s "
          f"({len(burden_df):,} individuals, {len(occurrence_df):,} genes)")
    _input_matrix_cache[key] = (burden_df, occurrence_df)
    return burden_df, occurrence_df


# ── gnomAD constraint (CDS length, pLI, missense z-score), HGNC-resolved ─────
_gnomad_canon_cache = None
_hgnc_maps_cache = None


def _load_gnomad_canonical() -> pd.DataFrame:
    global _gnomad_canon_cache
    if _gnomad_canon_cache is not None:
        return _gnomad_canon_cache
    print(f"Loading gnomAD constraint metrics from {Path(GNOMAD_PATH).name} ...")
    usecols = ["gene", "canonical", "mane_select", "cds_length", "mis.z_score", "lof.pLI", "chromosome"]
    gnomad = pd.read_csv(GNOMAD_PATH, sep="\t", compression="gzip",
                          usecols=usecols, low_memory=False)
    # Same canonical/MANE-select selection as scripts/core/gene_summary.py,
    # extended to also carry pLI/mis_z/chromosome off the same selected transcript row.
    canon = gnomad[gnomad["canonical"].astype(str).str.strip().str.lower() == "true"].copy()
    canon["_mane"] = canon["mane_select"].astype(str).str.strip().str.lower() == "true"
    canon["cds_length"] = pd.to_numeric(canon["cds_length"], errors="coerce")
    canon = canon.sort_values(["_mane", "cds_length"], ascending=[False, False], na_position="last")
    canon = canon.drop_duplicates(subset="gene", keep="first").set_index("gene")
    _gnomad_canon_cache = canon[["cds_length", "mis.z_score", "lof.pLI", "chromosome"]]
    return _gnomad_canon_cache


def _load_hgnc_maps():
    global _hgnc_maps_cache
    if _hgnc_maps_cache is not None:
        return _hgnc_maps_cache
    hgnc = pd.read_csv(HGNC_PATH, sep="\t", low_memory=False)
    _hgnc_maps_cache = build_hgnc_maps(hgnc)
    return _hgnc_maps_cache


_hgnc_id_map_cache: dict = {}


def load_hgnc_id_to_symbol() -> dict:
    """hgnc_id -> current approved symbol, for backfilling gene-evidence
    files that only carry hgnc_id (see load_ndd_evidence's hgnc_id_to_symbol
    argument)."""
    if not _hgnc_id_map_cache:
        hgnc = pd.read_csv(HGNC_PATH, sep="\t", low_memory=False, usecols=["hgnc_id", "symbol"])
        _hgnc_id_map_cache.update(
            dict(zip(hgnc["hgnc_id"].astype(str).str.strip(), hgnc["symbol"]))
        )
    return _hgnc_id_map_cache


def build_gene_table(genes) -> pd.DataFrame:
    """genes -> DataFrame[gene, CDS_length, mis_z, pLI, chromosome, match_type],
    resolved via direct gnomAD match or HGNC alias/prev_symbol fallback."""
    canon = _load_gnomad_canonical()
    gnomad_genes = set(canon.index)
    approved, alias_map = _load_hgnc_maps()

    rows = []
    for gene in genes:
        matched, mtype = resolve_symbol(gene, gnomad_genes, approved, alias_map)
        if matched is not None:
            rec = canon.loc[matched]
            rows.append({
                "gene": gene,
                "CDS_length": rec["cds_length"],
                "mis_z": pd.to_numeric(rec["mis.z_score"], errors="coerce"),
                "pLI": pd.to_numeric(rec["lof.pLI"], errors="coerce"),
                "chromosome": rec["chromosome"],
                "match_type": mtype,
            })
        else:
            rows.append({"gene": gene, "CDS_length": np.nan, "mis_z": np.nan,
                         "pLI": np.nan, "chromosome": np.nan, "match_type": "unmatched"})
    return pd.DataFrame(rows)


# ── GO term index, 3-500 genes/term, all ontologies combined ─────────────────
_go_index_cache: dict = {}


def build_go_term_index(min_genes: int = 3, max_genes: int = 500) -> dict:
    """{gene -> set(GO id)}, restricted to GO terms annotated to between
    min_genes and max_genes distinct genes (all of BP/MF/CC combined -- one
    Jaccard number per gene pair rather than splitting by ontology)."""
    key = (min_genes, max_genes)
    if key in _go_index_cache:
        return _go_index_cache[key]
    print(f"Building GO term index ({min_genes}-{max_genes} genes/term, BP+MF+CC) "
          f"from {Path(GO_PATH).name} ...")
    go = pd.read_csv(GO_PATH, sep="\t", dtype=str, encoding="utf-8")
    go = go.drop_duplicates(subset=["SYMBOL", "GO"])
    term_genes = go.groupby("GO")["SYMBOL"].apply(set)
    sizes = term_genes.map(len)
    term_genes = term_genes[(sizes >= min_genes) & (sizes <= max_genes)]

    gene_terms: dict = {}
    for term, genes_ in term_genes.items():
        for g in genes_:
            gene_terms.setdefault(g, set()).add(term)
    print(f"  {len(term_genes):,} qualifying terms, {len(gene_terms):,} genes with >=1 term")
    _go_index_cache[key] = gene_terms
    return gene_terms


# ── Pathway/GO enrichment (hypergeometric), for a gene SET as a whole -- a
# different question from build_go_term_index's pairwise Jaccard above:
# "is this gene list enriched for term X" rather than "do these 2 genes
# share term X" ────────────────────────────────────────────────────────────
_term_universe_cache: dict = {}


def build_term_universe(min_genes: int = 3, max_genes: int = 500) -> pd.DataFrame:
    """One row per (source, term_id, term_name, genes) -- GO:BP/MF/CC (from
    Rgene2go_v2.txt) plus Reactome (ReactomePathways.gmt), restricted to
    terms with between min_genes and max_genes annotated genes."""
    key = (min_genes, max_genes)
    if key in _term_universe_cache:
        return _term_universe_cache[key]
    print(f"Building GO+Reactome term universe ({min_genes}-{max_genes} genes/term) ...")
    go = pd.read_csv(GO_PATH, sep="\t", dtype=str, encoding="utf-8")
    go = go.drop_duplicates(subset=["SYMBOL", "GO"])
    go_terms = go.groupby(["GO", "ONTOLOGY", "TERM"])["SYMBOL"].apply(set).reset_index()
    go_terms = go_terms.rename(columns={"GO": "term_id", "TERM": "term_name", "SYMBOL": "genes"})
    go_terms["source"] = "GO:" + go_terms["ONTOLOGY"]
    go_terms = go_terms[["source", "term_id", "term_name", "genes"]]

    reactome = load_reactome(str(DB_DIR))
    rx_rows = [{"source": "Reactome", "term_id": p["pathway"], "term_name": p["pathway"],
               "genes": set(p["genes"])} for p in reactome]
    rx_terms = pd.DataFrame(rx_rows)

    universe = pd.concat([go_terms, rx_terms], ignore_index=True)
    sizes = universe["genes"].map(len)
    universe = universe[(sizes >= min_genes) & (sizes <= max_genes)].reset_index(drop=True)
    print(f"  {len(universe):,} qualifying terms ({(universe['source'].str.startswith('GO')).sum():,} GO, "
          f"{(universe['source'] == 'Reactome').sum():,} Reactome)")
    _term_universe_cache[key] = universe
    return universe


def pathway_enrichment(query_genes, background_genes, min_genes: int = 3, max_genes: int = 500) -> pd.DataFrame:
    """Hypergeometric over-representation test: for every GO/Reactome term,
    is `query_genes` enriched for that term relative to `background_genes`
    (the universe the query was drawn from -- NOT necessarily the whole
    genome; pass whichever population makes the comparison meaningful)?
    One-sided (over-representation only), BH-adjusted across every term
    actually testable (>=1 background gene in the term)."""
    universe = build_term_universe(min_genes, max_genes)
    background = set(background_genes)
    query = set(query_genes) & background
    n_query = len(query)
    n_background = len(background)

    rows = []
    for row in universe.itertuples():
        term_in_background = row.genes & background
        K = len(term_in_background)
        if K == 0:
            continue
        k = len(query & term_in_background)
        if k == 0:
            continue
        p = _stats.hypergeom.sf(k - 1, n_background, K, n_query)
        rows.append({
            "source": row.source, "term_id": row.term_id, "term_name": row.term_name,
            "n_query_hits": k, "n_query": n_query, "n_term_in_background": K,
            "n_background": n_background, "fold_enrichment": (k / n_query) / (K / n_background),
            "genes": ",".join(sorted(query & term_in_background)), "p_raw": p,
        })
    result = pd.DataFrame(rows)
    if len(result):
        result["p_bh"] = bh_adjust(result["p_raw"].values)
        result = result.sort_values("p_bh")
    return result


# ── PPI (STRING): local short list is unrelated/tiny (60 genes) for our gene
# universe, so use load_ppi's live-API path, cached to disk after first pull ──
def load_ppi_for_genes(genes) -> dict:
    """STRING combined_score (0-1000 internal scale, see common.db_loaders.
    ppi_score) for the given gene set. First call fetches from the live
    STRING API (v12 confirmed reachable) and caches to CACHE_DIR; later
    calls/runs read the cached file so results stay reproducible and the API
    isn't re-hit every run."""
    cache_path = CACHE_DIR / "string_ppi_pull.tsv"
    print(f"Loading STRING PPI for {len(genes)} genes "
          f"({'from cache' if cache_path.exists() else 'live API pull, may take ~1 min'}) ...")
    ppi = load_ppi(set(genes), ppi_file=str(cache_path))
    return ppi


# ── Coexpression: PsychENCODE INT-09 WGCNA modules, pair-level ───────────────
# Replaces the earlier BrainSpan/Soto gene-level "module size - 1" metric,
# which wasn't a clear/intuitive quantity. This is deliberately as simple as
# possible: are the two genes of a pair in the SAME module, yes/no -- no
# hidden module-size exclusion. Reported as % of pairs, the same way the PPI
# panels report % of pairs above a threshold.
_psychencode_module_cache = None


def load_coexpression_modules() -> dict:
    """gene -> WGCNA module id, from PsychENCODE's INT-09_WGCNA_modules_
    hgnc_ids.xlsx (http://resource.psychencode.org/Datasets/Integrative/
    ModelParams/). Only "Sheet1" has data (Sheet2-6 are empty tabs in the
    source file, confirmed by inspection) -- one row per module: module name
    in column A, member gene symbols filling the rest of that row (a ragged
    row per module, not a normal rectangular table)."""
    global _psychencode_module_cache
    if _psychencode_module_cache is not None:
        return _psychencode_module_cache
    print(f"Loading PsychENCODE coexpression modules from {Path(PSYCHENCODE_PATH).name} ...")
    raw = pd.read_excel(PSYCHENCODE_PATH, sheet_name="Sheet1", header=None)
    module_rows = raw.iloc[2:]  # row 0 = single 'hgnc_Ids' label, row 1 = blank
    gene_to_module: dict = {}
    for _, row in module_rows.iterrows():
        vals = row.dropna().tolist()
        if len(vals) < 2:
            continue
        module = str(vals[0])
        for gene in vals[1:]:
            gene_to_module[str(gene).strip()] = module
    print(f"  {module_rows.iloc[:, 0].notna().sum():,} modules, {len(gene_to_module):,} genes")
    _psychencode_module_cache = gene_to_module
    return gene_to_module


def coexpressed_pair(g1: str, g2: str, modules: dict):
    """True/False, or None if either gene isn't in the module table at all
    (kept separate from False so "no data" isn't silently counted as "not
    coexpressed" upstream)."""
    m1, m2 = modules.get(g1), modules.get(g2)
    if m1 is None or m2 is None:
        return None
    return m1 == m2


# ── Gene-list databases: SFARI, NDD evidence, G2P_DD, ClinGen, HPA brain ─────
_db_cache: dict = {}


def _load_database_sets():
    if not _db_cache:
        print("Loading gene-list databases (SFARI/NDD/G2P_DD/ClinGen/HPA brain) ...")
        _db_cache["sfari"] = load_sfari_scores(str(DB_DIR))
        ndd_ev = load_ndd_evidence(str(DB_DIR), pattern=["NDDgenes_updated*", "NDDgenes*"],
                                    hgnc_id_to_symbol=load_hgnc_id_to_symbol())
        _db_cache["ndd"] = load_ndd(evidence=ndd_ev)
        _db_cache["sui"] = load_sui(evidence=ndd_ev)          # Sui_NDD686 -- reported, not folded into ndd_gene
        _db_cache["g2p"] = load_g2p_dd(str(DB_DIR))[0]        # gene -> confidence level
        _db_cache["clingen"] = load_clingen(str(DB_DIR))[0]   # gene -> classification
        _db_cache["brain_expressed"] = load_brain_expressed(str(DB_DIR))
        _db_cache["brain_enriched"] = load_brain_enriched(str(DB_DIR))
    return _db_cache


def build_database_hits(genes) -> pd.DataFrame:
    """Per-gene hits across SFARI / NDD gene-evidence / G2P_DD / ClinGen /
    HPA brain-expressed / HPA brain-enriched -- the "databases" layer of the
    gene-pair annotation table.

    `sui_hc` (Sui_NDD686 evidence) is reported alongside `ndd_gene` but is
    NOT folded into it -- ndd_gene stays the union across all NDDgenes.txt
    sources; sui_hc is its own column so the Sui-specific list can be
    inspected/filtered on separately."""
    db = _load_database_sets()
    rows = []
    for g in genes:
        rows.append({
            "gene": g,
            "sfari_score": db["sfari"].get(g, ""),
            "ndd_gene": g in db["ndd"],
            "sui_hc": g in db["sui"],
            "g2p_dd_confidence": db["g2p"].get(g, ""),
            "clingen_classification": db["clingen"].get(g, ""),
            "brain_expressed": g in db["brain_expressed"],
            "brain_enriched": g in db["brain_enriched"],
        })
    return pd.DataFrame(rows)


# ── Parsed per-variant dataset, and per-individual variant-count helpers ────
_parsed_dataset_cache = None


def _load_parsed_dataset() -> pd.DataFrame:
    global _parsed_dataset_cache
    if _parsed_dataset_cache is None:
        print(f"Loading {PARSED_DATASET.name} ...")
        _parsed_dataset_cache = pd.read_csv(PARSED_DATASET, sep="\t", low_memory=False)
    return _parsed_dataset_cache


load_parsed_dataset = _load_parsed_dataset  # public alias


def variant_counts_per_individual(variant_rows: pd.DataFrame, all_individuals: pd.DataFrame) -> pd.DataFrame:
    """Row count per spid in `variant_rows` (e.g. already filtered to a
    consequence/origin subset), reindexed against the FULL cohort in
    `all_individuals` (columns spid, pheno) so individuals with zero
    matching variants show up as 0 rather than being silently dropped."""
    counts = variant_rows.groupby("spid").size()
    out = all_individuals.drop_duplicates("spid").set_index("spid")[["pheno"]].copy()
    out["n_variants"] = counts.reindex(out.index).fillna(0).astype(int)
    return out.reset_index()


_inheritance_cache: dict = {}


def build_inheritance_origin_table(rarecomb_output_path, case_value: str) -> pd.DataFrame:
    """Per-(spid, gene pair) raw inheritance-origin-category instances --
    DD/DM/DP/MP/MM/PP (both de novo, de novo+maternal, de novo+paternal,
    maternal+paternal, both maternal, both paternal) -- for every individual
    RareComb counted as carrying a gene pair.

    case_value: "asd" for the case-direction output (Case_Samples = probands)
    or "sib" for the reversed control-direction output (Case_Samples =
    siblings -- confirmed earlier from that column's .s-suffixed spids).

    Calls annotate_inheritance.py's build_origin_instances() directly, not
    its flag_selection() step: flag_selection hardcodes a proband-only regex
    and an "affected-parent same-source" validity rule that's specific to
    scoring proband disease-relevance, with no clean sibling/control
    analogue -- so this reports the raw origin-category assignment for
    BOTH groups rather than that case-only "valid support" filter.

    Memoized (both Fig6 and build_annotations.py call this for both
    directions -- ~150s of real work per direction, not worth repeating
    within one run_all.py process).
    """
    key = (str(rarecomb_output_path), case_value)
    if key in _inheritance_cache:
        return _inheritance_cache[key]
    df = _load_parsed_dataset()
    result = build_origin_instances(
        df, str(rarecomb_output_path), str(AFFECTED_PARENTS),
        category_col="pheno", case_value=case_value, cadd_cutoff=20.0,
    )
    _inheritance_cache[key] = result
    return result


# ── Length-permutation results (simulation/, case-direction only) ───────────
def load_permutation_results() -> pd.DataFrame:
    """simulation/simulation_results_empirical.tsv, unchanged. Only the 477
    case gene pairs were simulated (464 with a usable result) -- there is
    no permutation test for control gene pairs."""
    return pd.read_csv(SIMULATION_DIR / "simulation_results_empirical.tsv", sep="\t")
