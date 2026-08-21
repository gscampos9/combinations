"""
"""

import bisect
import random
import sys
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "simulation"  # comb/simulation -- data lives there, code here
CDS_BED = str(DATA_DIR / "ncbiRefSeqSelect_noalt.bed")        # chrom, start, end, gene, ...
VARIANTS_BED = str(DATA_DIR / "LGDMIS_DNM_PIV_hg38_proband_variants.bed")
EXCLUDE_BED = str(DATA_DIR / "sorted_SD_recent_repeat_LCR_gap_cen_par_hg38.bed")
PAIRS = str(DATA_DIR / "gene_pairs.txt")                # geneA,geneB (sem header)
OBS_OUT = str(DATA_DIR / "observed_counts.tsv")
NULL_OUT = str(DATA_DIR / "null_distributions_per_chr.tsv")
N_ITER = 10000
SEED = 42


def load_cds_with_gene(path):
    pieces = []
    for line in open(path):
        f = line.split()
        if len(f) < 4:
            continue
        chrom, start, end, gene = f[0], int(f[1]), int(f[2]), f[3]
        if end > start:
            pieces.append((chrom, start, end, gene))
    return pieces


def merged_index(path):
    by_chrom = defaultdict(list)
    for line in open(path):
        f = line.split()
        if len(f) < 3:
            continue
        chrom, start, end = f[0], int(f[1]), int(f[2])
        if end > start:
            by_chrom[chrom].append((start, end))
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


def subtract(pieces, exclude):
    out = []
    for chrom, start, end, gene in pieces:
        got = exclude.get(chrom)
        if not got:
            out.append((chrom, start, end, gene))
            continue
        starts, ends = got
        i = bisect.bisect_right(ends, start)
        cur = start
        while i < len(starts) and starts[i] < end:
            if starts[i] > cur:
                out.append((chrom, cur, min(starts[i], end), gene))
            cur = max(cur, ends[i])
            if cur >= end:
                break
            i += 1
        if cur < end:
            out.append((chrom, cur, end, gene))
    return out


def gene_info(pieces):
    """gene -> total CDS length, gene -> chromosome, gene -> list of real regions."""
    length = defaultdict(int)
    chrom_of = {}
    regions_of = defaultdict(list)
    for chrom, start, end, gene in pieces:
        length[gene] += end - start
        regions_of[gene].append((chrom, start, end))
        prev = chrom_of.get(gene)
        if prev is not None and prev != chrom:
            print(f"warning: {gene} has CDS pieces on multiple chromosomes "
                  f"({prev}, {chrom})", file=sys.stderr)
        chrom_of[gene] = chrom
    return length, chrom_of, regions_of


def build_patchworks(pieces):
    by_chrom = defaultdict(list)
    for chrom, start, end, _gene in pieces:
        by_chrom[chrom].append((chrom, start, end))
    patchworks = {}
    for chrom, pool in by_chrom.items():
        offsets, total = [], 0
        for _, s, e in pool:
            offsets.append(total)
            total += e - s
        patchworks[chrom] = (pool, offsets, total)
    return patchworks


def map_window(pool, offsets, total, r, length):
    regions = []
    remaining = length
    idx = bisect.bisect_right(offsets, r) - 1
    pos = r
    while remaining > 0:
        chrom, pstart, pend = pool[idx]
        offset_in_piece = pos - offsets[idx]
        avail = (pend - pstart) - offset_in_piece
        take = min(avail, remaining)
        s = pstart + offset_in_piece
        regions.append((chrom, s, s + take))
        remaining -= take
        idx += 1
        if idx == len(pool):
            idx = 0
            pos = 0
        else:
            pos = offsets[idx]
    return regions


def sample_regions(pool, offsets, total, target, rng):
    r = rng.randrange(total)
    return map_window(pool, offsets, total, r, target)


def load_variants(path):
    by_chrom = defaultdict(list)
    for line in open(path):
        f = line.split()
        if len(f) >= 3:
            by_chrom[f[0]].append(int(f[1]))
    return {c: sorted(v) for c, v in by_chrom.items()}


def load_pairs(path):
    pairs = []
    for line in open(path):
        f = line.strip().split(",")
        if len(f) < 2:
            continue
        gene1, gene2 = f[0], f[1]
        pairs.append((f"{gene1}_{gene2}", gene1, gene2))
    return pairs


def count_variants(regions, variants):
    n = 0
    for chrom, start, end in regions:
        pos = variants.get(chrom)
        if pos:
            n += bisect.bisect_left(pos, end) - bisect.bisect_left(pos, start)
    return n


def sample_pair_regions(patchworks, gene_length, gene_chrom, gene1, gene2, rng):
    regions = []
    for gene in (gene1, gene2):
        chrom = gene_chrom[gene]
        pool, offsets, total = patchworks[chrom]
        length = gene_length[gene]
        regions.extend(sample_regions(pool, offsets, total, length, rng))
    return regions


def main():
    rng = random.Random(SEED)
    pieces = load_cds_with_gene(CDS_BED)

    if EXCLUDE_BED:
        before = sum(e - s for _, s, e, _ in pieces)
        pieces = subtract(pieces, merged_index(EXCLUDE_BED))
        after = sum(e - s for _, s, e, _ in pieces)
        print(f"pool: {before} bp -> {after} bp after exclusion "
              f"({100.0 * (before - after) / before:.2f}% removed)", file=sys.stderr)

    if not pieces:
        sys.exit("empty CDS pool")

    gene_length, gene_chrom, gene_regions = gene_info(pieces)
    patchworks = build_patchworks(pieces)

    variants = load_variants(VARIANTS_BED)
    pairs = load_pairs(PAIRS)

    with open(OBS_OUT, "w") as obs_fh, open(NULL_OUT, "w") as null_fh:
        obs_fh.write("comb\tgene1\tgene2\tlength1\tlength2\tn_obs\n")
        null_fh.write("comb\titeration\tnull_count\n")

        for i, (comb, gene1, gene2) in enumerate(pairs, 1):
            if gene1 not in gene_length or gene2 not in gene_length:
                print(f"skip {comb}: gene(s) not found in CDS bed", file=sys.stderr)
                continue

            n_obs = count_variants(gene_regions[gene1] + gene_regions[gene2], variants)
            obs_fh.write(f"{comb}\t{gene1}\t{gene2}\t{gene_length[gene1]}\t{gene_length[gene2]}\t{n_obs}\n")
            obs_fh.flush()

            for j in range(N_ITER):
                regions = sample_pair_regions(patchworks, gene_length, gene_chrom, gene1, gene2, rng)
                count = count_variants(regions, variants)
                null_fh.write(f"{comb}\t{j}\t{count}\n")

            print(f"[{i}/{len(pairs)}] {comb} n_obs={n_obs} done ({N_ITER} iterations)", file=sys.stderr)


if __name__ == "__main__":
    main()