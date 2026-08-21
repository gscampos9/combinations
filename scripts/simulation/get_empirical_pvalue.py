#!/usr/bin/env python3

import math
from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests

DATA_DIR = Path(__file__).resolve().parents[2] / "simulation"  # comb/simulation -- data lives there, code here
OBS = str(DATA_DIR / "observed_counts.tsv")          # comb, gene1, gene2, length1, length2, n_obs
NULL = str(DATA_DIR / "null_distributions_per_chr.tsv")      # comb, iteration, null_count
OUT = str(DATA_DIR / "simulation_results_empirical.tsv")

# -------------------------------------
# Read inputs
# -------------------------------------

obs = pd.read_csv(OBS, sep="\t")
null = pd.read_csv(NULL, sep="\t")

rows = []

for _, pair in obs.iterrows():

    comb = pair["comb"]
    gene1 = pair["gene1"]
    gene2 = pair["gene2"]
    length1 = pair["length1"]
    length2 = pair["length2"]
    n_obs = pair["n_obs"]

    x = null.loc[null["comb"] == comb, "null_count"].values

    mean = x.mean()
    sd = x.std(ddof=0)

    if sd > 0:
        z = (n_obs - mean) / sd
    else:
        z = float("nan")

    # one-sided empirical p (enrichment)
    p_one = ((x >= n_obs).sum() + 1) / (len(x) + 1)

    # two-sided empirical p
    if z >= 0:
        p_two = 2 * p_one
    else:
        p_two = 2 * (((x <= n_obs).sum() + 1) / (len(x) + 1))

    p_two = min(1.0, p_two)

    rows.append({
        "comb": comb,
        "gene1": gene1,
        "gene2": gene2,
        "length1": length1,
        "length2": length2,
        "n_obs": n_obs,
        "null_mean": mean,
        "null_sd": sd,
        "z": z,
        "p_onesided": p_one,
        "p_two_sided": p_two,
    })

res = pd.DataFrame(rows)

res["q_bh_onesided"] = multipletests(
    res["p_onesided"],
    method="fdr_bh"
)[1]

res["q_bh_two_sided"] = multipletests(
    res["p_two_sided"],
    method="fdr_bh"
)[1]

res.to_csv(OUT, sep="\t", index=False)

print(f"Saved {OUT}")
print(f"One-sided BH <0.05 : {(res.q_bh_onesided < 0.05).sum()}")
print(f"Two-sided BH <0.05 : {(res.q_bh_two_sided < 0.05).sum()}")