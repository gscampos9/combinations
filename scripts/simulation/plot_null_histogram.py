"""Plot null distributions and the observed count for three example pairs:
most variants, fewest variants, and a median one."""

import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_DIR = Path(__file__).resolve().parents[2] / "simulation"  # comb/simulation -- data lives there, code here
RESULTS = str(DATA_DIR / "simulation_results_empirical.tsv")   # comb gene1 gene2 length1 length2 n_obs ... null_mean null_sd z p_onesided
NULL_FILE = str(DATA_DIR / "null_distributions.tsv")    # comb iteration null_count
OUT_PNG = str(DATA_DIR / "null_examples.png")


def normal_pdf(x, mean, sd):
    return math.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))


stats = {}
with open(RESULTS) as fh:
    header = fh.readline().rstrip("\n").split("\t")
    col = {name: i for i, name in enumerate(header)}
    for line in fh:
        f = line.rstrip("\n").split("\t")
        if f[col["z"]] == "NA":
            continue
        stats[f[col["comb"]]] = {
            "length": int(f[col["length1"]]) + int(f[col["length2"]]),
            "n_obs": int(f[col["n_obs"]]),
            "mean": float(f[col["null_mean"]]),
            "sd": float(f[col["null_sd"]]),
            "z": float(f[col["z"]]),
            "p": float(f[col["p_onesided"]]),
        }

by_z = sorted(stats, key=lambda c: stats[c]["z"])
picks = [
    ("most depleted", by_z[0]),
    ("closest to null", min(stats, key=lambda c: abs(stats[c]["z"]))),
    ("most enriched", by_z[-1]),
]
wanted = {c for _, c in picks}

null = defaultdict(list)
with open(NULL_FILE) as fh:
    next(fh)
    for line in fh:
        f = line.split()
        if len(f) >= 3 and f[0] in wanted:
            null[f[0]].append(int(f[2]))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

for ax, (label, comb) in zip(axes, picks):
    counts = null[comb]
    s = stats[comb]
    mean, sd, n_obs = s["mean"], s["sd"], s["n_obs"]

    lo, hi = min(min(counts), n_obs), max(max(counts), n_obs)
    ax.hist(counts, bins=range(lo, hi + 2), density=True, color="#4C72B0", alpha=0.75,
            edgecolor="white", linewidth=0.3, label="null (simulated)")

    xs = [lo + i * (hi - lo) / 200 for i in range(201)]
    ax.plot(xs, [normal_pdf(x, mean, sd) for x in xs], color="#C44E52",
            linewidth=2, label="fitted normal")
    ax.axvline(n_obs, color="black", linestyle="--", linewidth=2,
               label=f"observed = {n_obs}")
    ax.axvline(mean, color="#C44E52", linestyle=":", linewidth=1.5,
               label=f"null mean = {mean:.1f}")

    ax.set_title(f"{comb}\n{label} | n_obs={n_obs}  z={s['z']:.2f}  p={s['p']:.3g}",
                 fontsize=10)
    ax.set_xlabel(f"variants in {s['length']} bp of random CDS")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150)
print(f"wrote {OUT_PNG}")
