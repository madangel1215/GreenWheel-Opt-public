#!/usr/bin/env python3
"""Augmented statistics for the 7A main comparison - no re-running needed.

feature (re-analysis of existing per-seed HV in
results/phase7/7a_main_comparison/results.json):

  * bootstrap 95% CIs for HV, exact sample counts per comparison,
           effect-size interpretation thresholds, and multiple-comparison
           correction (Holm) over the pairwise family.
  * 3 / - median + IQR and log-transformed HV to characterise the
           very high variance (std sometimes > mean), plus count of
           failed/empty-front runs.

Usage:
    python scripts/reanalyze_statistics.py \
        --results results/phase7/7a_main_comparison/results.json \
        --output results/phase7/7a_reanalysis
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

def bootstrap_ci(x, n_boot=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    means = rng.choice(x, (n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)

def a12(a, b):
    """Vargha-Delaney A12: P(a>b) + 0.5 P(a=b). Magnitude per VD (2000)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = sum((x > y) for x in a for y in b)
    eq = sum((x == y) for x in a for y in b)
    val = (gt + 0.5 * eq) / (len(a) * len(b))
    d = abs(val - 0.5) * 2
    mag = ("negligible" if d < 0.147 else "small" if d < 0.33
           else "medium" if d < 0.474 else "large")
    return float(val), mag

def holm(pvals):
    """Holm-Bonferroni step-down correction. Returns adjusted p-values."""
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    prev = 0.0
    for rank, i in enumerate(idx):
        a = (m - rank) * pvals[i]
        prev = adj[i] = min(1.0, max(a, prev))
    return adj.tolist()

def descriptive(values):
    x = np.asarray([v for v in values if v is not None], float)
    n_fail = sum(v is None for v in values)
    if len(x) == 0:
        return {"n": 0, "n_failed": n_fail}
    lo, hi = bootstrap_ci(x)
    logx = np.log10(x[x > 0])
    return {
        "n": int(len(x)),
        "n_failed": int(n_fail),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "cv": float(x.std(ddof=1) / x.mean()) if len(x) > 1 and x.mean() else None,
        "median": float(np.median(x)),
        "q1": float(np.percentile(x, 25)),
        "q3": float(np.percentile(x, 75)),
        "iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
        "ci95_mean": [lo, hi],
        "log10_mean": float(logx.mean()) if len(logx) else None,
        "log10_std": float(logx.std(ddof=1)) if len(logx) > 1 else None,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/phase7/7a_main_comparison/results.json")
    p.add_argument("--output", default="results/phase7/7a_reanalysis")
    args = p.parse_args()

    data = json.load(open(args.results))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    report = {}
    for scale, variants in data.items():
        hv = {v: [r.get("hv") for r in runs] for v, runs in variants.items()}
        desc = {v: descriptive(vals) for v, vals in hv.items()}

        # Pairwise Wilcoxon over the family, with Holm correction
        pairs, raw_p, eff = [], [], {}
        for a, b in combinations(hv.keys(), 2):
            xa = [r.get("hv") for r in variants[a]]
            xb = [r.get("hv") for r in variants[b]]
            paired = [(u, w) for u, w in zip(xa, xb) if u is not None and w is not None]
            if len(paired) < 5:
                continue
            ua, ub = zip(*paired)
            try:
                stat, pv = wilcoxon(ua, ub)
            except ValueError:
                pv = 1.0
            a12v, mag = a12(ua, ub)
            pairs.append((a, b))
            raw_p.append(float(pv))
            eff[f"{a} vs {b}"] = {"n_pairs": len(paired), "wilcoxon_p": float(pv),
                                  "a12": a12v, "a12_magnitude": mag}
        adj = holm(raw_p) if raw_p else []
        for (a, b), q in zip(pairs, adj):
            eff[f"{a} vs {b}"]["holm_p"] = float(q)
            eff[f"{a} vs {b}"]["significant_holm_0.05"] = bool(q < 0.05)

        report[scale] = {"descriptive": desc, "pairwise": eff}
        print(f"\n=== {scale} ===  (n per variant, median HV, CV, 95% CI of mean)")
        for v, dd in desc.items():
            if dd["n"] == 0:
                print(f"  {v:14s}: all {dd['n_failed']} runs failed")
                continue
            print(f"  {v:14s}: n={dd['n']}(+{dd['n_failed']} fail)  "
                  f"med={dd['median']:.2e}  CV={dd['cv']:.2f}  "
                  f"CI95=[{dd['ci95_mean'][0]:.2e},{dd['ci95_mean'][1]:.2e}]")

    json.dump(report, open(out / "augmented_statistics.json", "w"),
              indent=2, default=str)
    print(f"\nSaved -> {out}/augmented_statistics.json")
    print("Effect-size thresholds (Vargha-Delaney): "
          "negligible<0.56, small<0.64, medium<0.71, large≥0.71 (|A12-0.5|·2 scale)")

if __name__ == "__main__":
    main()
