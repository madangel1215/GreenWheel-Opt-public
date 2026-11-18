#!/usr/bin/env python3
"""Equal-cost reading of the anytime runs on the Table 5 instances.

For each surrogate-assisted variant and its base algorithm, the hypervolume of the
surrogate-assisted run after 200 generations is compared with the base algorithm's at
(i) the same generation count (Table 5's comparison), (ii) the same number of exact
evaluations, (iii) the same wall-clock time, and (iv) 300 generations. The base
algorithm's value at a cost c is that of its last generation whose cumulative cost does
not exceed c. One reference point per instance, built as in Table 5.

    python scripts/analyze_anytime.py --input results/anytime_table5
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

PAIRS = [("NSGA-II", "SA-NSGA-II"), ("MOEA/D", "SA-MOEA/D"), ("NSGA-III", "SA-NSGA-III")]


def at(run, key, limit):
    h = [x for x in run["hv_history"] if x[key] <= limit]
    return h[-1]["hv"] if h else 0.0


def cmp(a, b):
    a, b = np.asarray(a), np.asarray(b)
    try:
        p = float(wilcoxon(a, b).pvalue)
    except ValueError:
        p = float("nan")
    return {"sa_mean": float(a.mean()), "base_mean": float(b.mean()),
            "ratio_of_means": float(a.mean() / b.mean()) if b.mean() > 0 else float("inf"),
            "sa_wins": int((a > b).sum()), "n": len(a), "wilcoxon_p": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/anytime_table5")
    a = ap.parse_args()
    d = json.load(open(Path(a.input) / "results.json"))
    n_gen = d["_meta"]["n_gen_sa"]
    out = {}
    for scale in [k for k in d if k != "_meta"]:
        out[scale] = {}
        for base, sa in PAIRS:
            if base not in d[scale]:
                continue
            B, S = d[scale][base], d[scale][sa]
            sa_hv = [s["hv_at_n_gen"] for s in S]
            sa_ev = [s["n_true_evals_total"] for s in S]
            sa_t = [s["elapsed_s_total"] for s in S]
            r = {"equal_generations": cmp(sa_hv, [at(b, "generation", n_gen) for b in B]),
                 "equal_evaluations": cmp(sa_hv, [at(b, "n_true_evals", e) for b, e in zip(B, sa_ev)]),
                 "equal_wall_clock": cmp(sa_hv, [at(b, "elapsed_s", t) for b, t in zip(B, sa_t)]),
                 "base_at_300_generations": cmp(sa_hv, [b["hv_history"][-1]["hv"] for b in B]),
                 "median_sa_evaluations": float(np.median(sa_ev)),
                 "median_base_evaluations_at_n_gen": float(np.median([at_ev(b, n_gen) for b in B])),
                 "median_sa_seconds": float(np.median(sa_t)),
                 "median_base_seconds_at_n_gen": float(np.median([at_t(b, n_gen) for b in B])),
                 "base_runs_covering_sa_time": int(sum(b["hv_history"][-1]["elapsed_s"] >= t for b, t in zip(B, sa_t)))}
            out[scale][sa] = r
            print(f"== {scale} {sa} vs {base}: SA evals {r['median_sa_evaluations']:.0f} vs {r['median_base_evaluations_at_n_gen']:.0f}, "
                  f"seconds {r['median_sa_seconds']:.0f} vs {r['median_base_seconds_at_n_gen']:.0f}")
            for k in ("equal_generations", "equal_evaluations", "equal_wall_clock", "base_at_300_generations"):
                c = r[k]
                print(f"   {k:24s} ratio {c['ratio_of_means']:6.2f}x  SA wins {c['sa_wins']}/{c['n']}  p={c['wilcoxon_p']:.4f}")
    json.dump(out, open(Path(a.input) / "equal_cost.json", "w"), indent=2)


def at_ev(run, gen):
    return [x for x in run["hv_history"] if x["generation"] <= gen][-1]["n_true_evals"]


def at_t(run, gen):
    return [x for x in run["hv_history"] if x["generation"] <= gen][-1]["elapsed_s"]


if __name__ == "__main__":
    main()
