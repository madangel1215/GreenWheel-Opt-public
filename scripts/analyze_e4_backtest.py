#!/usr/bin/env python3
"""Backtest analysis: cost to target, HV at budget fractions, realized NT$ effect.

    python scripts/analyze_e4_backtest.py [--dir results/retailer/e4]

Reads the run_e4_backtest.py outputs and writes <dir>/analysis.json plus a printed table.

Analysis choices, fixed before any result existed:
- A method that never reaches the target within B is given cost B + c_exact, so its
  speed-up is NSGA-II's cost to target / (B + c_exact) < 1 and ranks below every method
  that does reach it. Only generations whose cumulative cost is <= B count.
- Holm runs over every test of (a), (b), (d) and (c) in every cell together: the paper
  states one claim across all cells, so the family is all of them.
- "Last month's best" for month t is the pick (run_e4_backtest.pick) of the NSGA-II run of month t-1 (same
  archetype, rho, seed), settled on month t. It exists only where t-1 is a replay month
  (4, 7, 8, 9, 10, 11); it is a reference, not part of any criterion.
- (e): at each budget fraction f, realized profit of SA-GNN's pick minus NSGA-II's, one-sided
  Wilcoxon over (month, seed). Its Holm family is every cell x every f, kept apart from
  (a)-(d) so that adding it does not change criteria (a)-(d). A benefit is
  claimed at f only where Holm p < 0.05 and the median difference is positive.
- References, descriptive only: VSS = realized profit of a pick minus that of the EV pick; the
  hindsight bound UB is the best realized profit among the oracle pick and every pick whose
  realized worst fulfilment is not below the baseline's (NSGA-II on the realized month is not
  guaranteed optimal); captured share = (pick - baseline) / (UB - baseline), skipped when the
  denominator is not positive.
"""
import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from pymoo.indicators.hv import HV
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.reanalyze_statistics import holm  # noqa: E402
from src.greenwheel.data import replay2021 as rp  # noqa: E402
from src.greenwheel.data import retailer2021 as rt  # noqa: E402

METHODS = ("nsga2", "sa_gnn", "sa_random", "sa_low")
EXTRA = ("ev", "oracle")   # descriptive references, in no criterion
FRACTIONS = (0.25, 0.5, 0.75, 1.0)
TARGET, REF, ALPHA, SHARE = 0.99, 1.1, 0.05, 2 / 3

def hv_of(F, best, worst, hv):
    # Maximised indicators mapped to [0, 1] minimisation over the union of final fronts.
    Z = (best - np.asarray(F)) / np.maximum(best - worst, 1e-12)
    return float(hv(Z))

def one_sided(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0 or np.all(x == 0):
        return 1.0
    return float(wilcoxon(x, alternative="greater").pvalue)

def instance(runs, budget_eps=1e-9):
    """HV trajectories and cost to target for one (cell, month, seed)."""
    finals = np.vstack([np.asarray(runs[k]["history"][-1]["F"]) for k in METHODS])
    best, worst = finals.max(axis=0), finals.min(axis=0)
    hv = HV(ref_point=np.full(finals.shape[1], REF))
    B = runs["nsga2"]["budget"]
    traj = {k: [(h["cost"], hv_of(h["F"], best, worst, hv)) for h in runs[k]["history"]] for k in METHODS}
    target = TARGET * traj["nsga2"][-1][1]
    out = {}
    for k in METHODS:
        within = [(c, v) for c, v in traj[k] if c <= B * (1 + budget_eps)]
        hit = next((c for c, v in within if v >= target), None)
        at = {f: max((v for c, v in within if c <= f * B * (1 + budget_eps)), default=0.0) for f in FRACTIONS}
        out[k] = dict(cost=hit if hit is not None else B + runs[k]["unit"]["c_exact"], reached=hit is not None,
                      hv=at)
    for k in METHODS:
        out[k]["speedup"] = out["nsga2"]["cost"] / out[k]["cost"]
    return out

def last_month(e4, arch, rho, month, seed, cache):
    prev = month - 1
    f = e4 / f"{arch}_rho{rho}_m{prev}_s{seed}_nsga2.pkl"
    if prev not in rp.MONTHS or not f.exists():
        return None
    key = (arch, rho, seed, month)
    if key not in cache:
        mp = rt.RetailerMonth(rt.Retailer(arch, rho, seed), month)
        r = mp.evaluate_real(pickle.load(open(f, "rb"))["W_pick"])
        cache[key] = dict(profit=float(r["profit"]), worst=float(r["worst"]))
    return cache[key]

def references(rows):
    """VSS against the EV pick and the share of the hindsight gain each pick captures."""
    vss, share, skipped = defaultdict(list), defaultdict(list), 0
    for r in rows:
        z, b = r["realized"], r["baseline"]
        if "ev" in z:
            for k in (*METHODS, "last_month"):
                if k in z:
                    vss[k].append(z[k]["profit"] - z["ev"]["profit"])
        if "oracle" not in z:
            continue
        ub = max([z["oracle"]["profit"]] + [x["profit"] for x in z.values() if x["worst"] >= b["worst"]])
        if ub <= b["profit"]:
            skipped += 1
            continue
        for k, x in z.items():
            if k != "oracle":
                share[k].append((x["profit"] - b["profit"]) / (ub - b["profit"]))
    return dict(median_vss_ntd={k: float(np.median(v)) for k, v in vss.items()},
                n_vss={k: len(v) for k, v in vss.items()},
                median_captured_share={k: float(np.median(v)) for k, v in share.items()},
                n_share={k: len(v) for k, v in share.items()}, share_skipped=skipped)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/retailer/e4")
    a = ap.parse_args()
    e4 = Path(a.dir)
    runs = defaultdict(dict)
    for f in sorted(e4.glob("*.json")):
        r = json.load(open(f))
        if "method" not in r:
            continue
        runs[(r["arch"], r["rho"], r["month"], r["seed"])][r["method"]] = r
    incomplete = sorted(f"{k}" for k, v in runs.items() if set(v) != set(METHODS + EXTRA))
    cells = defaultdict(list)
    lm_cache = {}
    for (arch, rho, month, seed), v in sorted(runs.items()):
        if not set(METHODS) <= set(v):
            continue
        res = instance(v)
        base = v["sa_gnn"]["realized"]["baseline"]
        realized = {k: v[k]["realized"]["pick"] for k in METHODS + EXTRA if k in v}
        lm = last_month(e4, arch, rho, month, seed, lm_cache)
        if lm is not None:
            realized["last_month"] = lm
        cells[(arch, rho)].append(dict(month=month, seed=seed, search=res, baseline=base, realized=realized,
                                         realized_at={k: v[k]["realized_at"] for k in METHODS}))

    report, tests, tests_e = {}, [], []
    for (arch, rho), rows in sorted(cells.items()):
        log_s = {k: np.log([r["search"][k]["speedup"] for r in rows]) for k in METHODS}
        d_profit = {k: [r["realized"][k]["profit"] - r["baseline"]["profit"] for r in rows if k in r["realized"]]
                    for k in (*METHODS, *EXTRA, "last_month")}
        d_worst = {k: [r["realized"][k]["worst"] - r["baseline"]["worst"] for r in rows if k in r["realized"]]
                   for k in (*METHODS, *EXTRA, "last_month")}
        base_profit = {k: [r["baseline"]["profit"] for r in rows if k in r["realized"]] for k in d_profit}
        better = [p > 0 and w >= 0 for p, w in zip(d_profit["sa_gnn"], d_worst["sa_gnn"])]
        key = f"{arch}_rho{rho}"
        cell_tests = {
            "a_speedup_vs_nsga2": one_sided(log_s["sa_gnn"]),
            "b_speedup_vs_random": one_sided(log_s["sa_gnn"] - log_s["sa_random"]),
            "d_speedup_vs_low": one_sided(log_s["sa_gnn"] - log_s["sa_low"]),
            "c_profit_vs_baseline": one_sided(d_profit["sa_gnn"]),
        }
        tests += [(key, t, p) for t, p in cell_tests.items()]
        equal = {}
        for f in FRACTIONS:
            rz = [{k: r["realized_at"][k][str(f)] for k in METHODS} for r in rows]
            dp = [z["sa_gnn"]["profit"] - z["nsga2"]["profit"] for z in rz]
            dw = [z["sa_gnn"]["worst"] - z["nsga2"]["worst"] for z in rz]
            p_e = one_sided(dp)
            tests_e.append((key, str(f), p_e))
            equal[str(f)] = dict(
                minutes=float(np.median([z["nsga2"]["minutes"] for z in rz])),
                median_d_profit_ntd=float(np.median(dp)), median_d_worst=float(np.median(dw)), p_raw=p_e,
                median_vs_nsga2_ntd={k: float(np.median([z[k]["profit"] - z["nsga2"]["profit"] for z in rz]))
                                     for k in METHODS if k != "nsga2"})
        report[key] = dict(
            n=len(rows),
            reached={k: int(sum(r["search"][k]["reached"] for r in rows)) for k in METHODS},
            median_speedup={k: float(np.exp(np.median(log_s[k]))) for k in METHODS},
            mean_hv={k: {str(f): float(np.mean([r["search"][k]["hv"][f] for r in rows])) for f in FRACTIONS}
                     for k in METHODS},
            c_share_better=float(np.mean(better)),
            realized={k: dict(n=len(d_profit[k]),
                              median_d_profit_ntd=float(np.median(d_profit[k])) if d_profit[k] else None,
                              median_d_profit_pct=float(np.median(np.array(d_profit[k]) / np.abs(base_profit[k])) * 100)
                              if d_profit[k] else None,
                              median_d_worst=float(np.median(d_worst[k])) if d_worst[k] else None)
                      for k in d_profit},
            p_raw=cell_tests,
            equal_budget=equal,
            references=references(rows),
            rows=rows,
        )
    adj = holm([p for _, _, p in tests]) if tests else []
    for (key, t, _), p in zip(tests, adj):
        report[key].setdefault("p_holm", {})[t] = p
    for key, r in report.items():
        ph = r["p_holm"]
        r["pass"] = dict(a=ph["a_speedup_vs_nsga2"] < ALPHA, b=ph["b_speedup_vs_random"] < ALPHA,
                         d=ph["d_speedup_vs_low"] < ALPHA,
                         c=r["c_share_better"] >= SHARE and ph["c_profit_vs_baseline"] < ALPHA)
        r["pass"]["all"] = all(r["pass"].values())
    adj_e = holm([p for _, _, p in tests_e]) if tests_e else []
    for (key, f, _), p in zip(tests_e, adj_e):
        z = report[key]["equal_budget"][f]
        z["p_holm"] = p
        z["benefit"] = bool(p < ALPHA and z["median_d_profit_ntd"] > 0)
    json.dump(dict(target=TARGET, ref=REF, holm_family=len(tests), holm_family_e=len(tests_e),
                   incomplete=incomplete, cells=report),
              open(e4 / "analysis.json", "w"), indent=2)

    print(f"Holm family: {len(tests)} tests; incomplete instances: {len(incomplete)}")
    for key, r in report.items():
        ms, rz = r["median_speedup"], r["realized"]["sa_gnn"]
        print(f"{key} (n={r['n']}): speed-up median gnn {ms['sa_gnn']:.2f} random {ms['sa_random']:.2f} "
              f"low {ms['sa_low']:.2f}; reached {r['reached']}")
        print(f"  realized vs baseline, SA-GNN: median dProfit NT$ {rz['median_d_profit_ntd']:,.0f} "
              f"({rz['median_d_profit_pct']:+.2f}%), median dWorst {rz['median_d_worst']:+.4f}, "
              f"better in {r['c_share_better']:.0%}")
        for k in ("nsga2", "last_month"):
            z = r["realized"][k]
            if z["n"]:
                print(f"  {k}: n={z['n']} median dProfit NT$ {z['median_d_profit_ntd']:,.0f} "
                      f"({z['median_d_profit_pct']:+.2f}%), dWorst {z['median_d_worst']:+.4f}")
        print("  Holm p: " + ", ".join(f"{t.split('_')[0]} {p:.4f}" for t, p in r["p_holm"].items())
              + " | pass " + " ".join(f"{k}={'Y' if v else 'n'}" for k, v in r["pass"].items()))
        ref = r["references"]
        if ref["n_share"]:
            print("  captured share of hindsight gain (median): " + ", ".join(
                f"{k} {v:.0%}" for k, v in ref["median_captured_share"].items()) + f"; skipped {ref['share_skipped']}")
        if ref["n_vss"]:
            print("  VSS vs EV pick (median NT$): " + ", ".join(f"{k} {v:,.0f}" for k, v in ref["median_vss_ntd"].items()))
        for f, z in r["equal_budget"].items():
            print(f"  (e) {float(f):.0%} of B (~{z['minutes']:.1f} min): SA-GNN - NSGA-II median NT$ "
                  f"{z['median_d_profit_ntd']:,.0f}, dWorst {z['median_d_worst']:+.4f}, Holm p {z['p_holm']:.4f}"
                  f"{' BENEFIT' if z['benefit'] else ''}")

if __name__ == "__main__":
    main()
