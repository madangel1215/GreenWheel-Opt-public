#!/usr/bin/env python3
"""E2f: is the real-scenario retailer problem multi-objective, per archetype and rho?

    python scripts/run_e2f_objectives.py --arch L --month 7 --seed 1 --rho 0.1

Search: 2,000 random W, one single-objective GA per indicator and NSGA-II per pair; a
pair conflicts when the front has >= 10 points and both range gaps are >= 5%
(judge_e2f.py aggregates the cases).
--quick shrinks every budget for a smoke test and writes nothing to results/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.run_e2_conflict import RowCap, random_W  # noqa: E402
from src.greenwheel.data import retailer2021 as rt  # noqa: E402

PAIRS = ("B", "C")

def indicators(r):
    p = r["profit"]
    worst = np.sort(p)[: max(1, int(np.ceil(0.2 * len(p))))]
    return dict(profit=p.mean(), worst=r["worst"].mean(), cvar=worst.mean())

def objectives(pair, ind):
    # Minimisation form. B: profit vs worst contract fulfilment; C: profit vs CVaR20.
    if pair == "B":
        return -ind["profit"], -ind["worst"]
    return -ind["profit"], -ind["cvar"]

class Prob(ElementwiseProblem):
    def __init__(self, mp, pair):
        self.mp, self.pair, self.m, self.n = mp, pair, mp.m, mp.n
        super().__init__(n_var=mp.m * mp.n, n_obj=2, xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *a, **k):
        out["F"] = np.array(objectives(self.pair, indicators(self.mp.evaluate(x.reshape(self.m, self.n)))))

class Single(ElementwiseProblem):
    def __init__(self, mp, key):
        self.mp, self.key, self.m, self.n = mp, key, mp.m, mp.n
        super().__init__(n_var=mp.m * mp.n, n_obj=1, xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *a, **k):
        out["F"] = [-indicators(self.mp.evaluate(x.reshape(self.m, self.n)))[self.key]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=rt.ARCHETYPES, required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    n_rand, ga_gen, nsga_gen = (50, 3, 3) if a.quick else (2000, 200, 300)
    tag = f"{a.arch} rho {a.rho} 2021-{a.month:02d} seed {a.seed}"
    t0 = time.time()
    ret = rt.Retailer(a.arch, a.rho, a.seed)
    mp = rt.RetailerMonth(ret, a.month)
    print(f"{tag}: stations={mp.m} customers={mp.n} scenarios={len(mp.scen)}", flush=True)
    rng = np.random.default_rng(a.seed)
    pool = [random_W(rng, mp.m, mp.n) for _ in range(n_rand)]
    for key in ("profit", "worst", "cvar"):
        res = minimize(Single(mp, key), GA(pop_size=100, repair=RowCap()), ("n_gen", ga_gen), seed=a.seed, verbose=False)
        pool += [x.reshape(mp.m, mp.n) for x in res.pop.get("X")]
        print(f"{tag} single {key} done ({time.time() - t0:.0f}s)", flush=True)
    evals = [indicators(mp.evaluate(W)) for W in pool]
    out = dict(arch=a.arch, month=a.month, seed=a.seed, rho=a.rho, m=mp.m, n=mp.n, scenarios=len(mp.scen),
               budget=dict(random=n_rand, ga_gen=ga_gen, nsga_gen=nsga_gen), pairs={})
    for pair in PAIRS:
        res = minimize(Prob(mp, pair), NSGA2(pop_size=100, repair=RowCap()), ("n_gen", nsga_gen), seed=a.seed, verbose=False)
        F = np.vstack([np.array([objectives(pair, e) for e in evals]), np.atleast_2d(res.pop.get("F"))])
        front = np.unique(F[NonDominatedSorting().do(F, only_non_dominated_front=True)], axis=0)
        b1, b2 = front[front[:, 0].argmin()], front[front[:, 1].argmin()]
        span = F.max(axis=0) - F.min(axis=0)
        g1, g2 = abs(b2[0] - b1[0]) / max(span[0], 1e-12), abs(b1[1] - b2[1]) / max(span[1], 1e-12)
        ok = len(front) >= 10 and g1 >= 0.05 and g2 >= 0.05
        out["pairs"][pair] = dict(front=len(front), gap_range_obj1=g1, gap_range_obj2=g2, conflict=bool(ok),
                                  extremes=[b1.tolist(), b2.tolist()], span=span.tolist(), front_points=front.tolist())
        print(f"{tag} pair {pair}: front={len(front)} range-gaps {g1:.1%} / {g2:.1%} "
              f"-> {'CONFLICT' if ok else 'no conflict'} ({time.time() - t0:.0f}s)", flush=True)
    if a.quick:
        return
    path = Path(f"results/retailer/e2f/{a.arch}_2021{a.month:02d}_s{a.seed}_rho{a.rho}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(path, "w"), indent=2)

if __name__ == "__main__":
    main()
