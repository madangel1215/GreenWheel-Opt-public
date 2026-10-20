#!/usr/bin/env python3
"""E2: do retailer profit and RE100 shortfall conflict in the real scenario?

Instances: real-case months (Taipower generation + EWELD load), one contract per
consumer, point-13 settlement, 2020-2021 prices. Assumed: green prices
U(4.5, 5.5) per consumer and RE100 targets U(0.3, 1.0), both seeded.
"""
import json
import sys
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.greenwheel.data.realcase import build_month
from src.greenwheel.simulator.settlement import ConsumerContractSettlement, tou_labels

# 2020-2021 (pre-tax): ancillary 0.0250 + dispatch 0.0046 + transmission 0.0101 +
# distribution 0.0158 (2020); avoided cost 2.18 (108年度); FIT as generator opportunity
# cost: solar 4.0, onshore wind 2.32.
FEE, SURPLUS, GEN_COST = 0.0555, 2.18, {"solar": 4.0, "wind": 2.32}
SUMMER = {5, 6, 7, 8, 9, 10}   # 5/16-10/15; the half months are labelled by majority

class Case:
    def __init__(self, year, month, seed, n=100):
        self.inst = build_month(year, month, n, seed)
        rng = np.random.default_rng(seed + 10_000)
        self.price = rng.uniform(4.5, 5.5, n)
        self.target = rng.uniform(0.3, 1.0, n)
        days = self.inst.n_intervals // 96
        import datetime as dt
        start_dow = dt.date(year, month, 1).weekday()
        self.settle = ConsumerContractSettlement(self.inst, tou_labels(days, summer=month in SUMMER,
                                                                       start_weekday=start_dow))
        G = self.settle.G_raw
        self.gen_total = G.sum()
        self.gen_cost = sum(GEN_COST[g.technology] * G[i].sum() for i, g in enumerate(self.inst.generators))
        self.demand = self.settle.D.sum(axis=1)
        self.m, self.n = len(self.inst.generators), n

    def objectives(self, W):
        s = self.settle.settle_quantities(W)
        delivered = s.sum(axis=0)
        total = delivered.sum()
        profit = self.price @ delivered + SURPLUS * (self.gen_total - total) - FEE * total - self.gen_cost
        shortfall = np.maximum(0.0, self.target * self.demand - delivered).sum()
        return -profit, shortfall

class RowCap(Repair):
    # Contract shares of one generator may not exceed 100% (轉供契約第二條).
    def _do(self, problem, X, **kw):
        X = np.clip(X, 0, 1)
        W = X.reshape(len(X), problem.m, problem.n)
        s = W.sum(axis=2, keepdims=True)
        W = np.where(s > 1, W / s, W)
        return W.reshape(len(X), -1)

class Prob(ElementwiseProblem):
    def __init__(self, case):
        self.case, self.m, self.n = case, case.m, case.n
        super().__init__(n_var=case.m * case.n, n_obj=2, xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *a, **k):
        out["F"] = np.array(self.case.objectives(x.reshape(self.m, self.n)))

def random_W(rng, m, n):
    W = rng.random((m, n)) * (rng.random((m, n)) < rng.uniform(0.05, 1.0))
    s = W.sum(axis=1, keepdims=True)
    return np.where(s > 0, W / np.maximum(s, 1e-12) * rng.uniform(0.3, 1.0, (m, 1)), 0.0)

def main():
    results = []
    for (year, month) in [(2020, 7), (2021, 1)]:
        for seed in (1, 2, 3):
            case = Case(year, month, seed)
            rng = np.random.default_rng(seed)
            F_rand = np.array([case.objectives(random_W(rng, case.m, case.n)) for _ in range(2000)])
            res = minimize(Prob(case), NSGA2(pop_size=100, repair=RowCap()), ("n_gen", 100), seed=seed, verbose=False)
            F_ga = np.atleast_2d(res.pop.get("F"))
            F = np.vstack([F_rand, F_ga])
            front = F[NonDominatedSorting().do(F, only_non_dominated_front=True)]
            best_p, best_s = front[front[:, 0].argmin()], front[front[:, 1].argmin()]
            dp = abs(best_s[0] - best_p[0]) / abs(best_p[0])
            ds = abs(best_p[1] - best_s[1]) / max(best_p[1], 1.0)
            rho = spearmanr(-F[:, 0], F[:, 1])[0]
            ok = len(front) >= 10 and dp >= 0.01 and ds >= 0.05
            row = dict(year=year, month=month, seed=seed, m=case.m, n=case.n, front=len(front),
                       profit_gap=dp, shortfall_gap=ds, spearman_profit_shortfall=rho, conflict=bool(ok),
                       best_profit=float(-best_p[0]), best_profit_shortfall=float(best_p[1]),
                       best_shortfall=float(best_s[1]), best_shortfall_profit=float(-best_s[0]))
            results.append(row)
            print(f"{year}-{month:02d} seed {seed}: m={case.m} front={len(front)} "
                  f"profit gap {dp:.2%} shortfall gap {ds:.2%} spearman(profit, shortfall) {rho:+.3f} "
                  f"-> {'CONFLICT' if ok else 'no conflict'}", flush=True)

    out = Path("results/retailer/e2_conflict.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print("conflict in", sum(r["conflict"] for r in results), "of", len(results), "cases")

if __name__ == "__main__":
    main()
