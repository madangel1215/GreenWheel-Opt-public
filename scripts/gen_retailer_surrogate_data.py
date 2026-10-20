#!/usr/bin/env python3
"""Surrogate training data for the real-scenario retailer.

    python scripts/gen_retailer_surrogate_data.py --arch L --rho 0.1 --seed 1 --year 2020 --month 7 \
        --out data/surrogate/retailer

One run builds one instance (archetype, rho, seed, target month) and writes
<out>/<arch>_rho<rho>_s<seed>_<year><month>.pkl: a list of HeteroData with exact
multi-scenario targets (retailer_graph.indicators). W comes from the same three sources
as the published data (surrogate/dataset.py): random allocations, NSGA-II population
snapshots, and Gaussian perturbations of the final population; NSGA-II here optimises
the three indicators (retailer_graph.indicators). Every search starts from the
contract-share baseline, so NSGA-II here does too, and multiplicative perturbations of the baseline
cover its neighbourhood, where the backtest searches. Training instances use years before 2021 only; 2021 is
the replay and is held out.
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.run_e2_conflict import RowCap, random_W  # noqa: E402
from src.greenwheel.data import retailer2021 as rt  # noqa: E402
from src.greenwheel.surrogate import retailer_graph as rg  # noqa: E402

class Three(Problem):
    def __init__(self, mp):
        self.mp, self.m, self.n = mp, mp.m, mp.n
        super().__init__(n_var=mp.m * mp.n, n_obj=3, xl=0.0, xu=1.0)

    def _evaluate(self, X, out, *a, **k):
        out["F"] = -np.array([rg.indicators(self.mp.evaluate(x.reshape(self.m, self.n))) for x in X])

class Snapshots(Callback):
    def __init__(self, gens):
        super().__init__()
        self.gens, self.items = set(gens), []

    def notify(self, algorithm):
        if algorithm.n_gen in self.gens:
            for ind in algorithm.pop:
                self.items.append((ind.X.copy(), -ind.F.copy()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=rt.ARCHETYPES, required=True)
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-random", type=int, default=600)
    ap.add_argument("--pop", type=int, default=50)
    ap.add_argument("--gens", type=int, default=100)
    ap.add_argument("--snap", type=int, nargs="+", default=[1, 25, 50, 75, 100])
    ap.add_argument("--perturb", type=int, default=4)
    ap.add_argument("--near-base", type=int, default=200)
    a = ap.parse_args()
    t0 = time.time()
    ret = rt.Retailer(a.arch, a.rho, a.seed)
    mp = rt.RetailerMonth(ret, a.month, year=a.year)
    inst = rg.mean_instance(mp)
    rng = np.random.default_rng(1000 * a.seed + a.month)
    samples = []
    for _ in range(a.n_random):
        W = random_W(rng, mp.m, mp.n)
        samples.append((W, rg.indicators(mp.evaluate(W))))
    base = mp.baseline()
    samples.append((base, rg.indicators(mp.evaluate(base))))
    for _ in range(a.near_base):
        # Multiplicative, so a buyer the baseline leaves at 0 stays at 0 and small shares
        # stay small; additive noise of 0.05-0.15 would swamp shares of 1e-4.
        W = base * rng.lognormal(0, rng.uniform(0.05, 0.5), base.shape)
        s = W.sum(axis=1, keepdims=True)
        W = np.where(s > 1, W / s, W)
        samples.append((W, rg.indicators(mp.evaluate(W))))
    cb = Snapshots(a.snap)
    X0 = mp.initial_population(a.pop, np.random.default_rng(a.seed + 900))
    res = minimize(Three(mp), NSGA2(pop_size=a.pop, sampling=X0, repair=RowCap()), ("n_gen", a.gens),
                   seed=a.seed, callback=cb, verbose=False)
    samples += [(x.reshape(mp.m, mp.n), y) for x, y in cb.items]
    for x in res.pop.get("X"):
        for _ in range(a.perturb):
            W = np.clip(x.reshape(mp.m, mp.n) + rng.normal(0, rng.uniform(0.05, 0.15), (mp.m, mp.n)), 0, 1)
            s = W.sum(axis=1, keepdims=True)
            W = np.where(s > 1, W / s, W)
            samples.append((W, rg.indicators(mp.evaluate(W))))
    graphs = [rg.to_graph(inst, np.round(W, 4), y) for W, y in samples]
    for g in graphs:
        g.instance_key = f"{a.arch}_rho{a.rho}_s{a.seed}_{a.year}{a.month:02d}"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{a.arch}_rho{a.rho}_s{a.seed}_{a.year}{a.month:02d}.pkl", "wb") as f:
        pickle.dump(graphs, f)
    print(f"{a.arch} rho {a.rho} s{a.seed} {a.year}-{a.month:02d}: m={mp.m} n={mp.n} scen={len(mp.scen)} "
          f"samples={len(graphs)} ({time.time() - t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
