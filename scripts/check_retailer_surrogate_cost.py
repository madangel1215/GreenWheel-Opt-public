#!/usr/bin/env python3
"""Prerequisite for surrogate screening: is exact multi-scenario evaluation dearer than 2 x surrogate inference?

    OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/check_retailer_surrogate_cost.py

Both sides single-threaded on CPU, at each archetype's real size (2021-07, rho 0.1, seed 1).
Exact: RetailerMonth.evaluate plus the indicators, per candidate. Surrogate: the deployed
path per candidate in a batch of 100 (graph build, feature normalisation, batched GNN
forward); weights do not change the cost, so an untrained model of the trained shape is
timed. Run with the machine otherwise idle: concurrent jobs distort both sides.
Pass when exact > 2 x surrogate (with retention 0.5, SA spends c_sur + 0.5 c_exact per
offspring against c_exact for NSGA-II). Also gives k = round(S c_sur / c_exact) and
c_low(k), the low-fidelity screen (sa_low in run_e4_backtest.py), and writes the unit
costs to --out for the cost accounting of the backtest.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.run_e2_conflict import random_W  # noqa: E402
from src.greenwheel.data import retailer2021 as rt  # noqa: E402
from src.greenwheel.surrogate import retailer_graph as rg  # noqa: E402
from src.greenwheel.surrogate.model import WheelingGNN  # noqa: E402
from src.greenwheel.surrogate.training import compute_norm_stats, normalize_features  # noqa: E402

BATCH = 100

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-readout", type=int, nargs="*", default=[],
                    help="Time the shape with the per-buyer soft-min head")
    a = ap.parse_args()
    costs = {}
    torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    model = WheelingGNN(edge_dim=6, min_outputs=tuple(a.min_readout)).eval()
    for arch in rt.ARCHETYPES:
        mp = rt.RetailerMonth(rt.Retailer(arch, 0.1, 1), 7)
        inst = rg.mean_instance(mp)
        Ws = [random_W(rng, mp.m, mp.n) for _ in range(BATCH)]
        norm = compute_norm_stats([rg.to_graph(inst, W, np.zeros(3)) for W in Ws[:10]])
        rg.indicators(mp.evaluate(Ws[0]))
        t0 = time.perf_counter()
        for W in Ws[:20]:
            rg.indicators(mp.evaluate(W))
        c_exact = (time.perf_counter() - t0) / 20
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            graphs = [rg.to_graph(inst, np.round(W, 4)) for W in Ws]
            t_build = time.perf_counter() - t0
            normalize_features(graphs, norm)
            b = Batch.from_data_list(graphs)
            with torch.no_grad():
                WheelingGNN.from_heterodata(b, model)
            ts.append((time.perf_counter() - t0, t_build))
        total, build = min(ts)
        c_sur = total / BATCH
        k = max(1, round(len(mp.scen) * c_sur / c_exact))
        idx = np.random.default_rng(1).choice(len(mp.scen), k, replace=False)
        rg.indicators(mp.evaluate_subset(Ws[0], idx))
        t0 = time.perf_counter()
        for W in Ws[:20]:
            rg.indicators(mp.evaluate_subset(W, idx))
        c_low = (time.perf_counter() - t0) / 20
        costs[arch] = dict(c_exact=c_exact, c_sur=c_sur, k=k, c_low=c_low, scenarios=len(mp.scen))
        print(f"{arch}: m={mp.m} n={mp.n} scenarios={len(mp.scen)}  exact {c_exact * 1000:.1f} ms  "
              f"surrogate {c_sur * 1000:.1f} ms (graph build {build / BATCH * 1000:.1f})  "
              f"ratio {c_exact / c_sur:.2f} -> {'PASS' if c_exact > 2 * c_sur else 'FAIL'}  "
              f"| low-fidelity k={k} {c_low * 1000:.1f} ms", flush=True)
    if a.out:
        json.dump(costs, open(a.out, "w"), indent=2)

if __name__ == "__main__":
    main()
