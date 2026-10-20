#!/usr/bin/env python3
"""Acceptance check: within-population ranking of the retailer surrogate on 2021 instances.

    python scripts/diag_retailer_local_rank.py --ckpt checkpoints/retailer --arch L --rho 0.1 \
        --month 7 --seeds 1 2 3 4 5

For each seed: build the 2021 instance (never seen in training), run exact NSGA-II on the
three indicators (pop 100, 100 generations, started from the baseline as in the
backtest), and compare the surrogate's predictions with the exact indicators over the final population by Spearman rho, per
objective. Pass: for every objective, rho > 0 by a one-sided Wilcoxon over the
seeds at p < 0.05, and the mean rho over seeds and objectives is at least 0.5. FN is not
used: on converged populations the first front is too large for it.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from scipy.stats import spearmanr, wilcoxon
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.gen_retailer_surrogate_data import Three  # noqa: E402
from scripts.run_e2_conflict import RowCap  # noqa: E402
from src.greenwheel.data import retailer2021 as rt  # noqa: E402
from src.greenwheel.surrogate import retailer_graph as rg  # noqa: E402
from src.greenwheel.surrogate.model import WheelingGNN  # noqa: E402
from src.greenwheel.surrogate.training import load_trained_model, normalize_features  # noqa: E402

NAMES = ("profit", "worst", "cvar")

def predict(model, norm, inst, Ws):
    graphs = [rg.to_graph(inst, np.round(W, 4)) for W in Ws]
    normalize_features(graphs, norm)
    with torch.no_grad():
        return WheelingGNN.from_heterodata(Batch.from_data_list(graphs), model).numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", choices=rt.ARCHETYPES, required=True)
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--gens", type=int, default=100, help="NSGA-II generations; the acceptance check uses 100")
    a = ap.parse_args()
    model, norm = load_trained_model(Path(a.ckpt) / "best.pt", device="cpu")
    model.eval()
    rhos = []
    for seed in a.seeds:
        mp = rt.RetailerMonth(rt.Retailer(a.arch, a.rho, seed), a.month)
        X0 = mp.initial_population(100, np.random.default_rng(seed + 900))
        res = minimize(Three(mp), NSGA2(pop_size=100, sampling=X0, repair=RowCap()), ("n_gen", a.gens),
                       seed=seed, verbose=False)
        Ws = [x.reshape(mp.m, mp.n) for x in res.pop.get("X")]
        y = np.array([rg.indicators(mp.evaluate(W)) for W in Ws])
        p = predict(model, norm, rg.mean_instance(mp), Ws)
        r = [spearmanr(y[:, k], p[:, k])[0] if np.ptp(y[:, k]) > 0 else np.nan for k in range(3)]
        rhos.append(r)
        print(f"{a.arch} rho {a.rho} 2021-{a.month:02d} seed {seed}: local Spearman "
              + "  ".join(f"{nm} {v:.3f}" for nm, v in zip(NAMES, r)), flush=True)
    R = np.array(rhos)
    ps = [wilcoxon(R[:, k], alternative="greater").pvalue if np.all(np.isfinite(R[:, k])) else np.nan
          for k in range(3)]
    ok = all(p < 0.05 for p in ps) and np.nanmean(R) >= 0.5
    print(f"mean {np.nanmean(R, axis=0).round(3).tolist()} overall {np.nanmean(R):.3f}  one-sided Wilcoxon p "
          f"{[round(p, 4) for p in ps]} -> {'PASS' if ok else 'FAIL'}")
    out = Path(a.ckpt) / f"local_rank_{a.arch}_rho{a.rho}_m{a.month}.json"
    json.dump(dict(rho=R.tolist(), p=ps, passed=bool(ok)), open(out, "w"), indent=2)

if __name__ == "__main__":
    main()
