#!/usr/bin/env python3
"""Local-ranking and FP/FN screening diagnosis.

Tests why a surrogate with high *global* test-set Spearman (>0.90) can fail to
help optimization on semi-real instances. Pre-screening must rank the
*clustered* offspring of a single generation, not a diverse global sample. We
contrast:
  - GLOBAL set: diverse Latin-hypercube allocations (like the held-out test set)
  - LOCAL set:  a converged NSGA-II population (like late-stage offspring)
for both the synthetic and semi-real surrogates, reporting within-set Spearman
and the false-negative rate of top-c_f pre-screening (good candidates wrongly
screened out).

Run locally (all data/checkpoints present):
    python -u scripts/run_screening_diagnosis.py --device cpu \
        --output results/screening_diagnosis
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from src.greenwheel.config import WheelingConfig
from src.greenwheel.data.synthetic import generate_instance, generate_semireal_instance
from src.greenwheel.surrogate.dataset import generate_lhs_samples, generate_nsga2_snapshots
from src.greenwheel.surrogate.inference import BatchSurrogate
from src.greenwheel.surrogate.training import load_trained_model

SIZE = {"medium": (10, 20), "large": (20, 50)}
OBJS = ["profit", "surplus", "shortfall"]

def _topk(F, k):
    """Indices selected by top non-dominated fronts up to k (minimization)."""
    fronts = NonDominatedSorting().do(np.asarray(F), only_non_dominated_front=False)
    sel = []
    for fr in fronts:
        sel.extend(int(i) for i in fr)
        if len(sel) >= k:
            break
    return set(sel[:k])

def _analyze(samples, surrogate):
    """samples: list of (W, profit, surplus, shortfall). Returns metrics dict."""
    Ws = np.stack([s[0].reshape(-1) for s in samples])
    true = np.array([[s[1], s[2], s[3]] for s in samples], dtype=float)  # profit,surplus,shortfall
    pr, su, sh = surrogate.evaluate_batch(Ws)
    pred = np.column_stack([pr, su, sh])
    # within-set Spearman per objective
    rho = {OBJS[i]: float(spearmanr(true[:, i], pred[:, i])[0]) for i in range(3)}
    # selection: minimize [-profit, surplus, shortfall]
    F_true = np.column_stack([-true[:, 0], true[:, 1], true[:, 2]])
    F_pred = np.column_stack([-pred[:, 0], pred[:, 1], pred[:, 2]])
    k = len(samples) // 2  # c_f = 0.5
    sel_true, sel_pred = _topk(F_true, k), _topk(F_pred, k)
    fn = len(sel_true - sel_pred) / max(1, len(sel_true))  # good ones screened out
    return {"n": len(samples), "local_rho": rho,
            "mean_local_rho": float(np.mean(list(rho.values()))),
            "fn_rate": float(fn), "selection_overlap": len(sel_true & sel_pred) / max(1, len(sel_true))}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--scales", default="medium,large")
    p.add_argument("--n-seeds", dest="n_seeds", type=int, default=5)
    p.add_argument("--output", default="results/screening_diagnosis")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.scales, args.n_seeds = "medium", 2

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    domains = {
        "synthetic": ("checkpoints/surrogate_phase7_synthetic/best.pt", generate_instance),
        "semireal": ("checkpoints/surrogate_phase7_semireal/best.pt", generate_semireal_instance),
    }
    report = {}
    for dom, (ckpt, gen_fn) in domains.items():
        model, norm = load_trained_model(ckpt, device=args.device)
        surrogate = BatchSurrogate(model, norm, device=args.device)
        report[dom] = {}
        for scale in args.scales.split(","):
            m, n = SIZE[scale]
            glob_rho, glob_fn, loc_rho, loc_fn = [], [], [], []
            print(f"\n=== {dom} / {scale} ({m}x{n}) ===", flush=True)
            for i in range(args.n_seeds):
                sd = 95000 + i * 100
                cfg = WheelingConfig(); cfg.n_generators, cfg.n_consumers = m, n
                inst = gen_fn(cfg, seed=sd)
                surrogate.set_instance(inst)
                t0 = time.time()
                g = _analyze(generate_lhs_samples(inst, 100, seed=sd), surrogate)
                loc_samples = generate_nsga2_snapshots(inst, pop_size=100, n_generations=200,
                                                       snapshot_gens=[200], seed=sd)
                l = _analyze(loc_samples, surrogate)
                glob_rho.append(g["mean_local_rho"]); glob_fn.append(g["fn_rate"])
                loc_rho.append(l["mean_local_rho"]); loc_fn.append(l["fn_rate"])
                print(f"  seed {i+1}: GLOBAL ρ={g['mean_local_rho']:.3f} FN={g['fn_rate']:.2f} | "
                      f"LOCAL ρ={l['mean_local_rho']:.3f} FN={l['fn_rate']:.2f} ({time.time()-t0:.0f}s)", flush=True)
            report[dom][scale] = {
                "global_mean_rho": float(np.mean(glob_rho)), "global_fn_rate": float(np.mean(glob_fn)),
                "local_mean_rho": float(np.mean(loc_rho)), "local_fn_rate": float(np.mean(loc_fn)),
            }
            r = report[dom][scale]
            print(f"  >>> GLOBAL ρ={r['global_mean_rho']:.3f} FN={r['global_fn_rate']:.2f} | "
                  f"LOCAL ρ={r['local_mean_rho']:.3f} FN={r['local_fn_rate']:.2f}", flush=True)

    json.dump(report, open(out / "results.json", "w"), indent=2, default=str)
    print(f"\nSaved -> {out}/results.json", flush=True)

if __name__ == "__main__":
    main()
