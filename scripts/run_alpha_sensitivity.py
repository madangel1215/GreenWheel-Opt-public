#!/usr/bin/env python3
"""Fairness-tolerance (alpha) sensitivity analysis.

How the fairness tolerance
alpha in the proportional-fairness constraint affects optimization quality.
Sweeps alpha, running vanilla NSGA-II and SA-NSGA-II at a fixed scale, and
reports hypervolume (per (alpha, seed) common reference point across the two
methods) plus solution counts.

Launch unbuffered for live progress:
    python -u scripts/run_alpha_sensitivity.py \
        --model checkpoints/surrogate_phase7_synthetic/best.pt \
        --scale large --alphas 0.05,0.10,0.15,0.20,0.25 --n-seeds 10 \
        --device cuda --output results/alpha_sensitivity
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.greenwheel.config import WheelingConfig
from src.greenwheel.data.synthetic import generate_instance
from src.greenwheel.optimization.metrics import compute_hypervolume
from src.greenwheel.optimization.nsga2 import run_nsga2
from src.greenwheel.optimization.samoea import SAMOEAConfig, run_samoea
from src.greenwheel.surrogate.inference import BatchSurrogate
from src.greenwheel.surrogate.training import load_trained_model

SIZE = {"small": (5, 10), "medium": (10, 20), "large": (20, 50), "xlarge": (50, 100)}

def _ref_point(*fronts):
    valid = [np.asarray(f) for f in fronts if len(f) > 0]
    if not valid:
        return np.ones(3) * 1e10
    return np.max(np.vstack(valid), axis=0) * 1.1

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--scale", default="large")
    p.add_argument("--alphas", default="0.05,0.10,0.15,0.20,0.25")
    p.add_argument("--n-seeds", dest="n_seeds", type=int, default=10)
    p.add_argument("--n-gen", dest="n_gen", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="results/alpha_sensitivity")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.n_seeds, args.n_gen, args.scale = 2, 30, "small"
    alphas = [float(a) for a in args.alphas.split(",")]
    m, n = SIZE[args.scale]
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    model, norm = load_trained_model(args.model, device=args.device)
    surrogate = BatchSurrogate(model, norm, device=args.device)

    results = {}
    for alpha in alphas:
        results[alpha] = {"NSGA-II": [], "SA-NSGA-II": []}
        print(f"\n=== alpha={alpha} ({args.scale} {m}x{n}, {args.n_seeds} seeds) ===", flush=True)
        for i in range(args.n_seeds):
            seed = 90000 + i * 100
            cfg = WheelingConfig()
            cfg.n_generators, cfg.n_consumers = m, n
            cfg.n_generations, cfg.population_size = args.n_gen, 100
            cfg.fairness_tolerance = alpha
            inst = generate_instance(cfg, seed=seed)

            t0 = time.time()
            van = run_nsga2(inst, cfg, seed=seed, verbose=False)
            surrogate.set_instance(inst)
            sa = run_samoea(inst, surrogate,
                            SAMOEAConfig(base_algorithm="nsga2", correction_fraction=0.5,
                                         n_generations=args.n_gen),
                            seed=seed, verbose=False)
            ref = _ref_point(van.pareto_front, sa.pareto_front)
            hv_v = compute_hypervolume(van.pareto_front, ref)
            hv_s = compute_hypervolume(sa.pareto_front, ref)
            results[alpha]["NSGA-II"].append({"seed": seed, "hv": hv_v,
                                              "n_sol": len(van.pareto_solutions)})
            results[alpha]["SA-NSGA-II"].append({"seed": seed, "hv": hv_s,
                                                "n_sol": len(sa.pareto_solutions),
                                                "eval_savings": sa.eval_savings})
            print(f"  seed {i+1}/{args.n_seeds}: vanilla HV={hv_v:.3e}, "
                  f"SA HV={hv_s:.3e} ({time.time()-t0:.0f}s)", flush=True)

        for method in ("NSGA-II", "SA-NSGA-II"):
            hvs = [r["hv"] for r in results[alpha][method]]
            print(f"  [{method}] median HV={np.median(hvs):.3e}", flush=True)

    json.dump(results, open(out / "results.json", "w"), indent=2, default=str)
    print(f"\nSaved -> {out}/results.json", flush=True)

if __name__ == "__main__":
    main()
