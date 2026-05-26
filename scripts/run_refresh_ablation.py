#!/usr/bin/env python3
"""Full-evaluation interval (surrogate-refresh) ablation.

Empirical evidence on the claim that
periodic full evaluations limit surrogate drift. Sweeps the full-evaluation
interval g_e for SA-NSGA-II (including a no-refresh setting), reporting
hypervolume and evaluation savings.

Launch unbuffered:
    python -u scripts/run_refresh_ablation.py \
        --model checkpoints/surrogate_phase7_synthetic/best.pt \
        --scale large --intervals 10,20,40,999 --n-seeds 10 \
        --device cuda --output results/refresh_ablation
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
    return np.max(np.vstack(valid), axis=0) * 1.1 if valid else np.ones(3) * 1e10

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--scale", default="large")
    p.add_argument("--intervals", default="10,20,40,999",
                   help="full_eval_interval values; 999 ~= no refresh")
    p.add_argument("--n-seeds", dest="n_seeds", type=int, default=10)
    p.add_argument("--n-gen", dest="n_gen", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="results/refresh_ablation")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.n_seeds, args.n_gen, args.scale = 2, 30, "small"
        args.intervals = "10,999"
    intervals = [int(x) for x in args.intervals.split(",")]
    m, n = SIZE[args.scale]
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    model, norm = load_trained_model(args.model, device=args.device)
    surrogate = BatchSurrogate(model, norm, device=args.device)

    results = {}
    for ge in intervals:
        label = "no-refresh" if ge > args.n_gen else f"every-{ge}"
        results[label] = []
        print(f"\n=== full_eval_interval={ge} ({label}) ===", flush=True)
        for i in range(args.n_seeds):
            seed = 91000 + i * 100
            cfg = WheelingConfig()
            cfg.n_generators, cfg.n_consumers = m, n
            cfg.n_generations, cfg.population_size = args.n_gen, 100
            inst = generate_instance(cfg, seed=seed)
            # vanilla reference for a common ref point
            van = run_nsga2(inst, cfg, seed=seed, verbose=False)
            surrogate.set_instance(inst)
            t0 = time.time()
            sa = run_samoea(inst, surrogate,
                            SAMOEAConfig(base_algorithm="nsga2", correction_fraction=0.5,
                                         n_generations=args.n_gen, full_eval_interval=ge),
                            seed=seed, verbose=False)
            ref = _ref_point(van.pareto_front, sa.pareto_front)
            hv = compute_hypervolume(sa.pareto_front, ref)
            results[label].append({"seed": seed, "full_eval_interval": ge, "hv": hv,
                                    "eval_savings": sa.eval_savings})
            print(f"  seed {i+1}/{args.n_seeds}: HV={hv:.3e}, "
                  f"savings={sa.eval_savings*100:.0f}% ({time.time()-t0:.0f}s)", flush=True)
        hvs = [r["hv"] for r in results[label]]
        sv = [r["eval_savings"] for r in results[label]]
        print(f"  [{label}] median HV={np.median(hvs):.3e}, mean savings={np.mean(sv)*100:.0f}%",
              flush=True)

    json.dump(results, open(out / "results.json", "w"), indent=2, default=str)
    print(f"\nSaved -> {out}/results.json", flush=True)

if __name__ == "__main__":
    main()
