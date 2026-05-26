#!/usr/bin/env python3
"""Per-generation convergence curves for vanilla vs SA-NSGA-II.

Convergence-stability evidence. Records the
hypervolume trajectory per generation (computed post-hoc with a common
per-seed reference point) for vanilla NSGA-II and SA-NSGA-II, across several
seeds, so the spread across seeds characterises stability.

Launch unbuffered:
    python -u scripts/run_convergence.py \
        --model checkpoints/surrogate_phase7_synthetic/best.pt \
        --scales medium,large --n-seeds 5 --device cuda \
        --output results/convergence
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
    p.add_argument("--scales", default="medium,large")
    p.add_argument("--n-seeds", dest="n_seeds", type=int, default=5)
    p.add_argument("--n-gen", dest="n_gen", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="results/convergence")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.scales, args.n_seeds, args.n_gen = "small", 2, 30

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    model, norm = load_trained_model(args.model, device=args.device)
    surrogate = BatchSurrogate(model, norm, device=args.device)

    report = {}
    for scale in args.scales.split(","):
        m, n = SIZE[scale]
        report[scale] = {"NSGA-II": [], "SA-NSGA-II": []}
        print(f"\n=== {scale} ({m}x{n}) convergence, {args.n_seeds} seeds ===", flush=True)
        for i in range(args.n_seeds):
            seed = 92000 + i * 100
            cfg = WheelingConfig()
            cfg.n_generators, cfg.n_consumers = m, n
            cfg.n_generations, cfg.population_size = args.n_gen, 100
            inst = generate_instance(cfg, seed=seed)
            t0 = time.time()
            van = run_nsga2(inst, cfg, seed=seed, verbose=False, track_convergence=True)
            surrogate.set_instance(inst)
            sa = run_samoea(inst, surrogate,
                            SAMOEAConfig(base_algorithm="nsga2", correction_fraction=0.5,
                                         n_generations=args.n_gen),
                            seed=seed)
            ref = _ref_point(van.pareto_front, sa.pareto_front)
            report[scale]["NSGA-II"].append({"seed": seed,
                                            "hv_history": van.compute_hv_history(ref)})
            report[scale]["SA-NSGA-II"].append({"seed": seed,
                                               "hv_history": sa.compute_hv_history(ref)})
            print(f"  seed {i+1}/{args.n_seeds} done ({time.time()-t0:.0f}s)", flush=True)

    json.dump(report, open(out / "results.json", "w"), indent=2, default=str)
    print(f"\nSaved -> {out}/results.json", flush=True)

if __name__ == "__main__":
    main()
