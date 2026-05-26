#!/usr/bin/env python3
"""Profile GNN surrogate inference cost and GPU memory per scale.

Empirical inference time
per batch and per allocation, peak GPU memory, and inference-vs-simulation cost
ratio, across problem scales. Asymptotic complexity is derived analytically in
the paper; this script supplies the measured constants.

Usage:
    python scripts/profile_surrogate.py \
        --model checkpoints/surrogate_phase7_synthetic/best.pt \
        --device cuda --batch 100 --repeats 30 --output results/complexity.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.greenwheel.config import WheelingConfig
from src.greenwheel.data.synthetic import generate_instance
from src.greenwheel.optimization.samoea import _evaluate_true_batch
from src.greenwheel.surrogate.inference import BatchSurrogate
from src.greenwheel.surrogate.training import load_trained_model

SIZE_CONFIGS = {
    "small": (5, 10), "medium": (10, 20), "large": (20, 50), "xlarge": (50, 100),
}

def make_instance(m, n, seed=7):
    cfg = WheelingConfig()
    cfg.n_generators = m
    cfg.n_consumers = n
    return generate_instance(cfg, seed=seed), cfg

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=100)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--output", default="results/complexity.json")
    args = p.parse_args()

    model, norm = load_trained_model(args.model, device=args.device)
    use_cuda = args.device == "cuda" and torch.cuda.is_available()

    rows = {}
    for scale, (m, n) in SIZE_CONFIGS.items():
        instance, _ = make_instance(m, n)
        surrogate = BatchSurrogate(model, norm, device=args.device)
        surrogate.set_instance(instance)

        rng = np.random.default_rng(0)
        W = rng.random((args.batch, m * n)).astype(np.float32)

        # warm-up (JIT/cuda kernels, caching)
        surrogate.evaluate_batch(W)
        if use_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            surrogate.evaluate_batch(W)
            if use_cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        times = np.array(times)
        per_batch_ms = float(times.mean() * 1e3)
        per_alloc_ms = per_batch_ms / args.batch
        peak_mem_mb = (torch.cuda.max_memory_allocated() / 1e6) if use_cuda else None

        # True simulator cost on the SAME machine (CPU), for a hardware-consistent
        # inference-vs-simulation ratio. Smaller batch since it is slow.
        sim_batch = min(20, args.batch)
        Wsim = rng.random((sim_batch, m * n)).astype(np.float32)
        _evaluate_true_batch(Wsim.copy(), instance)  # warm-up
        sim_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            _evaluate_true_batch(Wsim.copy(), instance)
            sim_times.append(time.perf_counter() - t0)
        sim_per_alloc_ms = float(np.mean(sim_times) * 1e3) / sim_batch

        rows[scale] = {
            "m": m, "n": n, "edges_mn": m * n, "batch": args.batch,
            "infer_per_batch_ms": round(per_batch_ms, 3),
            "infer_per_alloc_ms": round(per_alloc_ms, 4),
            "infer_per_batch_std_ms": round(float(times.std() * 1e3), 3),
            "peak_gpu_mem_mb": round(peak_mem_mb, 1) if peak_mem_mb else None,
            "sim_per_alloc_ms": round(sim_per_alloc_ms, 4),
            "speedup_sim_over_infer": round(sim_per_alloc_ms / per_alloc_ms, 2),
        }
        print(f"{scale:8s} m*n={m*n:5d}  infer={per_alloc_ms:.3f}ms/alloc  "
              f"sim={sim_per_alloc_ms:.3f}ms/alloc  "
              f"sim/infer={rows[scale]['speedup_sim_over_infer']}x  "
              f"peak_mem={rows[scale]['peak_gpu_mem_mb']}MB")

    n_params = sum(p.numel() for p in model.parameters())
    out = {
        "n_params": int(n_params),
        "device": str(next(model.parameters()).device),
        "arch": {"hidden_dim": 64, "n_layers": 3, "heads": 4},
        "scales": rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"\nparams={n_params}  saved -> {args.output}")

if __name__ == "__main__":
    main()
