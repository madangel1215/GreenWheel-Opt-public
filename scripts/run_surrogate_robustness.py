#!/usr/bin/env python3
"""Surrogate overfitting / robustness study.

feature for : (a) variance of test accuracy across independent
retrainings with different random seeds, and (b) a training-data-size ablation.
Both reuse the exact GNN training and evaluation pipeline.

Launch unbuffered:
    python -u scripts/run_surrogate_robustness.py \
        --data data/surrogate/phase7_synthetic \
        --retrains 5 --fractions 0.25,0.5,0.75,1.0 --epochs 300 \
        --device cuda --output results/surrogate_robustness
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.greenwheel.surrogate.training import TrainConfig, train_surrogate
from src.greenwheel.surrogate.validation import evaluate_model

OBJS = ["profit", "surplus", "shortfall"]

def _load(d, name):
    with open(Path(d) / f"{name}.pkl", "rb") as f:
        return pickle.load(f)

def _clone(data):
    return [d.clone() for d in data]

def _train_eval(train, val, test, epochs, device, seed, ckpt):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, norm, _ = train_surrogate(
        _clone(train), _clone(val), TrainConfig(epochs=epochs),
        device=device, checkpoint_dir=ckpt, verbose=False)
    m = evaluate_model(model, test, norm, device=device)
    return {o: {"spearman": m[o]["spearman"], "r2": m[o]["r2"]} for o in OBJS}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--retrains", type=int, default=5)
    p.add_argument("--fractions", default="0.25,0.5,0.75,1.0")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="results/surrogate_robustness")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.retrains, args.epochs, args.fractions = 2, 20, "0.5,1.0"

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    train = _load(args.data, "train"); val = _load(args.data, "val"); test = _load(args.data, "test")
    print(f"Loaded train={len(train)} val={len(val)} test={len(test)}", flush=True)

    report = {"retrain_variance": [], "training_size": []}

    # (a) Variance across independent retrainings (full data, different seeds)
    print(f"\n=== retrain variance ({args.retrains} runs, full data) ===", flush=True)
    for s in range(args.retrains):
        t0 = time.time()
        r = _train_eval(train, val, test, args.epochs, args.device, seed=s,
                        ckpt=str(out / f"_tmp_retrain{s}"))
        report["retrain_variance"].append({"seed": s, **r})
        print(f"  run {s+1}/{args.retrains}: profit rho={r['profit']['spearman']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    for o in OBJS:
        vals = [run[o]["spearman"] for run in report["retrain_variance"]]
        print(f"  [{o}] Spearman mean={np.mean(vals):.4f} std={np.std(vals):.4f}", flush=True)

    # (b) Training-size ablation (fixed seed, subsample train samples)
    print(f"\n=== training-size ablation ===", flush=True)
    rng = np.random.default_rng(0)
    fractions = [float(x) for x in args.fractions.split(",")]
    for frac in fractions:
        k = max(1, int(len(train) * frac))
        idx = rng.choice(len(train), k, replace=False)
        sub = [train[i] for i in idx]
        t0 = time.time()
        r = _train_eval(sub, val, test, args.epochs, args.device, seed=0,
                        ckpt=str(out / f"_tmp_frac{frac}"))
        report["training_size"].append({"fraction": frac, "n_train": k, **r})
        print(f"  frac={frac} (n={k}): profit rho={r['profit']['spearman']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    json.dump(report, open(out / "results.json", "w"), indent=2, default=str)
    print(f"\nSaved -> {out}/results.json", flush=True)

if __name__ == "__main__":
    main()
