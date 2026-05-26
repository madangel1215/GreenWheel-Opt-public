#!/usr/bin/env python3
"""Train non-GNN surrogate baselines and score them like the GNN.

Trains classical surrogates (RandomForest, XGBoost) and a stronger
non-graph neural baseline (DeepSets). All baselines
use the SAME phase7 dataset and the SAME metric schema as the GNN so the
numbers drop straight into the surrogate-accuracy comparison table.

Usage:
    python scripts/train_baselines.py \
        --data data/surrogate/phase7_synthetic \
        --output results/baselines \
        --models rf,xgb,deepsets --device cuda

    # fast smoke test (subsample + few epochs)
    python scripts/train_baselines.py --data data/surrogate/phase7_synthetic \
        --output results/baselines_quick --quick
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.greenwheel.surrogate.baselines import (
    train_classical_per_scale,
    train_deepsets,
)

def _load(data_dir: Path, name: str) -> list:
    with open(data_dir / f"{name}.pkl", "rb") as f:
        return pickle.load(f)

def _subsample(data: list, k: int, seed: int = 0) -> list:
    if k >= len(data):
        return data
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data), k, replace=False)
    return [data[i] for i in idx]

def _print_summary(all_metrics: dict):
    print("\n" + "=" * 78)
    print("SURROGATE ACCURACY - aggregate (held-out test set)")
    print("=" * 78)
    print(f"{'model':>10s} | " + " | ".join(
        f"{o+' ρ':>12s}" for o in ["profit", "surplus", "shortfall"]))
    print("-" * 78)
    for name, res in all_metrics.items():
        agg = res["aggregate"]
        row = f"{name:>10s} | " + " | ".join(
            f"{agg[o]['spearman']:>12.3f}" for o in ["profit", "surplus", "shortfall"])
        print(row)
    print("(higher Spearman ρ = better ranking fidelity for pre-screening)")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="results/baselines")
    p.add_argument("--models", default="rf,xgb,deepsets",
                   help="comma list of: rf, xgb, deepsets")
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = _load(data_dir, "train")
    val = _load(data_dir, "val")
    test = _load(data_dir, "test")
    print(f"Loaded train={len(train)} val={len(val)} test={len(test)}")

    if args.quick:
        train = _subsample(train, 1500)
        val = _subsample(val, 400)
        test = _subsample(test, 400)
        args.epochs = 25
        args.n_estimators = 80
        print(f"[quick] subsampled train={len(train)} val={len(val)} test={len(test)}")

    models = [m.strip() for m in args.models.split(",")]
    all_metrics: dict = {}

    for kind in models:
        t0 = time.time()
        print(f"\n>>> training baseline: {kind}")
        if kind in ("rf", "xgb"):
            res = train_classical_per_scale(
                train, test, kind=kind, n_estimators=args.n_estimators)
        elif kind == "deepsets":
            res = train_deepsets(
                train, val, test, epochs=args.epochs, device=args.device)
        else:
            print(f"  !! unknown model '{kind}', skipping")
            continue
        res["train_seconds"] = round(time.time() - t0, 1)
        all_metrics[kind] = res
        with open(out_dir / f"metrics_{kind}.json", "w") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"  done in {res['train_seconds']}s -> {out_dir}/metrics_{kind}.json")

    with open(out_dir / "metrics_all.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    _print_summary(all_metrics)
    print(f"\nSaved -> {out_dir}/metrics_all.json")

if __name__ == "__main__":
    main()
