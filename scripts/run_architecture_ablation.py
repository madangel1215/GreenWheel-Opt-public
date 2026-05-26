#!/usr/bin/env python3
"""Architecture ablation for the WheelingGNN surrogate.

Varies the surrogate's depth, hidden width, attention heads, and
message-passing backbone one factor at a time from the deployed
configuration, reporting held-out test accuracy and parameter count so
the chosen architecture can be justified.

Launch unbuffered:
    python -u scripts/run_architecture_ablation.py \
        --data data/surrogate/phase7_synthetic --epochs 300 --device cuda \
        --output results/architecture_ablation
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

from src.greenwheel.surrogate.model import WheelingGNN
from src.greenwheel.surrogate.training import TrainConfig, train_surrogate
from src.greenwheel.surrogate.validation import evaluate_model

OBJS = ["profit", "surplus", "shortfall"]
BASE = {"hidden_dim": 64, "n_layers": 3, "heads": 4, "conv_type": "gatv2"}

# One factor varied from BASE per row
SWEEP = [
    ("baseline (3L, h64, 4heads, GATv2)", {}),
    ("layers=2", {"n_layers": 2}),
    ("layers=4", {"n_layers": 4}),
    ("hidden=32", {"hidden_dim": 32}),
    ("hidden=128", {"hidden_dim": 128}),
    ("heads=2", {"heads": 2}),
    ("heads=8", {"heads": 8}),
    ("backbone=GAT", {"conv_type": "gat"}),
    ("backbone=Transformer", {"conv_type": "transformer"}),
]

def _load(d, name):
    with open(Path(d) / f"{name}.pkl", "rb") as f:
        return pickle.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="results/architecture_ablation")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    sweep = SWEEP[:3] if args.quick else SWEEP
    epochs = 15 if args.quick else args.epochs

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    train = _load(args.data, "train"); val = _load(args.data, "val"); test = _load(args.data, "test")
    print(f"Loaded train={len(train)} val={len(val)} test={len(test)}", flush=True)

    results = []
    for name, override in sweep:
        kw = {**BASE, **override}
        torch.manual_seed(0); np.random.seed(0)
        n_params = sum(p.numel() for p in WheelingGNN(**kw).parameters())
        t0 = time.time()
        model, norm, _ = train_surrogate(
            [d.clone() for d in train], [d.clone() for d in val],
            TrainConfig(epochs=epochs), model_kwargs=kw,
            device=args.device, checkpoint_dir=str(out / f"_tmp_{name.split()[0]}"),
            verbose=False)
        m = evaluate_model(model, test, norm, device=args.device)
        row = {"config": name, "kwargs": kw, "n_params": int(n_params),
               "train_seconds": round(time.time() - t0, 1),
               **{o: {"spearman": m[o]["spearman"], "r2": m[o]["r2"]} for o in OBJS}}
        results.append(row)
        print(f"  {name:34s} params={n_params:>7d}  "
              f"profit rho={m['profit']['spearman']:.4f}  "
              f"surplus rho={m['surplus']['spearman']:.4f}  "
              f"shortfall rho={m['shortfall']['spearman']:.4f}  "
              f"({row['train_seconds']}s)", flush=True)

    json.dump(results, open(out / "results.json", "w"), indent=2, default=str)
    print(f"\nSaved -> {out}/results.json", flush=True)

if __name__ == "__main__":
    main()
