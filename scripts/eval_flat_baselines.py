#!/usr/bin/env python3
"""Held-out accuracy of the flat baselines, and the effect of their input.

Two questions.

1. What is the MLP's accuracy on the held-out test set? One MLP is trained per scale
   (small, medium, large) with the published pipeline, a fixed seed, and scored on the
   test instances of its scale; predictions are then pooled over scales exactly as for
   the random forest and XGBoost (baselines.train_classical_per_scale).

2. Does the flat baselines' weakness come from the size of their input? Each flat model
   is also trained on a reduced input, the allocation matrix plus the six instance-level
   features (mn + 6 values), in place of the published input (random forest and XGBoost:
   node, instance and all three edge features, 8m + 11n + 6 + 3mn; MLP: node and
   instance features and the allocation matrix, 8m + 11n + 6 + mn).

The random forest and XGBoost on the published input are rerun as a check: they must
reproduce results/baselines/metrics_{rf,xgb}.json.

    python -u scripts/eval_flat_baselines.py --output results/flat_baselines
"""

import argparse
import copy
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.greenwheel.surrogate.baselines import (compute_metrics, group_by_scale,
                                                train_classical_per_scale)
from src.greenwheel.surrogate.mlp_training import (_heterodata_to_flat, load_trained_mlp,
                                                   train_mlp_surrogate)
from src.greenwheel.surrogate.training import TrainConfig, normalize_features


def reduced_np(d) -> np.ndarray:
    """Allocation matrix and instance-level features only (trees; raw values)."""
    g = d.global_feat.reshape(-1).numpy()
    w = d["gen", "supplies", "con"].edge_attr[:, 0].numpy()
    return np.concatenate([g, w]).astype(np.float32)


def reduced_torch(data_list, norm):
    """The same reduced input for the MLP, on features normalized as in training."""
    X = [torch.cat([d.global_feat.reshape(-1), d["gen", "supplies", "con"].edge_attr[:, 0]])
         for d in data_list]
    return torch.stack(X), torch.stack([d.y for d in data_list])


def mlp_per_scale(train, val, test, cfg, flatten_fn, seed, ckpt_root):
    tr_g, va_g, te_g = group_by_scale(train), group_by_scale(val), group_by_scale(test)
    per_scale, pooled_p, pooled_y, info = {}, [], [], {}
    for scale in sorted(tr_g):
        if scale not in te_g or scale not in va_g:
            continue
        tr, va, te = (copy.deepcopy(g[scale]) for g in (tr_g, va_g, te_g))
        y_raw = np.stack([d.y.numpy() for d in te]).astype(np.float32)
        torch.manual_seed(seed); np.random.seed(seed)
        ckpt = ckpt_root / f"{scale[0]}x{scale[1]}"
        _, norm, hist = train_mlp_surrogate(tr, va, cfg, device="cpu", checkpoint_dir=str(ckpt),
                                            verbose=False, flatten_fn=flatten_fn)
        model, norm = load_trained_mlp(ckpt / "best.pt", device="cpu")   # best validation epoch
        normalize_features(te, norm)
        X, _ = (flatten_fn or _heterodata_to_flat)(te, norm)
        with torch.no_grad():
            pred = norm.denormalize(model(X)).numpy().astype(np.float32)
        key = f"{scale[0]}x{scale[1]}"
        per_scale[key] = compute_metrics(pred, y_raw)
        info[key] = {"input_dim": int(X.shape[1]), "epochs_run": len(hist["val_loss"]),
                     "best_epoch": int(np.argmin(hist["val_loss"])), "n_train": len(tr),
                     "n_val": len(va), "n_test": len(te)}
        pooled_p.append(pred); pooled_y.append(y_raw)
        print(f"    MLP {key} seed {seed}: input {X.shape[1]}, best epoch {info[key]['best_epoch']}, "
              + " ".join(f"{o} rho={per_scale[key][o]['spearman']:+.3f} R2={per_scale[key][o]['r2']:+.2f}"
                         for o in ("profit", "surplus", "shortfall")), flush=True)
    return {"aggregate": compute_metrics(np.concatenate(pooled_p), np.concatenate(pooled_y)),
            "per_scale": per_scale, "info": info}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/surrogate/phase7_synthetic")
    ap.add_argument("--config", default="configs/surrogate.yaml")
    ap.add_argument("--output", default="results/flat_baselines")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n-jobs", dest="n_jobs", type=int, default=-1)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    d = Path(a.data)
    train, val, test = (pickle.load(open(d / f"{k}.pkl", "rb")) for k in ("train", "val", "test"))
    tc = yaml.safe_load(open(a.config)).get("training", {})
    cfg = TrainConfig(learning_rate=tc.get("learning_rate", 1e-3), weight_decay=tc.get("weight_decay", 1e-5),
                      epochs=tc.get("epochs", 300), patience=tc.get("patience", 30),
                      batch_size=tc.get("batch_size", 64), min_lr=tc.get("min_lr", 1e-6),
                      profit_weight=tc.get("profit_weight", 0.34), surplus_weight=tc.get("surplus_weight", 0.33),
                      shortfall_weight=tc.get("shortfall_weight", 0.33))
    n_est = 400
    if a.quick:
        rng = np.random.default_rng(0)
        train = [train[i] for i in rng.choice(len(train), 1500, replace=False)]
        cfg.epochs, n_est, a.seeds = 3, 20, "0"
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    res = {"published_input": {}, "reduced_input": {}}
    for name, feat_np, feat_t in (("published_input", None, None), ("reduced_input", reduced_np, reduced_torch)):
        print(f"== {name}", flush=True)
        for kind in ("rf", "xgb"):
            r = train_classical_per_scale(train, test, kind=kind, n_estimators=n_est, n_jobs=a.n_jobs,
                                          feature_fn=feat_np)
            res[name][kind] = r
            print(f"  {kind}: " + " ".join(f"{o}={r['aggregate'][o]['spearman']:+.3f}"
                                           for o in ("profit", "surplus", "shortfall")), flush=True)
        res[name]["mlp"] = {}
        for seed in [int(s) for s in a.seeds.split(",")]:
            r = mlp_per_scale(train, val, test, cfg, feat_t, seed, out / "ckpt" / name / f"seed{seed}")
            res[name]["mlp"][f"seed{seed}"] = r
            print(f"  mlp seed {seed} pooled: " + " ".join(f"{o}={r['aggregate'][o]['spearman']:+.3f}"
                                                         for o in ("profit", "surplus", "shortfall")), flush=True)
        json.dump(res, open(out / "results.json", "w"), indent=2, default=float)
    if not a.quick:   # the published-input trees must reproduce the stored Table 6 rows
        for kind in ("rf", "xgb"):
            old = json.load(open(ROOT / f"results/baselines/metrics_{kind}.json"))["aggregate"]
            for o in ("profit", "surplus", "shortfall"):
                new = res["published_input"][kind]["aggregate"][o]["spearman"]
                print(f"reproduce {kind} {o}: {new:.4f} vs stored {old[o]['spearman']:.4f}", flush=True)


if __name__ == "__main__":
    main()
