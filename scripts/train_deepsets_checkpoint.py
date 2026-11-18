#!/usr/bin/env python3
"""Train the DeepSets baseline once more and save it for deployment.

The Table 6 run (scripts/train_baselines.py) kept only metrics, and its training is
unseeded, so this retraining cannot reproduce that model bit for bit. It is accepted
only if its held-out Spearman matches Table 6 within the spread one would expect from
reseeding; the script prints both so the comparison is explicit.

    python -u scripts/train_deepsets_checkpoint.py --seed 0
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

from src.greenwheel.surrogate.baselines import train_deepsets

TABLE6 = "results/baselines/metrics_deepsets.json"
OBJS = ("profit", "surplus", "shortfall")


def _load(d: Path, name: str):
    with open(d / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/surrogate/phase7_synthetic")
    p.add_argument("--output", default="checkpoints/deepsets_phase7_synthetic")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    d = Path(args.data)
    train, val, test = _load(d, "train"), _load(d, "val"), _load(d, "test")
    if args.quick:
        train, val, test, args.epochs = train[:800], val[:200], test[:200], 5
    t0 = time.time()
    res = train_deepsets(train, val, test, epochs=args.epochs, device=args.device,
                         return_model=True)
    model, norm = res.pop("model"), res.pop("norm")
    res["train_seconds"] = round(time.time() - t0, 1)
    res["seed"] = args.seed

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    dims = dict(gen_in=model.phi_gen[0].in_features, con_in=model.phi_con[0].in_features,
                global_dim=model.rho[0].in_features - 4 * model.rho[0].out_features)
    torch.save({"state_dict": model.state_dict(), "norm": norm, "dims": dims}, out / "best.pt")
    json.dump(res, open(out / "metrics.json", "w"), indent=2, default=str)

    ref = json.load(open(TABLE6))["aggregate"]
    print(f"\nepochs {res['epochs_run']}, {res['train_seconds']} s, params {res['n_params']}")
    print(f"{'':10s} {'Table 6':>8s} {'retrain':>8s}")
    for o in OBJS:
        print(f"{o:10s} {ref[o]['spearman']:8.3f} {res['aggregate'][o]['spearman']:8.3f}")
    print(f"saved -> {out}/best.pt")


if __name__ == "__main__":
    main()
