#!/usr/bin/env python3
"""Training-data sampling-composition ablation.

Are the three sampling sources (Latin hypercube,
NSGA-II population snapshots, Pareto perturbations) and their 60/30/10 mix
justified? Regenerates medium-scale datasets under different compositions,
retrains the GNN, and compares held-out accuracy. Self-contained: train and
test instances are generated fresh per composition (disjoint seeds).

Launch unbuffered (regenerates data, so slower):
    python -u scripts/run_sampling_ablation.py --device cuda --epochs 300 \
        --output results/sampling_ablation
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
from src.greenwheel.surrogate.dataset import (
    generate_lhs_samples, generate_nsga2_snapshots, generate_pareto_perturbations,
)
from src.greenwheel.surrogate.graph import instance_to_heterodata
from src.greenwheel.surrogate.training import TrainConfig, train_surrogate
from src.greenwheel.surrogate.validation import evaluate_model

OBJS = ["profit", "surplus", "shortfall"]
SNAP = [0, 50, 100, 150, 200]

# composition: (n_lhs, snapshot_gens, n_perturb_per) per instance
COMPOSITIONS = {
    "full-mix (60/30/10)": (1000, SNAP, 8),
    "LHS-only":            (1700, [], 0),
    "no-LHS (snap+perturb)": (0, SNAP, 40),
}

def _build(instance, n_lhs, snap_gens, n_perturb, seed):
    samples = []
    if n_lhs > 0:
        samples += generate_lhs_samples(instance, n_lhs, seed=seed)
    nsga2 = generate_nsga2_snapshots(instance, pop_size=100, n_generations=200,
                                     snapshot_gens=snap_gens, seed=seed) if snap_gens else []
    samples += nsga2
    if nsga2 and n_perturb > 0:
        samples += generate_pareto_perturbations(nsga2[-100:][:20], instance,
                                                 n_perturbations_per=n_perturb, seed=seed + 5000)
    graphs = []
    for W, p, s, sh in samples:
        graphs.append(instance_to_heterodata(instance, W,
                                             np.array([p, s, sh], dtype=np.float32)))
    return graphs

def _dataset(seeds, m, n, comp):
    cfg = WheelingConfig(); cfg.n_generators, cfg.n_consumers = m, n
    out = []
    for sd in seeds:
        inst = generate_instance(cfg, seed=sd)
        out += _build(inst, comp[0], comp[1], comp[2], seed=sd)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--m", type=int, default=10)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--train-seeds", type=int, default=6)
    ap.add_argument("--output", default="results/sampling_ablation")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    epochs = 15 if args.quick else args.epochs
    n_train = 2 if args.quick else args.train_seeds

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    tr_seeds = [3000 + i for i in range(n_train)]
    va_seeds = [3500, 3501]
    te_seeds = [3600, 3601]

    # Shared validation/test sets use the full mix (fair, fixed evaluation target)
    print("Building shared val/test sets...", flush=True)
    val = _dataset(va_seeds, args.m, args.n, COMPOSITIONS["full-mix (60/30/10)"])
    test = _dataset(te_seeds, args.m, args.n, COMPOSITIONS["full-mix (60/30/10)"])
    print(f"  val={len(val)} test={len(test)}", flush=True)

    results = []
    for name, comp in (list(COMPOSITIONS.items())[:2] if args.quick else COMPOSITIONS.items()):
        t0 = time.time()
        train = _dataset(tr_seeds, args.m, args.n, comp)
        torch.manual_seed(0); np.random.seed(0)
        model, norm, _ = train_surrogate(
            [d.clone() for d in train], [d.clone() for d in val],
            TrainConfig(epochs=epochs), device=args.device,
            checkpoint_dir=str(out / f"_tmp_{name.split()[0]}"), verbose=False)
        m = evaluate_model(model, test, norm, device=args.device)
        row = {"composition": name, "n_train": len(train),
               **{o: {"spearman": m[o]["spearman"], "r2": m[o]["r2"]} for o in OBJS}}
        results.append(row)
        print(f"  {name:24s} n_train={len(train):>5d}  "
              f"profit rho={m['profit']['spearman']:.4f}  "
              f"surplus rho={m['surplus']['spearman']:.4f}  "
              f"shortfall rho={m['shortfall']['spearman']:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    json.dump(results, open(out / "results.json", "w"), indent=2, default=str)
    print(f"\nSaved -> {out}/results.json", flush=True)

if __name__ == "__main__":
    main()
