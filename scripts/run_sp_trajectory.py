#!/usr/bin/env python3
"""In-run surrogate reliability trajectory, on the accuracy scale of Hanawa et al.

Hanawa, Harada & Miura, Array 27 (2025) 100461, control surrogate quality with a
pseudo-surrogate whose pairwise comparison is correct with probability sp, and
report that SAEAs beat the no-surrogate baseline once sp > 0.6, while sp ~ 0.5
is no better than using no surrogate at all. They note their pseudo-surrogate
"does not replicate the complex error patterns ... commonly observed in real
machine learning-based surrogate models".

This script measures what a *real* learned surrogate's effective sp looks like
along a real run. Full-evaluation generations already produce exact labels for
the deployed instance's own population, so one extra forward pass per such
generation yields sp = (1 + tau) / 2 at zero simulation cost.

Reported per objective and aggregated, so the collapse of local ordering can be
compared against the 0.6 threshold without relying on a single worst-case
snapshot.

    python -u scripts/run_sp_trajectory.py --device cpu \
        --scales medium,large --n-seeds 5 --output results/sp_trajectory
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.greenwheel.config import WheelingConfig
from src.greenwheel.data.synthetic import generate_instance, generate_semireal_instance
from src.greenwheel.optimization.samoea import SAMOEAConfig, run_samoea
from src.greenwheel.surrogate.inference import BatchSurrogate
from src.greenwheel.surrogate.training import load_trained_model

SIZE = {"small": (5, 10), "medium": (10, 20), "large": (20, 50)}
DOMAINS = {
    "synthetic": ("checkpoints/surrogate_phase7_synthetic/best.pt", generate_instance),
    "semireal": ("checkpoints/surrogate_phase7_semireal/best.pt", generate_semireal_instance),
}
OBJS = ["profit", "surplus", "shortfall"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--scales", default="medium,large")
    p.add_argument("--n-seeds", dest="n_seeds", type=int, default=5)
    p.add_argument("--n-gen", dest="n_gen", type=int, default=200)
    p.add_argument("--output", default="results/sp_trajectory")
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    for dom, (ckpt, gen_fn) in DOMAINS.items():
        model, norm = load_trained_model(ckpt, device=args.device)
        surrogate = BatchSurrogate(model, norm, device=args.device)
        report[dom] = {}

        for scale in args.scales.split(","):
            m, n = SIZE[scale]
            print(f"\n=== {dom} / {scale} ({m}x{n}) ===", flush=True)
            runs = []

            for i in range(args.n_seeds):
                seed = 42 + 74000 + i * 100
                cfg = WheelingConfig()
                cfg.n_generators, cfg.n_consumers = m, n
                inst = gen_fn(cfg, seed=seed)
                surrogate.set_instance(inst)

                sa_cfg = SAMOEAConfig(
                    base_algorithm="nsga2", correction_fraction=0.5,
                    n_generations=args.n_gen, probe_enabled=True,
                )
                t0 = time.time()
                res = run_samoea(inst, surrogate, sa_cfg, seed=seed, verbose=False)

                traj = [
                    {"gen": s.generation, "mode": s.mode, "n": s.n_probe,
                     "tau": s.probe_tau, "sp": s.probe_sp, "rho": s.probe_rho}
                    for s in res.gen_stats if s.probe_sp is not None
                ]
                # Aggregate over the run. Warmup generations are recorded but
                # kept separate: they precede any screening and their
                # populations are still diverse, so including them would
                # flatter the estimate of what screening actually operates on.
                post = [t for t in traj if t["mode"] == "full_eval"]
                sp_mean = float(np.mean([np.mean(t["sp"]) for t in post])) if post else float("nan")
                sp_min = float(np.min([np.min(t["sp"]) for t in post])) if post else float("nan")
                runs.append({"seed": seed, "elapsed": time.time() - t0,
                             "sp_mean_fulleval": sp_mean, "sp_min_fulleval": sp_min,
                             "trajectory": traj})
                print(f"  seed {i+1} ({seed}): sp(mean over full-eval gens)={sp_mean:.3f} "
                      f"sp(min)={sp_min:.3f}  n_probe_gens={len(post)} "
                      f"({time.time()-t0:.0f}s)", flush=True)

            sps = [r["sp_mean_fulleval"] for r in runs]
            print(f"  >>> sp = {np.mean(sps):.3f} +- {np.std(sps):.3f} "
                  f"[{np.min(sps):.3f}, {np.max(sps):.3f}]   "
                  f"(Hanawa et al. threshold: 0.6)", flush=True)
            report[dom][scale] = runs

    json.dump(report, open(out / "results.json", "w"), indent=2, default=str)
    print(f"\nSaved -> {out}/results.json", flush=True)


if __name__ == "__main__":
    main()
