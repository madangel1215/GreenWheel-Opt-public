#!/usr/bin/env python3
"""Screening recoverability: hard discard vs soft retention on semi-real data.

Tests whether carrying screened-out offspring forward with surrogate-predicted
objectives (``screening_mode="retain"``) removes the harm that irrecoverable
rejection (``"discard"``) causes when the surrogate's local ranking is weak.

Small scale is the decisive case: the manuscript reports pre-screening there as
significantly *harmful* on semi-real instances (p = 0.004, A12 = 0.44), which is
the signature of false negatives destroying good candidates outright.

Seeds and the shared-reference-point protocol match the published semi-real
experiment (7E in ``run_phase7_experiments.py``) so results are directly
comparable: seed = 42 + 74000 + 100*i, ref point = 1.1 x componentwise worst
over all methods compared on that instance.

    python -u scripts/run_recoverability_experiment.py --device cpu \
        --scales small,medium,large --n-seeds 10 \
        --output results/recoverability
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.greenwheel.config import WheelingConfig
from src.greenwheel.data.synthetic import generate_semireal_instance
from src.greenwheel.optimization.metrics import compute_hypervolume
from src.greenwheel.optimization.nsga2 import run_nsga2
from src.greenwheel.optimization.samoea import SAMOEAConfig, run_samoea
from src.greenwheel.surrogate.inference import BatchSurrogate
from src.greenwheel.surrogate.training import load_trained_model

SIZE = {"small": (5, 10), "medium": (10, 20), "large": (20, 50)}
CKPT = "checkpoints/surrogate_phase7_semireal/best.pt"


def _common_ref_point(*fronts: np.ndarray) -> np.ndarray:
    """Shared per-instance reference point: 1.1 x componentwise worst."""
    valid = [f for f in fronts if f is not None and len(f) > 0]
    if not valid:
        return np.ones(3) * 1e10
    combined = np.vstack(valid)
    return combined.max(axis=0) * 1.1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--scales", default="small,medium,large")
    p.add_argument("--n-seeds", dest="n_seeds", type=int, default=10)
    p.add_argument("--n-gen", dest="n_gen", type=int, default=200)
    p.add_argument("--output", default="results/recoverability")
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    model, norm = load_trained_model(CKPT, device=args.device)
    surrogate = BatchSurrogate(model, norm, device=args.device)

    variants = ["NSGA-II", "SA-discard", "SA-retain"]
    report = {}

    for scale in args.scales.split(","):
        m, n = SIZE[scale]
        print(f"\n=== semireal / {scale} ({m}x{n}) ===", flush=True)
        rows = {v: [] for v in variants}

        for i in range(args.n_seeds):
            seed = 42 + 74000 + i * 100
            config = WheelingConfig()
            config.n_generators, config.n_consumers = m, n
            instance = generate_semireal_instance(config, seed=seed)
            surrogate.set_instance(instance)

            fronts, recs = {}, {}

            t0 = time.time()
            r = run_nsga2(instance, config, seed=seed, verbose=False)
            fronts["NSGA-II"] = r.pareto_front
            recs["NSGA-II"] = {"seed": seed, "elapsed": time.time() - t0,
                               "n_true": getattr(r, "n_evaluations", None),
                               "savings": 0.0, "n_retained": 0}

            for name, mode in (("SA-discard", "discard"), ("SA-retain", "retain")):
                sa_cfg = SAMOEAConfig(base_algorithm="nsga2", correction_fraction=0.5,
                                      n_generations=args.n_gen, screening_mode=mode)
                t0 = time.time()
                sr = run_samoea(instance, surrogate, sa_cfg, seed=seed, verbose=False)
                fronts[name] = sr.pareto_front
                recs[name] = {
                    "seed": seed, "elapsed": time.time() - t0,
                    "n_true": sr.n_true_evaluations,
                    "savings": sr.eval_savings,
                    "n_retained": int(sum(s.n_retained for s in sr.gen_stats)),
                }

            ref = _common_ref_point(*fronts.values())
            line = []
            for v in variants:
                F = fronts[v]
                hv = compute_hypervolume(F, ref) if len(F) > 0 else 0.0
                recs[v]["hv"] = float(hv)
                recs[v]["n_solutions"] = int(len(F))
                rows[v].append(recs[v])
                line.append(f"{v}={hv:.3e}")
            print(f"  seed {i+1:2d} ({seed}): " + "  ".join(line), flush=True)

        report[scale] = rows
        print("  >>> median HV: " + "  ".join(
            f"{v}={np.median([r['hv'] for r in rows[v]]):.3e}" for v in variants), flush=True)

    json.dump(report, open(out / "results.json", "w"), indent=2, default=str)
    print(f"\nSaved -> {out}/results.json", flush=True)


if __name__ == "__main__":
    main()
