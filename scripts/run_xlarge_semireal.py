#!/usr/bin/env python3
"""Semi-real optimization quality at xlarge scale (50x100, 5,000 variables).

This fills the one empty cell of the experimental design. The manuscript's
semi-real experiments cover small, medium and large only, while the wall-clock
benefit of pre-screening appears solely at xlarge -- so every semi-real result
so far sits inside the scale band where the method is not expected to pay off.

Protocol matches the published semi-real run (7E) so the numbers are comparable:
seeds 42 + 74000 + 100*i, and a per-instance reference point at 1.1x the
componentwise worst over all methods compared on that instance.

Sharded by seed so several processes can run concurrently:
    python3 scripts/run_xlarge_semireal.py --seeds 0,1 --output results/xlarge_semireal
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
from src.greenwheel.optimization.moead import run_moead
from src.greenwheel.optimization.nsga2 import run_nsga2
from src.greenwheel.optimization.nsga3 import run_nsga3
from src.greenwheel.optimization.samoea import SAMOEAConfig, run_samoea
from src.greenwheel.surrogate.inference import BatchSurrogate
from src.greenwheel.surrogate.training import load_trained_model

CKPT = "checkpoints/surrogate_phase7_semireal/best.pt"
M, N = 50, 100

VANILLA = {"NSGA-II": run_nsga2, "MOEA/D": run_moead, "NSGA-III": run_nsga3}
SA = {"SA-NSGA-II": "nsga2", "SA-MOEA/D": "moead", "SA-NSGA-III": "nsga3"}


def _common_ref_point(*fronts):
    valid = [f for f in fronts if f is not None and len(f) > 0]
    if not valid:
        return np.ones(3) * 1e10
    return np.vstack(valid).max(axis=0) * 1.1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--seeds", default="0,1,2,3,4",
                   help="comma-separated seed indices i; seed = 42 + 74000 + 100*i")
    p.add_argument("--n-gen", dest="n_gen", type=int, default=200)
    p.add_argument("--output", default="results/xlarge_semireal")
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    model, norm = load_trained_model(CKPT, device=args.device)
    surrogate = BatchSurrogate(model, norm, device=args.device)

    for idx in [int(s) for s in args.seeds.split(",")]:
        seed = 42 + 74000 + idx * 100
        print(f"\n=== xlarge semi-real ({M}x{N} = {M*N} vars), seed {seed} ===", flush=True)

        config = WheelingConfig()
        config.n_generators, config.n_consumers = M, N
        # Applies to the vanilla algorithms as well, so that a reduced-budget
        # smoke test exercises every variant rather than only the SA ones.
        config.n_generations = args.n_gen
        instance = generate_semireal_instance(config, seed=seed)
        surrogate.set_instance(instance)

        fronts, recs = {}, {}

        for name, fn in VANILLA.items():
            t0 = time.time()
            r = fn(instance, config, seed=seed, verbose=False)
            fronts[name] = r.pareto_front
            recs[name] = {"seed": seed, "elapsed": time.time() - t0,
                          "n_true": getattr(r, "n_evaluations", None), "savings": 0.0}
            print(f"  {name:12s} done in {recs[name]['elapsed']:6.0f}s "
                  f"({len(fronts[name])} solutions)", flush=True)

        for name, base in SA.items():
            cfg = SAMOEAConfig(base_algorithm=base, correction_fraction=0.5,
                               n_generations=args.n_gen)
            t0 = time.time()
            r = run_samoea(instance, surrogate, cfg, seed=seed, verbose=False)
            fronts[name] = r.pareto_front
            recs[name] = {"seed": seed, "elapsed": time.time() - t0,
                          "n_true": r.n_true_evaluations, "savings": r.eval_savings}
            print(f"  {name:12s} done in {recs[name]['elapsed']:6.0f}s "
                  f"({len(fronts[name])} solutions, {r.eval_savings:.0%} savings)", flush=True)

        ref = _common_ref_point(*fronts.values())
        for name in fronts:
            F = fronts[name]
            recs[name]["hv"] = float(compute_hypervolume(F, ref)) if len(F) else 0.0
            recs[name]["n_solutions"] = int(len(F))

        print("  HV: " + "  ".join(f"{k}={recs[k]['hv']:.3e}" for k in recs), flush=True)
        json.dump(recs, open(out / f"seed{idx}.json", "w"), indent=2, default=str)
        print(f"  -> {out}/seed{idx}.json", flush=True)


if __name__ == "__main__":
    main()
