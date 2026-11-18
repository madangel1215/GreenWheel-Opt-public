#!/usr/bin/env python3
"""Does DeepSets deliver the GNN's gain inside SA-NSGA-II?

Reruns the large-scale Table 7 experiment (same seeds, same settings) with a fourth
arm, SA-NSGA-II with the DeepSets surrogate. Vanilla NSGA-II, SA-GNN and SA-MLP are
rerun rather than read back so that all four arms share one reference point per
instance. The three-arm hypervolumes are also recomputed with Table 7's own reference
point (vanilla, GNN, MLP) and compared with the stored Table 7 values: agreement shows
the rerun reproduces the published experiment, which is what makes the DeepSets arm
comparable to it.

    python -u scripts/run_deepsets_downstream.py --output results/deepsets_downstream
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.greenwheel.config import WheelingConfig
from src.greenwheel.data.synthetic import generate_instance
from src.greenwheel.optimization.metrics import compute_hypervolume
from src.greenwheel.optimization.nsga2 import run_nsga2
from src.greenwheel.optimization.samoea import SAMOEAConfig, run_samoea
from src.greenwheel.surrogate.deepsets_inference import load_deepsets
from src.greenwheel.surrogate.inference import BatchSurrogate
from src.greenwheel.surrogate.mlp_inference import MLPBatchSurrogate
from src.greenwheel.surrogate.mlp_training import load_trained_mlp
from src.greenwheel.surrogate.training import load_trained_model

TABLE7 = ROOT / "results/phase7/7d_gnn_vs_mlp/results.json"


def ref_point(*fronts):  # identical to run_phase7_experiments._common_ref_point
    valid = [f for f in fronts if len(f) > 0]
    return np.vstack(valid).max(axis=0) * 1.1 if valid else np.ones(3) * 1e10


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gnn", default="checkpoints/surrogate_phase7_synthetic/best.pt")
    p.add_argument("--mlp", default="checkpoints/surrogate_mlp_phase7_large/best.pt")
    p.add_argument("--deepsets", default="checkpoints/deepsets_phase7_synthetic/best.pt")
    p.add_argument("--n-seeds", dest="n_seeds", type=int, default=10)
    p.add_argument("--n-gen", dest="n_gen", type=int, default=200)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="results/deepsets_downstream")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    m, n = 20, 50
    if args.quick:
        args.n_seeds, args.n_gen, m, n = 1, 10, 5, 10

    model, norm = load_trained_model(args.gnn, device=args.device)
    sur = {"GNN": BatchSurrogate(model, norm, device=args.device)}
    mm, mn = load_trained_mlp(args.mlp, device=args.device)
    sur["MLP"] = MLPBatchSurrogate(mm, mn, device=args.device)
    sur["DeepSets"] = load_deepsets(args.deepsets, device=args.device)
    if args.quick:
        sur.pop("MLP")  # the large-scale MLP has a fixed input size

    t7 = {r["seed"]: r for r in json.load(open(TABLE7))["large"]}
    rows = []
    for i in range(args.n_seeds):
        seed = 42 + 73000 + i * 100                      # Table 7's seeds
        cfg = WheelingConfig()
        cfg.n_generators, cfg.n_consumers = m, n
        cfg.n_generations, cfg.population_size = args.n_gen, 100
        inst = generate_instance(cfg, seed=seed)
        t0 = time.time()
        F = {"vanilla": run_nsga2(inst, cfg, seed=seed, verbose=False).pareto_front}
        for name, s in sur.items():
            s.set_instance(inst)
            F[name] = run_samoea(inst, s, SAMOEAConfig(correction_fraction=0.5,
                                                       n_generations=args.n_gen),
                                 seed=seed, verbose=False).pareto_front
        ref4 = ref_point(*F.values())
        row = {"seed": seed, "hv": {k: compute_hypervolume(f, ref4) for k, f in F.items()}}
        if "MLP" in F:
            ref3 = ref_point(F["vanilla"], F["GNN"], F["MLP"])
            row["hv_table7_ref"] = {k: compute_hypervolume(F[k], ref3) for k in ("vanilla", "GNN", "MLP")}
            if seed in t7:
                row["table7_stored"] = {"vanilla": t7[seed]["hv_vanilla"], "GNN": t7[seed]["hv_gnn"],
                                        "MLP": t7[seed]["hv_mlp"]}
        rows.append(row)
        h = row["hv"]
        print(f"seed {seed}: " + "  ".join(f"{k}={v/1e16:.2f}" for k, v in h.items())
              + f"  ({time.time()-t0:.0f}s)", flush=True)
        if "table7_stored" in row:
            print("   reproduce Table 7: " + "  ".join(
                f"{k} {row['hv_table7_ref'][k]/1e16:.3f} vs {row['table7_stored'][k]/1e16:.3f}"
                for k in ("vanilla", "GNN", "MLP")), flush=True)

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    summary = {}
    if len(rows) >= 5:
        A = {k: np.array([r["hv"][k] for r in rows]) for k in rows[0]["hv"]}
        for a, b in [("DeepSets", "vanilla"), ("GNN", "vanilla"), ("DeepSets", "GNN")]:
            d = A[a] - A[b]
            summary[f"{a}_vs_{b}"] = {"mean_a": float(A[a].mean()), "mean_b": float(A[b].mean()),
                                      "ratio": float(A[a].mean() / A[b].mean()),
                                      "a_wins": int((d > 0).sum()), "n": len(d),
                                      "wilcoxon_p": float(wilcoxon(A[a], A[b]).pvalue)}
        print(json.dumps(summary, indent=2))
    json.dump({"rows": rows, "summary": summary, "args": vars(args)},
              open(out / "results.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
