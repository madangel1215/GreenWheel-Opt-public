#!/usr/bin/env python3
"""Phase 7: 3-Objective Multi-Algorithm Experiments for SCI Paper.

Factorial design experiments with 3 objectives (profit, surplus, shortfall)
and multi-algorithm SA-MOEA framework.

Experiments:
  7A - Main comparison: 3 MOEAs × 2 (vanilla/SA) × 3 scales × 10 seeds = 180 runs
  7B - Ablation on correction_fraction: 5 levels × 10 seeds × medium scale
  7C - Scalability: best method vs vanilla × 5 scales × 5 seeds
  7D - GNN vs MLP surrogate: best optimizer × 2 surrogates × 3 scales × 10 seeds
  7E - Semi-real validation: best method × 10 seeds × 3 scales
  7F - Statistical summary: Friedman + Nemenyi post-hoc

Usage:
    # Main comparison (recommended: run overnight)
    python scripts/run_phase7_experiments.py --exp 7a \
        --model checkpoints/surrogate_phase7_synthetic/best.pt

    # Quick test (1 seed, small, 30 gens)
    python scripts/run_phase7_experiments.py --exp 7a \
        --model checkpoints/surrogate_phase7_synthetic/best.pt --quick

    # Semi-real validation
    python scripts/run_phase7_experiments.py --exp 7e \
        --model checkpoints/surrogate_phase7_semireal/best.pt

    # Statistical summary (requires 7A results)
    python scripts/run_phase7_experiments.py --exp 7f
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
from src.greenwheel.optimization.metrics import (
    compute_hypervolume,
    run_statistical_comparison,
)
from src.greenwheel.optimization.moead import run_moead
from src.greenwheel.optimization.nsga2 import run_nsga2
from src.greenwheel.optimization.nsga3 import run_nsga3
from src.greenwheel.optimization.samoea import SAMOEAConfig, run_samoea

# ---------------------------------------------------------------------------
# Problem scale configurations
# ---------------------------------------------------------------------------

SIZE_CONFIGS = {
    "small": {"n_generators": 5, "n_consumers": 10},
    "medium": {"n_generators": 10, "n_consumers": 20},
    "large": {"n_generators": 20, "n_consumers": 50},
    "xlarge": {"n_generators": 50, "n_consumers": 100},
    "xxlarge": {"n_generators": 50, "n_consumers": 200},
}

VANILLA_ALGORITHMS = {
    "NSGA-II": run_nsga2,
    "MOEA/D": run_moead,
    "NSGA-III": run_nsga3,
}

SA_ALGORITHM_MAP = {
    "SA-NSGA-II": "nsga2",
    "SA-MOEA/D": "moead",
    "SA-NSGA-III": "nsga3",
}

def _make_config(size_cfg: dict, n_gen: int, pop_size: int = 100) -> WheelingConfig:
    cfg = WheelingConfig()
    cfg.n_generators = size_cfg["n_generators"]
    cfg.n_consumers = size_cfg["n_consumers"]
    cfg.n_generations = n_gen
    cfg.population_size = pop_size
    return cfg

def _load_surrogate(model_path: str, device: str):
    from src.greenwheel.surrogate.inference import BatchSurrogate
    from src.greenwheel.surrogate.training import load_trained_model

    model, norm = load_trained_model(model_path, device=device)
    return BatchSurrogate(model, norm, device=device)

def _common_ref_point(*fronts: np.ndarray) -> np.ndarray:
    """Compute common reference point from multiple Pareto fronts."""
    valid = [f for f in fronts if len(f) > 0]
    if not valid:
        return np.ones(3) * 1e10
    combined = np.vstack(valid)
    return combined.max(axis=0) * 1.1

# ===================================================================
# Experiment 7A: Main factorial comparison
# 3 MOEAs × 2 (vanilla/SA) × 3 scales × 10 seeds = 180 runs
# ===================================================================

def run_7a(args):
    out_dir = Path(args.output) / "7a_main_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    surrogate = _load_surrogate(args.model, args.device)

    if args.scales:
        scales = {k.strip(): SIZE_CONFIGS[k.strip()]
                  for k in args.scales.split(",")}
    elif args.quick:
        scales = {"small": SIZE_CONFIGS["small"]}
    else:
        scales = {k: SIZE_CONFIGS[k] for k in ["small", "medium", "large"]}
    n_seeds = args.n_seeds if args.n_seeds is not None else (1 if args.quick else 10)

    # All 6 algorithm variants
    all_variants = list(VANILLA_ALGORITHMS.keys()) + list(SA_ALGORITHM_MAP.keys())
    total_runs = len(all_variants) * len(scales) * n_seeds
    run_count = 0

    all_results = {}

    for scale_name, size_cfg in scales.items():
        print(f"\n{'='*60}")
        print(f"[7A] {scale_name} ({size_cfg['n_generators']}×{size_cfg['n_consumers']}), "
              f"{n_seeds} seeds, 6 variants")
        print(f"{'='*60}")

        config = _make_config(size_cfg, args.n_gen)
        scale_results = {v: [] for v in all_variants}

        for i in range(n_seeds):
            seed = args.seed + 70000 + i * 100
            instance = generate_instance(config, seed=seed)
            print(f"\n  Seed {i+1}/{n_seeds} (seed={seed})")

            # Collect all fronts for common reference point
            all_fronts = {}

            # Run vanilla algorithms
            for alg_name, alg_fn in VANILLA_ALGORITHMS.items():
                run_count += 1
                t0 = time.time()
                result = alg_fn(instance, config, seed=seed, verbose=False)
                elapsed = time.time() - t0
                F = result.pareto_front

                all_fronts[alg_name] = F
                scale_results[alg_name].append({
                    "seed": seed,
                    "elapsed": elapsed,
                    "n_evaluations": result.n_evaluations,
                    "n_solutions": len(result.pareto_solutions),
                    "pareto_front": F.tolist(),
                })
                print(f"    [{run_count}/{total_runs}] {alg_name:15s}: {elapsed:.1f}s, "
                      f"{len(result.pareto_solutions)} solutions")

            # Run SA variants
            surrogate.set_instance(instance)
            for sa_name, base_algo in SA_ALGORITHM_MAP.items():
                run_count += 1
                sa_config = SAMOEAConfig(
                    base_algorithm=base_algo,
                    correction_fraction=0.5,
                    n_generations=args.n_gen,
                )
                t0 = time.time()
                sa_result = run_samoea(
                    instance, surrogate, sa_config, seed=seed, verbose=False,
                )
                elapsed = time.time() - t0
                F = sa_result.pareto_front

                all_fronts[sa_name] = F
                scale_results[sa_name].append({
                    "seed": seed,
                    "elapsed": elapsed,
                    "n_evaluations": sa_result.n_true_evaluations,
                    "n_surrogate_evaluations": sa_result.n_surrogate_evaluations,
                    "eval_savings": sa_result.eval_savings,
                    "n_solutions": len(sa_result.pareto_solutions),
                    "pareto_front": F.tolist(),
                })
                print(f"    [{run_count}/{total_runs}] {sa_name:15s}: {elapsed:.1f}s, "
                      f"{len(sa_result.pareto_solutions)} solutions, "
                      f"savings={sa_result.eval_savings*100:.1f}%")

            # Compute HV with common reference point
            ref_point = _common_ref_point(*all_fronts.values())
            for alg_name in all_variants:
                F = np.array(all_fronts.get(alg_name, np.empty((0, 3))))
                hv = compute_hypervolume(F, ref_point) if len(F) > 0 else 0.0
                scale_results[alg_name][-1]["hv"] = hv

            # Print HV summary for this seed
            hvs = {v: scale_results[v][-1]["hv"] for v in all_variants}
            best_v = max(hvs, key=lambda v: hvs[v])
            print(f"    HV: " + ", ".join(
                f"{'*' if v == best_v else ''}{v}={hvs[v]:.4e}"
                for v in all_variants
            ))

        all_results[scale_name] = scale_results

        # Statistical tests (Friedman + pairwise)
        if n_seeds >= 5:
            hv_data = {v: [r["hv"] for r in results] for v, results in scale_results.items()}
            stat = run_statistical_comparison(
                hv_data, metric_name="hypervolume", higher_is_better=True,
            )

            print(f"\n  [{scale_name}] Statistical summary:")
            for pw in stat["pairwise"]:
                print(f"    {pw['method_a']:15s} vs {pw['method_b']:15s}: "
                      f"p={pw['wilcoxon']['p_value']:.4f}, "
                      f"A12={pw['a12']['a12']:.3f} ({pw['a12']['magnitude']})")

            # Save statistics
            stat_path = out_dir / f"statistics_{scale_name}.json"
            with open(stat_path, "w") as f:
                json.dump(stat, f, indent=2, default=str)

    # Save all results
    _save_results(all_results, out_dir / "results.json")

    # Generate summary table
    _print_summary_table(all_results, all_variants)

    print(f"\n[7A] Results saved to {out_dir}")

def _print_summary_table(all_results: dict, variants: list):
    """Print a summary table of mean HV across scales and variants."""
    print(f"\n{'='*80}")
    print("SUMMARY: Mean HV (± std) across seeds")
    print(f"{'='*80}")

    header = f"{'Scale':>10s}"
    for v in variants:
        header += f" | {v:>14s}"
    print(header)
    print("-" * len(header))

    for scale_name, scale_results in all_results.items():
        row = f"{scale_name:>10s}"
        for v in variants:
            hvs = [r["hv"] for r in scale_results[v]]
            row += f" | {np.mean(hvs):>10.2e}±{np.std(hvs):.1e}"
        print(row)

# ===================================================================
# Experiment 7B: Ablation on correction_fraction
# ===================================================================

def run_7b(args):
    out_dir = Path(args.output) / "7b_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    surrogate = _load_surrogate(args.model, args.device)

    cf_values = [0.3, 0.4, 0.5, 0.6, 0.7] if not args.quick else [0.3, 0.5, 0.7]
    n_seeds = 1 if args.quick else 10
    size_cfg = SIZE_CONFIGS["medium"]

    print(f"\n{'='*60}")
    print(f"[7B] Ablation: cf={cf_values}, "
          f"medium ({size_cfg['n_generators']}×{size_cfg['n_consumers']}), "
          f"{n_seeds} seeds")
    print(f"{'='*60}")

    config = _make_config(size_cfg, args.n_gen)

    # Generate instances
    instances = []
    seeds = []
    for i in range(n_seeds):
        seed = args.seed + 71000 + i * 100
        seeds.append(seed)
        instances.append(generate_instance(config, seed=seed))

    # Vanilla baselines
    print("\n  Running vanilla baselines...")
    vanilla_hvs = []
    for i in range(n_seeds):
        result = run_nsga2(instances[i], config, seed=seeds[i], verbose=False)
        ref = result.pareto_front.max(axis=0) * 1.1
        hv = compute_hypervolume(result.pareto_front, ref)
        vanilla_hvs.append({"hv": hv, "ref_point": ref.tolist()})
        print(f"    Seed {i+1}: HV={hv:.4e}")

    # SA-MOEA for each cf
    all_results = {"vanilla": vanilla_hvs}
    summary = []

    for cf in cf_values:
        print(f"\n  --- cf={cf} ---")
        cf_results = []

        for i in range(n_seeds):
            surrogate.set_instance(instances[i])
            sa_config = SAMOEAConfig(
                correction_fraction=cf,
                n_generations=args.n_gen,
            )
            sa_result = run_samoea(
                instances[i], surrogate, sa_config, seed=seeds[i], verbose=False,
            )
            F_sa = sa_result.pareto_front

            # Use common ref from both
            vanilla_r = run_nsga2(instances[i], config, seed=seeds[i], verbose=False)
            ref = _common_ref_point(vanilla_r.pareto_front, F_sa)
            hv_v = compute_hypervolume(vanilla_r.pareto_front, ref)
            hv_s = compute_hypervolume(F_sa, ref)

            cf_results.append({
                "seed": seeds[i],
                "hv_vanilla": hv_v,
                "hv_samoea": hv_s,
                "hv_ratio": hv_s / hv_v if hv_v > 0 else 0,
                "eval_savings": sa_result.eval_savings,
            })
            print(f"    Seed {i+1}: ratio={hv_s/hv_v:.4f}, savings={sa_result.eval_savings*100:.1f}%")

        all_results[f"cf_{cf}"] = cf_results
        mean_ratio = float(np.mean([r["hv_ratio"] for r in cf_results]))
        mean_savings = float(np.mean([r["eval_savings"] for r in cf_results]))
        summary.append({"cf": cf, "hv_ratio": mean_ratio, "eval_savings": mean_savings})
        print(f"    Mean: ratio={mean_ratio:.4f}, savings={mean_savings*100:.1f}%")

    _save_results(all_results, out_dir / "results.json")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[7B] Results saved to {out_dir}")

# ===================================================================
# Experiment 7C: Scalability
# ===================================================================

def run_7c(args):
    out_dir = Path(args.output) / "7c_scalability"
    out_dir.mkdir(parents=True, exist_ok=True)

    surrogate = _load_surrogate(args.model, args.device) if args.model else None

    if args.quick:
        scales = {k: SIZE_CONFIGS[k] for k in ["small", "medium"]}
        n_seeds = 1
    else:
        scales = {k: SIZE_CONFIGS[k] for k in ["small", "medium", "large", "xlarge"]}
        n_seeds = 5

    pop_overrides = {"xlarge": 200, "xxlarge": 200}
    gen_overrides = {
        "xlarge": 300 if not args.quick else args.n_gen,
        "xxlarge": 300 if not args.quick else args.n_gen,
    }

    all_results = {}

    for scale_name, size_cfg in scales.items():
        pop = pop_overrides.get(scale_name, 100)
        n_gen = gen_overrides.get(scale_name, args.n_gen)
        n_var = size_cfg["n_generators"] * size_cfg["n_consumers"]

        print(f"\n{'='*60}")
        print(f"[7C] {scale_name} ({size_cfg['n_generators']}×{size_cfg['n_consumers']}, "
              f"{n_var} vars), {n_seeds} seeds")
        print(f"{'='*60}")

        config = _make_config(size_cfg, n_gen, pop)
        scale_results = {"vanilla": [], "sa": []}

        for i in range(n_seeds):
            seed = args.seed + 72000 + i * 100
            instance = generate_instance(config, seed=seed)
            print(f"  Seed {i+1}/{n_seeds} (seed={seed})")

            # Vanilla NSGA-II
            t0 = time.time()
            v_result = run_nsga2(instance, config, seed=seed, verbose=False)
            v_time = time.time() - t0
            v_hv = compute_hypervolume(v_result.pareto_front)

            scale_results["vanilla"].append({
                "seed": seed, "n_var": n_var, "elapsed": v_time,
                "n_evaluations": v_result.n_evaluations, "hv": v_hv,
            })
            print(f"    Vanilla: {v_time:.1f}s, HV={v_hv:.4e}")

            # SA-MOEA (if surrogate available)
            if surrogate is not None:
                surrogate.set_instance(instance)
                sa_config = SAMOEAConfig(
                    correction_fraction=0.5, n_generations=n_gen,
                )
                t0 = time.time()
                sa_result = run_samoea(
                    instance, surrogate, sa_config, seed=seed, verbose=False,
                )
                sa_time = time.time() - t0
                sa_hv = compute_hypervolume(sa_result.pareto_front)

                scale_results["sa"].append({
                    "seed": seed, "n_var": n_var, "elapsed": sa_time,
                    "n_evaluations": sa_result.n_true_evaluations, "hv": sa_hv,
                    "eval_savings": sa_result.eval_savings,
                })
                speedup = v_time / sa_time if sa_time > 0 else 0
                print(f"    SA-MOEA: {sa_time:.1f}s, HV={sa_hv:.4e}, "
                      f"speedup={speedup:.1f}x")

        all_results[scale_name] = scale_results

    _save_results(all_results, out_dir / "results.json")
    print(f"\n[7C] Results saved to {out_dir}")

# ===================================================================
# Experiment 7D: GNN vs MLP surrogate comparison
# ===================================================================

def run_7d(args):
    out_dir = Path(args.output) / "7d_gnn_vs_mlp"
    out_dir.mkdir(parents=True, exist_ok=True)

    gnn_surrogate = _load_surrogate(args.model, args.device)

    scales = {"medium": SIZE_CONFIGS["medium"]} if args.quick else {
        k: SIZE_CONFIGS[k] for k in ["medium", "large"]
    }
    n_seeds = 1 if args.quick else 10

    all_results = {}

    for scale_name, size_cfg in scales.items():
        mlp_path = getattr(args, f"mlp_{scale_name}", None)
        if mlp_path is None:
            print(f"  [7D] Skipping {scale_name}: no --mlp-{scale_name} path")
            continue

        from src.greenwheel.surrogate.mlp_inference import MLPBatchSurrogate
        from src.greenwheel.surrogate.mlp_training import load_trained_mlp
        mlp_model, mlp_norm = load_trained_mlp(mlp_path, device=args.device)
        mlp_surrogate = MLPBatchSurrogate(mlp_model, mlp_norm, device=args.device)

        print(f"\n{'='*60}")
        print(f"[7D] {scale_name}, {n_seeds} seeds, GNN vs MLP")
        print(f"{'='*60}")

        config = _make_config(size_cfg, args.n_gen)
        scale_results = []

        for i in range(n_seeds):
            seed = args.seed + 73000 + i * 100
            instance = generate_instance(config, seed=seed)
            print(f"\n  Seed {i+1}/{n_seeds} (seed={seed})")

            # Vanilla baseline
            v_result = run_nsga2(instance, config, seed=seed, verbose=False)
            F_v = v_result.pareto_front

            # SA-MOEA with GNN
            gnn_surrogate.set_instance(instance)
            sa_config = SAMOEAConfig(correction_fraction=0.5, n_generations=args.n_gen)
            gnn_result = run_samoea(
                instance, gnn_surrogate, sa_config, seed=seed, verbose=False,
            )
            F_gnn = gnn_result.pareto_front

            # SA-MOEA with MLP
            mlp_surrogate.set_instance(instance)
            mlp_result = run_samoea(
                instance, mlp_surrogate, sa_config, seed=seed, verbose=False,
            )
            F_mlp = mlp_result.pareto_front

            ref = _common_ref_point(F_v, F_gnn, F_mlp)
            hv_v = compute_hypervolume(F_v, ref)
            hv_gnn = compute_hypervolume(F_gnn, ref)
            hv_mlp = compute_hypervolume(F_mlp, ref)

            scale_results.append({
                "seed": seed, "hv_vanilla": hv_v,
                "hv_gnn": hv_gnn, "hv_mlp": hv_mlp,
            })
            print(f"    Vanilla={hv_v:.4e}, GNN={hv_gnn:.4e}, MLP={hv_mlp:.4e}")

        all_results[scale_name] = scale_results

        if n_seeds >= 5:
            hv_data = {
                "NSGA-II": [r["hv_vanilla"] for r in scale_results],
                "SA(GNN)": [r["hv_gnn"] for r in scale_results],
                "SA(MLP)": [r["hv_mlp"] for r in scale_results],
            }
            stat = run_statistical_comparison(
                hv_data, metric_name="hypervolume", higher_is_better=True,
            )
            print(f"\n  [{scale_name}] Pairwise:")
            for pw in stat["pairwise"]:
                print(f"    {pw['method_a']} vs {pw['method_b']}: "
                      f"p={pw['wilcoxon']['p_value']:.4f}, "
                      f"A12={pw['a12']['a12']:.3f}")

    _save_results(all_results, out_dir / "results.json")
    print(f"\n[7D] Results saved to {out_dir}")

# ===================================================================
# Experiment 7E: Semi-real validation
# ===================================================================

def run_7e(args):
    out_dir = Path(args.output) / "7e_semireal"
    out_dir.mkdir(parents=True, exist_ok=True)

    surrogate = _load_surrogate(args.model, args.device)

    scales = {"medium": SIZE_CONFIGS["medium"]} if args.quick else {
        k: SIZE_CONFIGS[k] for k in ["small", "medium", "large"]
    }
    n_seeds = 1 if args.quick else 10

    all_results = {}
    all_statistics = {}

    for scale_name, size_cfg in scales.items():
        print(f"\n{'='*60}")
        print(f"[7E] {scale_name} semireal, {n_seeds} seeds, 6 variants")
        print(f"{'='*60}")

        config = _make_config(size_cfg, args.n_gen)
        scale_results = {v: [] for v in list(VANILLA_ALGORITHMS.keys()) + list(SA_ALGORITHM_MAP.keys())}

        for i in range(n_seeds):
            seed = args.seed + 74000 + i * 100
            instance = generate_semireal_instance(config, seed=seed)
            print(f"\n  Seed {i+1}/{n_seeds} (seed={seed})")

            all_fronts = {}

            # Vanilla
            for alg_name, alg_fn in VANILLA_ALGORITHMS.items():
                t0 = time.time()
                result = alg_fn(instance, config, seed=seed, verbose=False)
                elapsed = time.time() - t0
                F = result.pareto_front
                all_fronts[alg_name] = F
                scale_results[alg_name].append({
                    "seed": seed, "elapsed": elapsed,
                    "n_solutions": len(result.pareto_solutions),
                    "pareto_front": F.tolist(),
                })
                print(f"    {alg_name:15s}: {elapsed:.1f}s")

            # SA variants
            surrogate.set_instance(instance)
            for sa_name, base_algo in SA_ALGORITHM_MAP.items():
                sa_config = SAMOEAConfig(
                    base_algorithm=base_algo,
                    correction_fraction=0.5,
                    n_generations=args.n_gen,
                )
                t0 = time.time()
                sa_result = run_samoea(
                    instance, surrogate, sa_config, seed=seed, verbose=False,
                )
                elapsed = time.time() - t0
                F = sa_result.pareto_front
                all_fronts[sa_name] = F
                scale_results[sa_name].append({
                    "seed": seed, "elapsed": elapsed,
                    "eval_savings": sa_result.eval_savings,
                    "n_solutions": len(sa_result.pareto_solutions),
                    "pareto_front": F.tolist(),
                })
                print(f"    {sa_name:15s}: {elapsed:.1f}s, "
                      f"savings={sa_result.eval_savings*100:.1f}%")

            # HV
            ref = _common_ref_point(*all_fronts.values())
            for v in scale_results:
                F = np.array(all_fronts.get(v, np.empty((0, 3))))
                hv = compute_hypervolume(F, ref) if len(F) > 0 else 0.0
                scale_results[v][-1]["hv"] = hv

        all_results[scale_name] = scale_results

        if n_seeds >= 5:
            hv_data = {v: [r["hv"] for r in results] for v, results in scale_results.items()}
            stat = run_statistical_comparison(
                hv_data, metric_name="hypervolume", higher_is_better=True,
            )
            all_statistics[scale_name] = stat
            print(f"\n  [{scale_name}] Key pairwise results:")
            for pw in stat["pairwise"][:6]:  # Show first 6 pairs
                print(f"    {pw['method_a']:15s} vs {pw['method_b']:15s}: "
                      f"p={pw['wilcoxon']['p_value']:.4f}, "
                      f"A12={pw['a12']['a12']:.3f}")

    _save_results(all_results, out_dir / "results.json")
    if all_statistics:
        with open(out_dir / "statistics.json", "w") as f:
            json.dump(all_statistics, f, indent=2, default=str)

    print(f"\n[7E] Results saved to {out_dir}")

# ===================================================================
# Experiment 7F: Statistical summary from 7A results
# ===================================================================

def run_7f(args):
    out_dir = Path(args.output) / "7f_statistics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load 7A results
    results_path = Path(args.output) / "7a_main_comparison" / "results.json"
    if not results_path.exists():
        print("[7F] No 7A results found. Run 7A first.")
        return

    with open(results_path) as f:
        all_results = json.load(f)

    all_variants = list(VANILLA_ALGORITHMS.keys()) + list(SA_ALGORITHM_MAP.keys())

    print(f"\n{'='*60}")
    print("[7F] Statistical Summary from 7A Results")
    print(f"{'='*60}")

    summary = {}

    for scale_name, scale_results in all_results.items():
        print(f"\n--- {scale_name} ---")

        hv_data = {}
        for v in all_variants:
            if v in scale_results:
                hvs = [r["hv"] for r in scale_results[v]]
                hv_data[v] = hvs
                print(f"  {v:15s}: mean={np.mean(hvs):.4e} ± {np.std(hvs):.2e}")

        # Pairwise comparisons
        if len(hv_data) > 1 and len(next(iter(hv_data.values()))) >= 5:
            stat = run_statistical_comparison(
                hv_data, metric_name="hypervolume", higher_is_better=True,
            )

            # Focus on vanilla vs SA pairs
            print(f"\n  Pairwise (vanilla vs SA):")
            pairs_of_interest = [
                ("NSGA-II", "SA-NSGA-II"),
                ("MOEA/D", "SA-MOEA/D"),
                ("NSGA-III", "SA-NSGA-III"),
            ]
            for a, b in pairs_of_interest:
                for pw in stat["pairwise"]:
                    if (pw["method_a"] == a and pw["method_b"] == b) or \
                       (pw["method_a"] == b and pw["method_b"] == a):
                        print(f"    {a:15s} vs {b:15s}: "
                              f"p={pw['wilcoxon']['p_value']:.4f}, "
                              f"A12={pw['a12']['a12']:.3f} ({pw['a12']['magnitude']})")

            summary[scale_name] = stat

    # Compute eval savings summary
    print(f"\n{'='*60}")
    print("Evaluation Savings Summary")
    print(f"{'='*60}")

    for scale_name, scale_results in all_results.items():
        print(f"\n  {scale_name}:")
        for sa_name in SA_ALGORITHM_MAP:
            if sa_name in scale_results:
                savings = [r.get("eval_savings", 0) for r in scale_results[sa_name]]
                print(f"    {sa_name:15s}: {np.mean(savings)*100:.1f}% ± "
                      f"{np.std(savings)*100:.1f}%")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[7F] Summary saved to {out_dir}")

# ===================================================================
# Utilities
# ===================================================================

def _save_results(results: dict, path: Path):
    """Save results dict to JSON, handling numpy types."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)

# ===================================================================
# Main
# ===================================================================

EXPERIMENTS = {
    "7a": run_7a,
    "7b": run_7b,
    "7c": run_7c,
    "7d": run_7d,
    "7e": run_7e,
    "7f": run_7f,
}

def main():
    parser = argparse.ArgumentParser(
        description="Phase 7: 3-Objective Multi-Algorithm Experiments",
    )
    parser.add_argument("--exp", required=True, choices=list(EXPERIMENTS.keys()),
                        help="Experiment to run: 7a-7f")
    parser.add_argument("--model", default=None,
                        help="Path to GNN surrogate checkpoint")
    parser.add_argument("--mlp-medium", default=None,
                        help="MLP checkpoint for medium scale (7d)")
    parser.add_argument("--mlp-large", default=None,
                        help="MLP checkpoint for large scale (7d)")
    parser.add_argument("--output", default="results/phase7",
                        help="Output directory")
    parser.add_argument("--device", default="auto",
                        help="Device for surrogate inference")
    parser.add_argument("--n-gen", type=int, default=200,
                        help="Number of generations (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--scales", default=None,
                        help="Comma-separated scales overriding experiment "
                             "defaults, e.g. 'xlarge' or 'large,xlarge'")
    parser.add_argument("--n-seeds", dest="n_seeds", type=int, default=None,
                        help="Override number of seeds for the experiment")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 1 seed, small, 30 gens")
    args = parser.parse_args()

    if args.quick:
        args.n_gen = 30

    needs_model = ("7a", "7b", "7c", "7d", "7e")
    if args.exp in needs_model and args.model is None:
        parser.error(f"--model is required for experiment {args.exp}")

    print(f"Phase 7 Experiment: {args.exp}")
    print(f"  Generations: {args.n_gen}")
    print(f"  Quick mode: {args.quick}")
    if args.model:
        print(f"  Surrogate: {args.model}")
    print()

    t_start = time.time()
    EXPERIMENTS[args.exp](args)
    total = time.time() - t_start

    print(f"\nTotal time: {total:.1f}s ({total/60:.1f} min)")

if __name__ == "__main__":
    main()
