#!/usr/bin/env python
"""Generate training data for GNN surrogate.

Usage:
    # Phase 4A: Single-instance proof-of-concept
    python scripts/generate_training_data.py --phase 4a --output data/surrogate/phase4a

    # Phase 4B: Full multi-scale dataset (instance-stratified split)
    python scripts/generate_training_data.py --phase 4b --output data/surrogate/phase4b
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.greenwheel.config import WheelingConfig
from src.greenwheel.data.synthetic import generate_instance, generate_semireal_instance
from src.greenwheel.surrogate.dataset import (
    generate_lhs_samples,
    generate_nsga2_snapshots,
    generate_pareto_perturbations,
)
from src.greenwheel.surrogate.graph import instance_to_heterodata

def generate_for_instance(
    config: WheelingConfig,
    seed: int,
    n_lhs: int,
    nsga2_pop_size: int,
    nsga2_gens: int,
    snapshot_gens: list[int],
    n_pareto_perturb: int,
    verbose: bool = True,
    data_source: str = "synthetic",
) -> list:
    """Generate training samples for one instance. Returns list of HeteroData."""
    if data_source == "semireal":
        instance = generate_semireal_instance(config, seed=seed)
    else:
        instance = generate_instance(config, seed=seed)
    m, n = instance.m, instance.n
    if verbose:
        print(f"  Instance seed={seed}: {m}×{n}, "
              f"gen={instance.total_generation/1e6:.1f}MWh, "
              f"dem={instance.total_demand/1e6:.1f}MWh")

    samples = []

    # 1. LHS random samples
    t0 = time.time()
    lhs = generate_lhs_samples(instance, n_lhs, seed=seed)
    if verbose:
        print(f"    LHS: {len(lhs)} samples in {time.time()-t0:.1f}s")
    samples.extend(lhs)

    # 2. NSGA-II snapshots
    t0 = time.time()
    nsga2 = generate_nsga2_snapshots(
        instance,
        pop_size=nsga2_pop_size,
        n_generations=nsga2_gens,
        snapshot_gens=snapshot_gens,
        seed=seed,
    )
    if verbose:
        print(f"    NSGA-II: {len(nsga2)} samples in {time.time()-t0:.1f}s")
    samples.extend(nsga2)

    # 3. Pareto perturbations
    if nsga2:
        t0 = time.time()
        pareto_candidates = nsga2[-nsga2_pop_size:][:20]
        perturbations = generate_pareto_perturbations(
            pareto_candidates,
            instance,
            n_perturbations_per=n_pareto_perturb,
            seed=seed + 5000,
        )
        if verbose:
            print(f"    Perturb: {len(perturbations)} samples in {time.time()-t0:.1f}s")
        samples.extend(perturbations)

    # Convert to HeteroData
    graphs = []
    for W, profit, surplus, shortfall in samples:
        targets = np.array([profit, surplus, shortfall], dtype=np.float32)
        graph = instance_to_heterodata(instance, W, targets=targets)
        graphs.append(graph)

    return graphs

def main():
    parser = argparse.ArgumentParser(description="Generate GNN surrogate training data")
    parser.add_argument("--phase", choices=["4a", "4b", "6f"], default="4a",
                        help="Phase 4A (single-instance), 4B (multi-scale), or 6F (with xlarge)")
    parser.add_argument("--config", default="configs/surrogate.yaml",
                        help="Surrogate config file")
    parser.add_argument("--n-generators", type=int, default=None)
    parser.add_argument("--n-consumers", type=int, default=None)
    parser.add_argument("--n-instances", type=int, default=None)
    parser.add_argument("--samples-per-instance", type=int, default=None)
    parser.add_argument("--output", type=str, default="data/surrogate/phase4a")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-source", choices=["synthetic", "semireal"],
                        default="synthetic",
                        help="Data source: synthetic (default) or semireal (Taipower)")
    args = parser.parse_args()

    # Load surrogate config
    with open(args.config) as f:
        surr_cfg = yaml.safe_load(f)

    sampling = surr_cfg.get("sampling", {})
    nsga2_pop_size = sampling.get("nsga2_pop_size", 50)
    nsga2_gens = sampling.get("nsga2_generations", 200)
    snapshot_gens = sampling.get("snapshot_generations", [0, 50, 100, 150, 200])
    lhs_frac = sampling.get("lhs_fraction", 0.6)
    perturb_per = sampling.get("perturbation_per_solution", 5)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_key = {"4a": "phase4a", "4b": "phase4b", "6f": "phase6f"}[args.phase]
    split_ratio = surr_cfg.get("data", {}).get(
        phase_key, {}
    ).get("split_ratio", [0.8, 0.1, 0.1])

    if args.phase == "4a":
        # Phase 4A: single medium instance - sample-level split is fine
        data_cfg = surr_cfg.get("data", {}).get("phase4a", {})
        n_gen = args.n_generators or data_cfg.get("n_generators", 10)
        n_con = args.n_consumers or data_cfg.get("n_consumers", 20)
        n_inst = args.n_instances or data_cfg.get("n_instances", 1)
        spi = args.samples_per_instance or data_cfg.get("samples_per_instance", 5000)
        n_lhs = int(spi * lhs_frac)

        print(f"Phase 4A: {n_gen}×{n_con}, {n_inst} instance(s), ~{spi} samples each")

        config = WheelingConfig()
        config.n_generators = n_gen
        config.n_consumers = n_con

        all_graphs = []
        for i in range(n_inst):
            seed = args.seed + i * 100
            graphs = generate_for_instance(
                config, seed, n_lhs, nsga2_pop_size, nsga2_gens,
                snapshot_gens, perturb_per,
                data_source=args.data_source,
            )
            all_graphs.extend(graphs)

        # Sample-level random split
        rng = np.random.default_rng(args.seed)
        indices = rng.permutation(len(all_graphs))
        n_train = int(len(all_graphs) * split_ratio[0])
        n_val = int(len(all_graphs) * split_ratio[1])

        train_data = [all_graphs[i] for i in indices[:n_train]]
        val_data = [all_graphs[i] for i in indices[n_train:n_train + n_val]]
        test_data = [all_graphs[i] for i in indices[n_train + n_val:]]

        meta = {
            "phase": "4a", "split_strategy": "sample_level",
            "scales": [{"n_gen": n_gen, "n_con": n_con, "n_instances": n_inst}],
        }

    elif args.phase in ("4b", "6f"):
        # Phase 4B / 6F: INSTANCE-STRATIFIED SPLIT
        # No instance leaks: held-out instances are never seen during training
        scales = surr_cfg.get("data", {}).get(phase_key, {}).get("scales", {})

        # Collect per-instance graph lists with metadata
        instance_groups = []  # list of (scale_name, seed, graphs)
        total_t0 = time.time()

        for scale_name, scale_cfg in scales.items():
            n_gen = scale_cfg["n_generators"]
            n_con = scale_cfg["n_consumers"]
            n_inst = scale_cfg["n_instances"]
            spi = scale_cfg["samples_per_instance"]
            n_lhs = int(spi * lhs_frac)

            print(f"\nScale '{scale_name}': {n_gen}×{n_con}, {n_inst} instances, ~{spi} samples each")

            config = WheelingConfig()
            config.n_generators = n_gen
            config.n_consumers = n_con

            for i in range(n_inst):
                seed = args.seed + i * 100
                graphs = generate_for_instance(
                    config, seed, n_lhs, nsga2_pop_size, nsga2_gens,
                    snapshot_gens, perturb_per,
                    data_source=args.data_source,
                )
                instance_groups.append((scale_name, seed, graphs))

        total_time = time.time() - total_t0
        total_samples = sum(len(g) for _, _, g in instance_groups)
        print(f"\nTotal: {len(instance_groups)} instances, {total_samples} samples, {total_time:.0f}s")

        # Instance-stratified split: split instances per scale
        rng = np.random.default_rng(args.seed)
        train_data, val_data, test_data = [], [], []
        split_info = {}

        for scale_name in scales:
            scale_instances = [(s, seed, g) for s, seed, g in instance_groups if s == scale_name]
            n_scale = len(scale_instances)
            perm = rng.permutation(n_scale)

            n_tr = max(1, int(n_scale * split_ratio[0]))
            n_va = max(1, int(n_scale * split_ratio[1]))

            tr_idx = perm[:n_tr]
            va_idx = perm[n_tr:n_tr + n_va]
            te_idx = perm[n_tr + n_va:]

            tr_seeds, va_seeds, te_seeds = [], [], []
            for idx in tr_idx:
                train_data.extend(scale_instances[idx][2])
                tr_seeds.append(int(scale_instances[idx][1]))
            for idx in va_idx:
                val_data.extend(scale_instances[idx][2])
                va_seeds.append(int(scale_instances[idx][1]))
            for idx in te_idx:
                test_data.extend(scale_instances[idx][2])
                te_seeds.append(int(scale_instances[idx][1]))

            split_info[scale_name] = {
                "total_instances": n_scale,
                "train_instances": len(tr_idx),
                "val_instances": len(va_idx),
                "test_instances": len(te_idx),
                "train_seeds": tr_seeds,
                "val_seeds": va_seeds,
                "test_seeds": te_seeds,
            }
            print(f"  {scale_name}: train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)} instances")

        meta = {
            "phase": args.phase, "split_strategy": "instance_stratified",
            "split_info": split_info,
            "total_time_seconds": total_time,
        }

    print(f"\nSplit: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    # Save
    with open(out_dir / "train.pkl", "wb") as f:
        pickle.dump(train_data, f)
    with open(out_dir / "val.pkl", "wb") as f:
        pickle.dump(val_data, f)
    with open(out_dir / "test.pkl", "wb") as f:
        pickle.dump(test_data, f)

    meta.update({
        "total_samples": len(train_data) + len(val_data) + len(test_data),
        "n_train": len(train_data),
        "n_val": len(val_data),
        "n_test": len(test_data),
        "seed": args.seed,
    })
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved to {out_dir}")

if __name__ == "__main__":
    main()
