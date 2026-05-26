#!/usr/bin/env python
"""Train surrogate models (GNN or MLP) for wheeling allocation.

Usage:
    # GNN (default)
    python scripts/train_surrogate.py --data data/surrogate/phase4b

    # MLP baseline
    python scripts/train_surrogate.py --data data/surrogate/phase4b --model-type mlp \
        --checkpoint-dir checkpoints/surrogate_mlp_medium

    # MLP for specific scale (filter by input dim)
    python scripts/train_surrogate.py --data data/surrogate/phase4b --model-type mlp \
        --scale medium --checkpoint-dir checkpoints/surrogate_mlp_medium
"""

import argparse
import pickle
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.greenwheel.surrogate.training import TrainConfig, train_surrogate

def _filter_by_scale(data_list: list, scale: str) -> list:
    """Filter HeteroData samples by problem scale.

    Args:
        data_list: List of HeteroData objects.
        scale: One of 'small', 'medium', 'large', 'xlarge'.

    Returns:
        Filtered list matching the specified scale.
    """
    scale_sizes = {
        "small": (5, 10),
        "medium": (10, 20),
        "large": (20, 50),
        "xlarge": (50, 100),
        "xxlarge": (50, 200),
    }
    if scale not in scale_sizes:
        raise ValueError(f"Unknown scale: {scale}. Choose from {list(scale_sizes.keys())}")

    target_m, target_n = scale_sizes[scale]
    filtered = []
    for d in data_list:
        m = d["gen"].num_nodes
        n = d["con"].num_nodes
        if m == target_m and n == target_n:
            filtered.append(d)
    return filtered

def main():
    parser = argparse.ArgumentParser(description="Train surrogate model (GNN or MLP)")
    parser.add_argument("--config", default="configs/surrogate.yaml",
                        help="Surrogate config file")
    parser.add_argument("--data", required=True,
                        help="Data directory with train.pkl, val.pkl, test.pkl")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, mps, cuda, cpu")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override max epochs")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Override checkpoint directory")
    parser.add_argument("--model-type", choices=["gnn", "mlp"], default="gnn",
                        help="Model type: gnn (default) or mlp")
    parser.add_argument("--scale", default=None,
                        choices=["small", "medium", "large", "xlarge", "xxlarge"],
                        help="Filter data to specific scale (required for MLP)")
    parser.add_argument("--early-stop", default=None,
                        choices=["val_loss", "spearman"],
                        help="Early stopping metric (default: from config)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        surr_cfg = yaml.safe_load(f)

    # Load data
    data_dir = Path(args.data)
    print(f"Loading data from {data_dir}...")

    with open(data_dir / "train.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open(data_dir / "val.pkl", "rb") as f:
        val_data = pickle.load(f)

    print(f"  Loaded: Train={len(train_data)}, Val={len(val_data)}")

    # Filter by scale if specified
    if args.scale:
        train_data = _filter_by_scale(train_data, args.scale)
        val_data = _filter_by_scale(val_data, args.scale)
        print(f"  Filtered to {args.scale}: Train={len(train_data)}, Val={len(val_data)}")
        if len(train_data) == 0:
            print("ERROR: No training samples match the specified scale!")
            sys.exit(1)

    # Build TrainConfig
    tc = surr_cfg.get("training", {})
    train_cfg = TrainConfig(
        learning_rate=tc.get("learning_rate", 1e-3),
        weight_decay=tc.get("weight_decay", 1e-5),
        epochs=args.epochs or tc.get("epochs", 300),
        patience=tc.get("patience", 30),
        batch_size=tc.get("batch_size", 64),
        min_lr=tc.get("min_lr", 1e-6),
        profit_weight=tc.get("profit_weight", 0.34),
        surplus_weight=tc.get("surplus_weight", 0.33),
        shortfall_weight=tc.get("shortfall_weight", 0.33),
        early_stop_metric=args.early_stop or tc.get("early_stop_metric", "val_loss"),
    )

    device = args.device or surr_cfg.get("device", "auto")

    if args.model_type == "gnn":
        # GNN training (original path)
        model_kwargs = surr_cfg.get("model", {})
        ckpt_dir = args.checkpoint_dir or surr_cfg.get("checkpoint_dir", "checkpoints/surrogate")

        model, norm, history = train_surrogate(
            train_data=train_data,
            val_data=val_data,
            config=train_cfg,
            model_kwargs=model_kwargs,
            device=device,
            checkpoint_dir=ckpt_dir,
            verbose=True,
        )
        print(f"\nGNN training complete. Best model saved to {ckpt_dir}/best.pt")

    else:
        # MLP training
        from src.greenwheel.surrogate.mlp_training import train_mlp_surrogate

        if args.scale is None:
            print("WARNING: MLP input dim depends on problem size. "
                  "Training on mixed scales will fail. Use --scale to filter.")

        ckpt_dir = args.checkpoint_dir or f"checkpoints/surrogate_mlp_{args.scale or 'mixed'}"

        model, norm, history = train_mlp_surrogate(
            train_data=train_data,
            val_data=val_data,
            config=train_cfg,
            device=device,
            checkpoint_dir=ckpt_dir,
            verbose=True,
        )
        print(f"\nMLP training complete. Best model saved to {ckpt_dir}/best.pt")

    print(f"Normalization: profit mu={norm.mean_profit:.0f} sigma={norm.std_profit:.0f}, "
          f"surplus mu={norm.mean_surplus:.0f} sigma={norm.std_surplus:.0f}, "
          f"shortfall mu={norm.mean_shortfall:.0f} sigma={norm.std_shortfall:.0f}")

if __name__ == "__main__":
    main()
