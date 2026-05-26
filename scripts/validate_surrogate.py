#!/usr/bin/env python
"""Validate the trained WheelingGNN surrogate model.

Usage:
    # Phase 4A: single-instance validation
    python scripts/validate_surrogate.py \
        --model checkpoints/surrogate/best.pt \
        --data data/surrogate/phase4a \
        --output results/surrogate_validation

    # With training history plots
    python scripts/validate_surrogate.py \
        --model checkpoints/surrogate/best.pt \
        --data data/surrogate/phase4a \
        --history checkpoints/surrogate/history.json \
        --output results/surrogate_validation
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.greenwheel.surrogate.training import load_trained_model
from src.greenwheel.surrogate.validation import (
    evaluate_model,
    plot_residuals,
    plot_scatter,
    plot_training_history,
    print_metrics,
)

def main():
    parser = argparse.ArgumentParser(description="Validate WheelingGNN surrogate")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--data", required=True, help="Data directory with test.pkl")
    parser.add_argument("--history", default=None, help="Path to training history JSON")
    parser.add_argument("--output", default="results/surrogate_validation",
                        help="Output directory for figures and metrics")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--config", default="configs/surrogate.yaml",
                        help="Surrogate config for gate thresholds")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    model, norm = load_trained_model(args.model, device=args.device)

    # Load test data
    data_dir = Path(args.data)
    with open(data_dir / "test.pkl", "rb") as f:
        test_data = pickle.load(f)
    print(f"Test data: {len(test_data)} samples")

    # Evaluate
    metrics = evaluate_model(model, test_data, norm, device=args.device)
    print_metrics(metrics)

    # Generate plots
    out_dir = Path(args.output)
    plot_scatter(metrics, output_dir=out_dir)
    plot_residuals(metrics, output_dir=out_dir)
    print(f"\nFigures saved to {out_dir}")

    # Training history plots
    if args.history:
        with open(args.history) as f:
            history = json.load(f)
        plot_training_history(history, output_dir=out_dir)
        print(f"Training history plot saved to {out_dir}")

    # Kill-switch gate check
    with open(args.config) as f:
        surr_cfg = yaml.safe_load(f)

    gates = surr_cfg.get("validation", {})
    phase4a_gate = gates.get("phase4a_spearman_gate", 0.9)

    sp_profit = metrics["profit"]["spearman"]
    sp_surplus = metrics["surplus"]["spearman"]
    sp_shortfall = metrics["shortfall"]["spearman"]

    print(f"\n{'='*60}")
    print(f"GATE CHECK (Spearman > {phase4a_gate})")
    print(f"  Profit Spearman:    {sp_profit:.4f}  {'PASS' if sp_profit > phase4a_gate else 'FAIL'}")
    print(f"  Surplus Spearman:   {sp_surplus:.4f}  {'PASS' if sp_surplus > phase4a_gate else 'FAIL'}")
    print(f"  Shortfall Spearman: {sp_shortfall:.4f}  {'PASS' if sp_shortfall > phase4a_gate else 'FAIL'}")

    all_pass = sp_profit > phase4a_gate and sp_surplus > phase4a_gate and sp_shortfall > phase4a_gate
    if all_pass:
        print(f"\n  RESULT: PASS - All objectives meet threshold")
    else:
        print(f"\n  RESULT: FAIL - Kill switch triggered. Debug architecture.")
    print(f"{'='*60}")

    # Save metrics JSON
    metrics_save = {k: v for k, v in metrics.items() if k != "predictions"}
    metrics_save["gate"] = {
        "threshold": phase4a_gate,
        "profit_pass": sp_profit > phase4a_gate,
        "surplus_pass": sp_surplus > phase4a_gate,
        "shortfall_pass": sp_shortfall > phase4a_gate,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_save, f, indent=2)

if __name__ == "__main__":
    main()
