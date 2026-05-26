"""Validation and accuracy metrics for GNN surrogate.

Computes R², RMSE, MAE, Spearman, Kendall's tau per objective,
and generates diagnostic plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr
from torch_geometric.loader import DataLoader

from src.greenwheel.surrogate.model import WheelingGNN
from src.greenwheel.surrogate.training import (
    NormStats, _forward_batch, get_device, normalize_features,
)

def evaluate_model(
    model: WheelingGNN,
    test_data: list,
    norm: NormStats,
    device: str = "auto",
    batch_size: int = 64,
) -> dict:
    """Evaluate model on test data and return comprehensive metrics.

    Args:
        model: Trained WheelingGNN.
        test_data: List of HeteroData with raw (unnormalized) targets.
        norm: Normalization stats used during training.
        device: Device string.
        batch_size: Batch size for evaluation.

    Returns:
        Dict with metrics per objective and overall.
    """
    dev = get_device(device)
    model = model.to(dev)
    model.eval()

    # Clone to avoid modifying original data
    test_normed = [d.clone() for d in test_data]

    # Normalize input features
    normalize_features(test_normed, norm)

    # Normalize targets
    for d in test_normed:
        d.y = norm.normalize_targets(d.y.unsqueeze(0)).squeeze(0)

    loader = DataLoader(test_normed, batch_size=batch_size, shuffle=False)

    n_obj = norm.n_outputs
    all_pred_norm = []
    all_true_norm = []

    with torch.no_grad():
        for batch in loader:
            pred = _forward_batch(model, batch, dev)
            target = batch.y.to(dev).view(-1, n_obj)
            all_pred_norm.append(pred.cpu())
            all_true_norm.append(target.cpu())

    pred_norm = torch.cat(all_pred_norm).numpy()
    true_norm = torch.cat(all_true_norm).numpy()

    # Denormalize to raw scale
    pred_raw = np.column_stack([
        pred_norm[:, 0] * (norm.std_profit + 1e-8) + norm.mean_profit,
        pred_norm[:, 1] * (norm.std_surplus + 1e-8) + norm.mean_surplus,
        pred_norm[:, 2] * (norm.std_shortfall + 1e-8) + norm.mean_shortfall,
    ])
    true_raw = np.column_stack([
        true_norm[:, 0] * (norm.std_profit + 1e-8) + norm.mean_profit,
        true_norm[:, 1] * (norm.std_surplus + 1e-8) + norm.mean_surplus,
        true_norm[:, 2] * (norm.std_shortfall + 1e-8) + norm.mean_shortfall,
    ])

    metrics = {}
    for i, name in enumerate(["profit", "surplus", "shortfall"]):
        y_true = true_raw[:, i]
        y_pred = pred_raw[:, i]

        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-8)

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        sp, sp_p = spearmanr(y_true, y_pred)
        kt, kt_p = kendalltau(y_true, y_pred)

        metrics[name] = {
            "r2": float(r2),
            "rmse": float(rmse),
            "mae": float(mae),
            "spearman": float(sp),
            "spearman_p": float(sp_p),
            "kendall_tau": float(kt),
            "kendall_p": float(kt_p),
            "n_samples": len(y_true),
        }

    metrics["predictions"] = {
        "pred_raw": pred_raw,
        "true_raw": true_raw,
    }

    return metrics

def print_metrics(metrics: dict) -> None:
    """Print formatted metrics table."""
    print("\n" + "=" * 60)
    print("GNN Surrogate Validation Metrics")
    print("=" * 60)
    for name in ["profit", "surplus", "shortfall"]:
        m = metrics[name]
        print(f"\n  {name.upper()}:")
        print(f"    R²            = {m['r2']:.4f}")
        print(f"    RMSE          = {m['rmse']:.2f}")
        print(f"    MAE           = {m['mae']:.2f}")
        print(f"    Spearman ρ    = {m['spearman']:.4f} (p={m['spearman_p']:.2e})")
        print(f"    Kendall τ     = {m['kendall_tau']:.4f} (p={m['kendall_p']:.2e})")
        print(f"    N samples     = {m['n_samples']}")
    print("=" * 60)

def plot_scatter(
    metrics: dict,
    output_dir: str | Path = "results/surrogate_validation",
) -> None:
    """Generate scatter plots of predicted vs actual for both objectives.

    Args:
        metrics: Output from evaluate_model().
        output_dir: Directory to save figures.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pred = metrics["predictions"]["pred_raw"]
    true = metrics["predictions"]["true_raw"]

    obj_info = [("Profit", "NTD"), ("Surplus", "kWh"), ("Shortfall", "kWh")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (name, unit) in enumerate(obj_info):
        ax = axes[i]
        m = metrics[name.lower()]

        ax.scatter(true[:, i], pred[:, i], alpha=0.3, s=10, c="steelblue")

        # Perfect prediction line
        lo = min(true[:, i].min(), pred[:, i].min())
        hi = max(true[:, i].max(), pred[:, i].max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="Perfect")

        ax.set_xlabel(f"True {name} ({unit})")
        ax.set_ylabel(f"Predicted {name} ({unit})")
        ax.set_title(
            f"{name}: R²={m['r2']:.3f}, Sp={m['spearman']:.3f}"
        )
        ax.legend()

    plt.tight_layout()
    plt.savefig(out / "scatter_pred_vs_actual.png", dpi=150)
    plt.close()

def plot_residuals(
    metrics: dict,
    output_dir: str | Path = "results/surrogate_validation",
) -> None:
    """Generate residual plots for both objectives."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pred = metrics["predictions"]["pred_raw"]
    true = metrics["predictions"]["true_raw"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, name in enumerate(["Profit", "Surplus", "Shortfall"]):
        ax = axes[i]
        residuals = pred[:, i] - true[:, i]

        ax.scatter(true[:, i], residuals, alpha=0.3, s=10, c="steelblue")
        ax.axhline(0, color="r", linestyle="--", linewidth=1)
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel("Residual (Pred - True)")
        ax.set_title(f"{name} Residuals")

    plt.tight_layout()
    plt.savefig(out / "residual_plots.png", dpi=150)
    plt.close()

def plot_training_history(
    history: dict,
    output_dir: str | Path = "results/surrogate_validation",
) -> None:
    """Plot training loss and validation metrics over epochs."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].set_yscale("log")

    # R²
    axes[1].plot(history["val_r2_profit"], label="Profit")
    axes[1].plot(history["val_r2_surplus"], label="Surplus")
    axes[1].plot(history["val_r2_shortfall"], label="Shortfall")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("R²")
    axes[1].set_title("Validation R²")
    axes[1].legend()

    # Spearman
    axes[2].plot(history["val_spearman_profit"], label="Profit")
    axes[2].plot(history["val_spearman_surplus"], label="Surplus")
    axes[2].plot(history["val_spearman_shortfall"], label="Shortfall")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Spearman ρ")
    axes[2].set_title("Validation Spearman Correlation")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(out / "training_history.png", dpi=150)
    plt.close()
