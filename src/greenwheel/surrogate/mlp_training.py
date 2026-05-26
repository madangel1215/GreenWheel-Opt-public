"""Training loop for WheelingMLP surrogate baseline.

Converts HeteroData graphs to flat tensors and trains a simple MLP.
Shares NormStats with the GNN pipeline for consistent normalization.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from src.greenwheel.surrogate.mlp_model import WheelingMLP
from src.greenwheel.surrogate.training import NormStats, TrainConfig, compute_norm_stats, get_device

def _heterodata_to_flat(data_list: list, norm: NormStats) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert list of HeteroData to flat MLP inputs.

    For each graph, concatenate:
        gen_features_flat (8m) + con_features_flat (11n) + global_feat (6) + W_flat (mn)

    Features are normalized using norm stats.

    Args:
        data_list: List of HeteroData objects (already feature-normalized).
        norm: NormStats for edge feature normalization.

    Returns:
        Tuple of (X, Y) where X is (N, input_dim) and Y is (N, 3).
    """
    all_x = []
    all_y = []

    edge_mean_w = norm.edge_mean[0] if norm.edge_mean else 0.0
    edge_std_w = norm.edge_std[0] if norm.edge_std else 1.0

    for d in data_list:
        gen_feat = d["gen"].x.flatten()   # (8m,)
        con_feat = d["con"].x.flatten()   # (11n,)

        global_feat = d.global_feat
        if global_feat.dim() == 2:
            global_feat = global_feat.squeeze(0)  # (6,)

        # Extract W from edge features (feature 0 is w_ij)
        edge_attr = d["gen", "supplies", "con"].edge_attr  # (m*n, 3)
        w_flat = edge_attr[:, 0]  # (m*n,) - already normalized if features were normalized

        x = torch.cat([gen_feat, con_feat, global_feat, w_flat])
        all_x.append(x)
        all_y.append(d.y)

    return torch.stack(all_x), torch.stack(all_y)

def train_mlp_surrogate(
    train_data: list,
    val_data: list,
    config: TrainConfig | None = None,
    device: str = "auto",
    checkpoint_dir: str = "checkpoints/surrogate_mlp",
    verbose: bool = True,
) -> tuple[WheelingMLP, NormStats, dict]:
    """Train the WheelingMLP surrogate model.

    Reuses the same data pipeline as GNN training (HeteroData format)
    but flattens everything into a single vector per sample.

    Args:
        train_data: List of HeteroData for training (un-normalized).
        val_data: List of HeteroData for validation (un-normalized).
        config: Training hyperparameters.
        device: Device string.
        checkpoint_dir: Directory for saving best model.
        verbose: Print epoch logs.

    Returns:
        Tuple of (trained model, normalization stats, training history).
    """
    if config is None:
        config = TrainConfig()

    dev = get_device(device)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Compute normalization from training set (same as GNN)
    norm = compute_norm_stats(train_data)

    if verbose:
        print(f"Norm: profit mu={norm.mean_profit:.0f} sigma={norm.std_profit:.0f}, "
              f"surplus mu={norm.mean_surplus:.0f} sigma={norm.std_surplus:.0f}, "
              f"shortfall mu={norm.mean_shortfall:.0f} sigma={norm.std_shortfall:.0f}")

    # Normalize input features in-place (same function as GNN training)
    from src.greenwheel.surrogate.training import normalize_features
    normalize_features(train_data, norm)
    normalize_features(val_data, norm)

    # Normalize targets in-place
    for d in train_data:
        d.y = norm.normalize_targets(d.y.unsqueeze(0)).squeeze(0)
    for d in val_data:
        d.y = norm.normalize_targets(d.y.unsqueeze(0)).squeeze(0)

    # Convert to flat tensors
    X_train, Y_train = _heterodata_to_flat(train_data, norm)
    X_val, Y_val = _heterodata_to_flat(val_data, norm)

    input_dim = X_train.shape[1]

    if verbose:
        print(f"MLP input dim: {input_dim}")
        print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # DataLoaders
    train_ds = TensorDataset(X_train, Y_train)
    val_ds = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    # Model
    model = WheelingMLP(input_dim=input_dim, hidden_dim=256, n_layers=3).to(dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_kwargs = {"input_dim": input_dim, "hidden_dim": 256, "n_layers": 3}

    if verbose:
        print(f"WheelingMLP: {n_params:,} parameters, device={dev}")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.min_lr
    )

    loss_fn = nn.MSELoss()

    # Training loop
    history = {
        "train_loss": [], "val_loss": [],
        "val_r2_profit": [], "val_r2_surplus": [], "val_r2_shortfall": [],
        "val_spearman_profit": [], "val_spearman_surplus": [], "val_spearman_shortfall": [],
    }
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(dev)
            y_batch = y_batch.to(dev)

            pred = model(x_batch)
            loss = (
                config.profit_weight * loss_fn(pred[:, 0], y_batch[:, 0])
                + config.surplus_weight * loss_fn(pred[:, 1], y_batch[:, 1])
                + config.shortfall_weight * loss_fn(pred[:, 2], y_batch[:, 2])
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * y_batch.size(0)
            train_n += y_batch.size(0)

        scheduler.step()
        train_loss = train_loss_sum / train_n

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        all_pred = []
        all_true = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(dev)
                y_batch = y_batch.to(dev)

                pred = model(x_batch)
                loss = (
                    config.profit_weight * loss_fn(pred[:, 0], y_batch[:, 0])
                    + config.surplus_weight * loss_fn(pred[:, 1], y_batch[:, 1])
                    + config.shortfall_weight * loss_fn(pred[:, 2], y_batch[:, 2])
                )

                val_loss_sum += loss.item() * y_batch.size(0)
                val_n += y_batch.size(0)
                all_pred.append(pred.cpu())
                all_true.append(y_batch.cpu())

        val_loss = val_loss_sum / val_n
        all_pred = torch.cat(all_pred)
        all_true = torch.cat(all_true)

        # Metrics
        r2_profit = _r2_score(all_true[:, 0], all_pred[:, 0])
        r2_surplus = _r2_score(all_true[:, 1], all_pred[:, 1])
        r2_shortfall = _r2_score(all_true[:, 2], all_pred[:, 2])

        pred_np = all_pred.numpy()
        true_np = all_true.numpy()
        sp_profit = _safe_spearman(true_np[:, 0], pred_np[:, 0])
        sp_surplus = _safe_spearman(true_np[:, 1], pred_np[:, 1])
        sp_shortfall = _safe_spearman(true_np[:, 2], pred_np[:, 2])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_r2_profit"].append(r2_profit)
        history["val_r2_surplus"].append(r2_surplus)
        history["val_r2_shortfall"].append(r2_shortfall)
        history["val_spearman_profit"].append(sp_profit)
        history["val_spearman_surplus"].append(sp_surplus)
        history["val_spearman_shortfall"].append(sp_shortfall)

        elapsed = time.time() - t0
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(
                f"Epoch {epoch:3d}/{config.epochs} | "
                f"train={train_loss:.4f} val={val_loss:.4f} | "
                f"R2=[{r2_profit:.3f}, {r2_surplus:.3f}, {r2_shortfall:.3f}] "
                f"Sp=[{sp_profit:.3f}, {sp_surplus:.3f}, {sp_shortfall:.3f}] | "
                f"{elapsed:.1f}s"
            )

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_kwargs": model_kwargs,
                    "norm_stats": norm.to_dict(),
                    "model_type": "mlp",
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                ckpt_dir / "best.pt",
            )
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (best={best_epoch})")
            break

    # Load best model
    ckpt = torch.load(ckpt_dir / "best.pt", weights_only=False, map_location=dev)
    model.load_state_dict(ckpt["model_state"])

    # Save training history
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    if verbose:
        print(f"\nBest model at epoch {best_epoch}, val_loss={best_val_loss:.4f}")

    return model, norm, history

def load_trained_mlp(
    checkpoint_path: str | Path,
    device: str = "auto",
) -> tuple[WheelingMLP, NormStats]:
    """Load a trained MLP model from checkpoint."""
    dev = get_device(device)
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=dev)

    model = WheelingMLP(**ckpt["model_kwargs"]).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    norm = NormStats.from_dict(ckpt["norm_stats"])

    return model, norm

def _r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    if ss_tot < 1e-10:
        return 0.0
    return float(1 - ss_res / ss_tot)

def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_pred) < 1e-10 or np.std(y_true) < 1e-10:
        return 0.0
    r, _ = spearmanr(y_true, y_pred)
    return float(r) if not np.isnan(r) else 0.0
