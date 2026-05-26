"""Training loop for WheelingGNN surrogate.

Handles:
- Z-score normalization for targets AND input features
- MPS/CUDA/CPU device handling
- Early stopping on validation loss
- Model checkpointing
- Per-epoch logging of loss, R², and Spearman correlation
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from src.greenwheel.surrogate.model import WheelingGNN

@dataclass
class TrainConfig:
    """Training hyperparameters."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 300
    patience: int = 30
    batch_size: int = 64
    min_lr: float = 1e-6
    profit_weight: float = 0.34
    surplus_weight: float = 0.33
    shortfall_weight: float = 0.33
    early_stop_metric: str = "val_loss"  # "val_loss" or "spearman"

@dataclass
class NormStats:
    """Z-score normalization statistics for targets and input features."""

    # Target normalization (3 objectives)
    mean_profit: float = 0.0
    std_profit: float = 1.0
    mean_surplus: float = 0.0
    std_surplus: float = 1.0
    mean_shortfall: float = 0.0
    std_shortfall: float = 1.0

    # Input feature normalization (stored as lists for JSON serialization)
    gen_mean: list = field(default_factory=list)
    gen_std: list = field(default_factory=list)
    con_mean: list = field(default_factory=list)
    con_std: list = field(default_factory=list)
    edge_mean: list = field(default_factory=list)
    edge_std: list = field(default_factory=list)
    global_mean: list = field(default_factory=list)
    global_std: list = field(default_factory=list)

    @property
    def n_outputs(self) -> int:
        return 3

    def normalize_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Normalize raw targets to z-scores. Targets shape: (B, 3)."""
        normed = targets.clone()
        normed[:, 0] = (targets[:, 0] - self.mean_profit) / (self.std_profit + 1e-8)
        normed[:, 1] = (targets[:, 1] - self.mean_surplus) / (self.std_surplus + 1e-8)
        normed[:, 2] = (targets[:, 2] - self.mean_shortfall) / (self.std_shortfall + 1e-8)
        return normed

    def denormalize(self, normed: torch.Tensor) -> torch.Tensor:
        """Convert z-scores back to raw values. Shape: (B, 3)."""
        raw = normed.clone()
        raw[:, 0] = normed[:, 0] * (self.std_profit + 1e-8) + self.mean_profit
        raw[:, 1] = normed[:, 1] * (self.std_surplus + 1e-8) + self.mean_surplus
        raw[:, 2] = normed[:, 2] * (self.std_shortfall + 1e-8) + self.mean_shortfall
        return raw

    def to_dict(self) -> dict:
        return {
            "mean_profit": self.mean_profit,
            "std_profit": self.std_profit,
            "mean_surplus": self.mean_surplus,
            "std_surplus": self.std_surplus,
            "mean_shortfall": self.mean_shortfall,
            "std_shortfall": self.std_shortfall,
            "gen_mean": self.gen_mean,
            "gen_std": self.gen_std,
            "con_mean": self.con_mean,
            "con_std": self.con_std,
            "edge_mean": self.edge_mean,
            "edge_std": self.edge_std,
            "global_mean": self.global_mean,
            "global_std": self.global_std,
        }

    @staticmethod
    def from_dict(d: dict) -> "NormStats":
        # Handle backward compatibility with old checkpoints that have tmi fields
        if "mean_tmi" in d and "mean_surplus" not in d:
            d = d.copy()
            d.pop("mean_tmi", None)
            d.pop("std_tmi", None)
            d.setdefault("mean_surplus", 0.0)
            d.setdefault("std_surplus", 1.0)
            d.setdefault("mean_shortfall", 0.0)
            d.setdefault("std_shortfall", 1.0)
        return NormStats(**d)

def compute_norm_stats(data_list: list) -> NormStats:
    """Compute Z-score normalization statistics from training data."""
    targets = torch.stack([d.y for d in data_list])  # (N, 3)

    # Collect all features
    all_gen = torch.cat([d["gen"].x for d in data_list], dim=0)
    all_con = torch.cat([d["con"].x for d in data_list], dim=0)
    all_edge = torch.cat([d["gen", "supplies", "con"].edge_attr for d in data_list], dim=0)
    all_global = torch.cat([d.global_feat for d in data_list], dim=0)

    def _stats(t):
        m = t.mean(dim=0)
        s = t.std(dim=0)
        s[s < 1e-6] = 1.0  # Prevent division by zero for constant features
        return m.tolist(), s.tolist()

    gen_m, gen_s = _stats(all_gen)
    con_m, con_s = _stats(all_con)
    edge_m, edge_s = _stats(all_edge)
    global_m, global_s = _stats(all_global)

    return NormStats(
        mean_profit=float(targets[:, 0].mean()),
        std_profit=float(targets[:, 0].std()),
        mean_surplus=float(targets[:, 1].mean()),
        std_surplus=float(targets[:, 1].std()),
        mean_shortfall=float(targets[:, 2].mean()),
        std_shortfall=float(targets[:, 2].std()),
        gen_mean=gen_m, gen_std=gen_s,
        con_mean=con_m, con_std=con_s,
        edge_mean=edge_m, edge_std=edge_s,
        global_mean=global_m, global_std=global_s,
    )

def normalize_features(data_list: list, norm: NormStats) -> None:
    """Normalize all input features in-place using computed statistics."""
    gen_m = torch.tensor(norm.gen_mean, dtype=torch.float32)
    gen_s = torch.tensor(norm.gen_std, dtype=torch.float32)
    con_m = torch.tensor(norm.con_mean, dtype=torch.float32)
    con_s = torch.tensor(norm.con_std, dtype=torch.float32)
    edge_m = torch.tensor(norm.edge_mean, dtype=torch.float32)
    edge_s = torch.tensor(norm.edge_std, dtype=torch.float32)
    global_m = torch.tensor(norm.global_mean, dtype=torch.float32)
    global_s = torch.tensor(norm.global_std, dtype=torch.float32)

    for d in data_list:
        d["gen"].x = (d["gen"].x - gen_m) / gen_s
        d["con"].x = (d["con"].x - con_m) / con_s
        d["gen", "supplies", "con"].edge_attr = (
            d["gen", "supplies", "con"].edge_attr - edge_m
        ) / edge_s
        d["con", "demands", "gen"].edge_attr = (
            d["con", "demands", "gen"].edge_attr - edge_m
        ) / edge_s
        d.global_feat = (d.global_feat - global_m) / global_s

def get_device(requested: str = "auto") -> torch.device:
    """Get the best available device."""
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)

def _forward_batch(model: WheelingGNN, batch, device: torch.device) -> torch.Tensor:
    """Run forward pass on a batch, handling HeteroData format."""
    batch = batch.to(device)

    x_dict = {"gen": batch["gen"].x, "con": batch["con"].x}

    edge_index_dict = {}
    edge_attr_dict = {}
    for key in batch.edge_types:
        edge_index_dict[key] = batch[key].edge_index
        edge_attr_dict[key] = batch[key].edge_attr

    batch_dict = {"gen": batch["gen"].batch, "con": batch["con"].batch}

    global_feat = batch.global_feat if hasattr(batch, "global_feat") else None

    return model(x_dict, edge_index_dict, edge_attr_dict, batch_dict, global_feat)

def train_surrogate(
    train_data: list,
    val_data: list,
    config: TrainConfig | None = None,
    model_kwargs: dict | None = None,
    device: str = "auto",
    checkpoint_dir: str = "checkpoints",
    verbose: bool = True,
) -> tuple[WheelingGNN, NormStats, dict]:
    """Train the WheelingGNN surrogate model.

    Args:
        train_data: List of HeteroData for training.
        val_data: List of HeteroData for validation.
        config: Training hyperparameters.
        model_kwargs: WheelingGNN constructor kwargs.
        device: Device string ('auto', 'mps', 'cuda', 'cpu').
        checkpoint_dir: Directory for saving best model.
        verbose: Print epoch logs.

    Returns:
        Tuple of (trained model, normalization stats, training history).
    """
    if config is None:
        config = TrainConfig()
    if model_kwargs is None:
        model_kwargs = {}

    dev = get_device(device)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Compute normalization from training set
    norm = compute_norm_stats(train_data)

    if verbose:
        print(f"Norm: profit μ={norm.mean_profit:.0f} σ={norm.std_profit:.0f}, "
              f"surplus μ={norm.mean_surplus:.0f} σ={norm.std_surplus:.0f}, "
              f"shortfall μ={norm.mean_shortfall:.0f} σ={norm.std_shortfall:.0f}")

    # Normalize ALL features in-place
    normalize_features(train_data, norm)
    normalize_features(val_data, norm)

    # Normalize targets in-place
    for d in train_data:
        d.y = norm.normalize_targets(d.y.unsqueeze(0)).squeeze(0)
    for d in val_data:
        d.y = norm.normalize_targets(d.y.unsqueeze(0)).squeeze(0)

    # DataLoaders
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)

    # Model
    model = WheelingGNN(**model_kwargs).to(dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"WheelingGNN: {n_params:,} parameters, device={dev}")

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
    best_metric = float("inf") if config.early_stop_metric == "val_loss" else float("-inf")
    patience_counter = 0
    best_epoch = 0
    use_spearman_stop = config.early_stop_metric == "spearman"

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for batch in train_loader:
            pred = _forward_batch(model, batch, dev)
            target = batch.y.to(dev).view(-1, 3)

            loss = (
                config.profit_weight * loss_fn(pred[:, 0], target[:, 0])
                + config.surplus_weight * loss_fn(pred[:, 1], target[:, 1])
                + config.shortfall_weight * loss_fn(pred[:, 2], target[:, 2])
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * target.size(0)
            train_n += target.size(0)

        scheduler.step()
        train_loss = train_loss_sum / train_n

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        all_pred = []
        all_true = []

        with torch.no_grad():
            for batch in val_loader:
                pred = _forward_batch(model, batch, dev)
                target = batch.y.to(dev).view(-1, 3)

                loss = (
                    config.profit_weight * loss_fn(pred[:, 0], target[:, 0])
                    + config.surplus_weight * loss_fn(pred[:, 1], target[:, 1])
                    + config.shortfall_weight * loss_fn(pred[:, 2], target[:, 2])
                )

                val_loss_sum += loss.item() * target.size(0)
                val_n += target.size(0)
                all_pred.append(pred.cpu())
                all_true.append(target.cpu())

        val_loss = val_loss_sum / val_n
        all_pred = torch.cat(all_pred)
        all_true = torch.cat(all_true)

        # Metrics (handle NaN from constant predictions)
        r2_profit = _r2_score(all_true[:, 0], all_pred[:, 0])
        r2_surplus = _r2_score(all_true[:, 1], all_pred[:, 1])
        r2_shortfall = _r2_score(all_true[:, 2], all_pred[:, 2])

        pred_np = all_pred.numpy()
        true_np = all_true.numpy()
        sp_profit = _safe_spearman(true_np[:, 0], pred_np[:, 0])
        sp_surplus = _safe_spearman(true_np[:, 1], pred_np[:, 1])
        sp_shortfall = _safe_spearman(true_np[:, 2], pred_np[:, 2])

        # Log
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
                f"R²=[{r2_profit:.3f}, {r2_surplus:.3f}, {r2_shortfall:.3f}] "
                f"Sp=[{sp_profit:.3f}, {sp_surplus:.3f}, {sp_shortfall:.3f}] | "
                f"{elapsed:.1f}s"
            )

        # Checkpointing - choose metric for early stopping
        if use_spearman_stop:
            current_metric = (sp_profit + sp_surplus + sp_shortfall) / 3.0
            improved = current_metric > best_metric  # maximize Spearman
        else:
            current_metric = val_loss
            improved = current_metric < best_metric  # minimize loss

        if improved:
            best_metric = current_metric
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_kwargs": model_kwargs,
                    "norm_stats": norm.to_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                ckpt_dir / "best.pt",
            )
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            if verbose:
                metric_name = "mean_spearman" if use_spearman_stop else "val_loss"
                print(f"Early stopping at epoch {epoch} (best={best_epoch}, "
                      f"{metric_name}={best_metric:.4f})")
            break

    # Load best model
    ckpt = torch.load(ckpt_dir / "best.pt", weights_only=False, map_location=dev)
    model.load_state_dict(ckpt["model_state"])

    # Save training history
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    if verbose:
        metric_name = "mean_spearman" if use_spearman_stop else "val_loss"
        print(f"\nBest model at epoch {best_epoch}, {metric_name}={best_metric:.4f}")

    return model, norm, history

def _r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Compute R² score."""
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    if ss_tot < 1e-10:
        return 0.0
    return float(1 - ss_res / ss_tot)

def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Spearman correlation, returning 0.0 for constant inputs."""
    if np.std(y_pred) < 1e-10 or np.std(y_true) < 1e-10:
        return 0.0
    r, _ = spearmanr(y_true, y_pred)
    return float(r) if not np.isnan(r) else 0.0

def load_trained_model(
    checkpoint_path: str | Path,
    device: str = "auto",
) -> tuple[WheelingGNN, NormStats]:
    """Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint .pt file.
        device: Device to load onto.

    Returns:
        Tuple of (model, normalization stats).
    """
    dev = get_device(device)
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=dev)

    model = WheelingGNN(**ckpt["model_kwargs"]).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    norm = NormStats.from_dict(ckpt["norm_stats"])

    return model, norm
