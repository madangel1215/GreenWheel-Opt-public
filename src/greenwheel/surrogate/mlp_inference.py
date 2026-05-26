"""Batch inference interface for MLP surrogate (SA-MOEA integration).

Same interface as BatchSurrogate but uses the simple MLP model.
Takes flattened features instead of graph-structured input.
"""

import numpy as np
import torch

from src.greenwheel.config import WheelingInstance
from src.greenwheel.surrogate.features import (
    extract_consumer_features,
    extract_generator_features,
    extract_global_features,
)
from src.greenwheel.surrogate.mlp_model import WheelingMLP
from src.greenwheel.surrogate.training import NormStats, get_device

class MLPBatchSurrogate:
    """Batch inference wrapper for WheelingMLP surrogate.

    Provides the same interface as BatchSurrogate for drop-in replacement.

    Usage:
        surrogate = MLPBatchSurrogate(model, norm, device='mps')
        surrogate.set_instance(instance)
        profits, surpluses, shortfalls = surrogate.evaluate_batch(W_batch)
    """

    def __init__(
        self,
        model: WheelingMLP,
        norm: NormStats,
        device: str = "auto",
    ):
        self.model = model
        self.norm = norm
        self.device = get_device(device)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Cached instance data
        self._instance: WheelingInstance | None = None
        self._gen_feat_flat: np.ndarray | None = None
        self._con_feat_flat: np.ndarray | None = None
        self._global_feat_flat: np.ndarray | None = None

    def set_instance(self, instance: WheelingInstance) -> None:
        """Cache instance-level features (call once per instance)."""
        self._instance = instance

        # Extract and flatten node features
        gen_feat = extract_generator_features(instance).numpy()  # (m, 8)
        con_feat = extract_consumer_features(instance).numpy()   # (n, 11)
        global_feat = extract_global_features(instance).numpy()  # (6,)

        # Normalize features using stored stats
        gen_mean = np.array(self.norm.gen_mean[:8])
        gen_std = np.array(self.norm.gen_std[:8])
        con_mean = np.array(self.norm.con_mean[:11])
        con_std = np.array(self.norm.con_std[:11])
        global_mean = np.array(self.norm.global_mean[:6])
        global_std = np.array(self.norm.global_std[:6])

        gen_feat = (gen_feat - gen_mean) / gen_std
        con_feat = (con_feat - con_mean) / con_std
        global_feat = (global_feat - global_mean) / global_std

        self._gen_feat_flat = gen_feat.flatten()    # (8m,)
        self._con_feat_flat = con_feat.flatten()    # (11n,)
        self._global_feat_flat = global_feat        # (6,)

    def evaluate_batch(self, W_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate a batch of allocation matrices.

        Args:
            W_batch: Allocation matrices, shape (batch_size, m*n) or (batch_size, m, n).

        Returns:
            Tuple of (profits, surpluses, shortfalls), each shape (batch_size,).
        """
        assert self._instance is not None, "Call set_instance() first"

        instance = self._instance
        m, n = instance.m, instance.n

        # Reshape if needed
        if W_batch.ndim == 2 and W_batch.shape[1] == m * n:
            W_flat = W_batch  # already flat
        elif W_batch.ndim == 3:
            W_flat = W_batch.reshape(-1, m * n)
        else:
            raise ValueError(f"Unexpected W_batch shape: {W_batch.shape}")

        batch_size = len(W_flat)

        # Normalize W using edge stats (w_ij is feature 0 of edge features)
        edge_mean_w = self.norm.edge_mean[0] if self.norm.edge_mean else 0.0
        edge_std_w = self.norm.edge_std[0] if self.norm.edge_std else 1.0
        W_norm = (W_flat - edge_mean_w) / edge_std_w

        # Build input: [gen_feat_flat, con_feat_flat, global_feat, W_flat]
        gen_tile = np.tile(self._gen_feat_flat, (batch_size, 1))
        con_tile = np.tile(self._con_feat_flat, (batch_size, 1))
        global_tile = np.tile(self._global_feat_flat, (batch_size, 1))

        x = np.concatenate([gen_tile, con_tile, global_tile, W_norm], axis=1)
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        # Forward pass
        with torch.no_grad():
            pred_norm = self.model(x_tensor)

        # Denormalize
        pred_raw = self.norm.denormalize(pred_norm).cpu().numpy()

        return pred_raw[:, 0], pred_raw[:, 1], pred_raw[:, 2]

    def evaluate_single(self, W: np.ndarray) -> tuple[float, float, float]:
        """Evaluate a single allocation matrix."""
        if W.ndim == 1:
            W = W.reshape(1, -1)
        elif W.ndim == 2 and W.shape[0] != 1:
            W = W.reshape(1, -1)

        profits, surpluses, shortfalls = self.evaluate_batch(W)
        return float(profits[0]), float(surpluses[0]), float(shortfalls[0])
