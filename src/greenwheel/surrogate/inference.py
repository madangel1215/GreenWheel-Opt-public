"""Batch inference interface for SA-MOEA integration.

Provides BatchSurrogate that caches instance features and supports
efficient batch GPU inference for use inside MOEA evaluation loops.
"""

import numpy as np
import torch
from torch_geometric.data import Batch

from src.greenwheel.config import WheelingInstance
from src.greenwheel.surrogate.features import (
    extract_consumer_features,
    extract_generator_features,
    extract_global_features,
)
from src.greenwheel.surrogate.graph import instance_to_heterodata
from src.greenwheel.surrogate.model import WheelingGNN
from src.greenwheel.surrogate.training import NormStats, get_device, normalize_features

class BatchSurrogate:
    """Batch inference wrapper for WheelingGNN surrogate.

    Caches instance-level features (node features, global features)
    and only recomputes edge features per allocation matrix.

    Usage:
        surrogate = BatchSurrogate(model, norm, device='mps')
        surrogate.set_instance(instance)

        # Inside MOEA evaluation loop:
        profits, surpluses, shortfalls = surrogate.evaluate_batch(W_batch)

        # Single evaluation compatibility:
        profit, surplus, shortfall = surrogate.evaluate_single(W)
    """

    def __init__(
        self,
        model: WheelingGNN,
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
        self._gen_feat: torch.Tensor | None = None
        self._con_feat: torch.Tensor | None = None
        self._global_feat: torch.Tensor | None = None

    def set_instance(self, instance: WheelingInstance) -> None:
        """Cache instance-level features (call once per instance).

        Args:
            instance: Problem instance to cache features for.
        """
        self._instance = instance
        self._gen_feat = extract_generator_features(instance)
        self._con_feat = extract_consumer_features(instance)
        self._global_feat = extract_global_features(instance)

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

        # Reshape if flat
        if W_batch.ndim == 2 and W_batch.shape[1] == m * n:
            W_matrices = W_batch.reshape(-1, m, n)
        elif W_batch.ndim == 3:
            W_matrices = W_batch
        else:
            raise ValueError(f"Unexpected W_batch shape: {W_batch.shape}")

        # Build HeteroData batch
        data_list = []
        for W in W_matrices:
            # Repair allocation
            W_rep = W.copy()
            W_rep[W_rep < 0.01] = 0.0
            row_sums = W_rep.sum(axis=1)
            over = row_sums > 1.0
            if over.any():
                W_rep[over] /= row_sums[over, np.newaxis]

            graph = instance_to_heterodata(instance, W_rep)
            data_list.append(graph)

        # Normalize input features (critical: model was trained on normalized data)
        normalize_features(data_list, self.norm)

        batch = Batch.from_data_list(data_list).to(self.device)

        # Forward pass
        with torch.no_grad():
            x_dict = {"gen": batch["gen"].x, "con": batch["con"].x}
            edge_index_dict = {k: batch[k].edge_index for k in batch.edge_types}
            edge_attr_dict = {k: batch[k].edge_attr for k in batch.edge_types}
            batch_dict = {"gen": batch["gen"].batch, "con": batch["con"].batch}
            global_feat = batch.global_feat if hasattr(batch, "global_feat") else None

            pred_norm = self.model(
                x_dict, edge_index_dict, edge_attr_dict, batch_dict, global_feat
            )

        # Denormalize
        pred_raw = self.norm.denormalize(pred_norm).cpu().numpy()

        return pred_raw[:, 0], pred_raw[:, 1], pred_raw[:, 2]

    def evaluate_single(self, W: np.ndarray) -> tuple[float, float, float]:
        """Evaluate a single allocation matrix.

        Args:
            W: Allocation matrix shape (m, n) or flat (m*n,).

        Returns:
            Tuple of (profit, surplus, shortfall).
        """
        if W.ndim == 1:
            W = W.reshape(1, -1)
        elif W.ndim == 2 and W.shape[0] != 1:
            W = W.reshape(1, -1)

        profits, surpluses, shortfalls = self.evaluate_batch(W)
        return float(profits[0]), float(surpluses[0]), float(shortfalls[0])
