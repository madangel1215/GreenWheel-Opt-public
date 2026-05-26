"""HeteroData graph construction for GNN surrogate.

Converts WheelingInstance + allocation matrix W into a PyG HeteroData object
with bipartite generator→consumer and consumer→generator edges.
"""

import numpy as np
import torch
from torch_geometric.data import HeteroData

from src.greenwheel.config import WheelingInstance
from src.greenwheel.surrogate.features import (
    extract_consumer_features,
    extract_edge_features,
    extract_generator_features,
    extract_global_features,
)

def instance_to_heterodata(
    instance: WheelingInstance,
    W: np.ndarray,
    targets: np.ndarray | None = None,
) -> HeteroData:
    """Convert a wheeling instance and allocation matrix to a HeteroData graph.

    Creates a heterogeneous graph with:
    - Node types: 'gen' (generators), 'con' (consumers)
    - Edge types: ('gen', 'supplies', 'con') and ('con', 'demands', 'gen')
    - Edge features: allocation weight, price spread, gen/demand ratio

    Args:
        instance: Wheeling problem instance.
        W: Allocation matrix shape (m, n), values in [0, 1].
        targets: Optional target objectives [profit, surplus, shortfall], shape (3,).
            If provided, stored as data.y for supervised training.

    Returns:
        HeteroData with all node/edge features and optional targets.
    """
    m, n = instance.m, instance.n
    data = HeteroData()

    # --- Node features ---
    data["gen"].x = extract_generator_features(instance)   # (m, 8)
    data["con"].x = extract_consumer_features(instance)    # (n, 11)

    # --- Bipartite edges: full connectivity ---
    # gen→con edges: all (i, j) pairs
    gen_idx = torch.arange(m).repeat_interleave(n)  # [0,0,...,0, 1,1,...,1, ...]
    con_idx = torch.arange(n).repeat(m)              # [0,1,...,n-1, 0,1,...,n-1, ...]

    data["gen", "supplies", "con"].edge_index = torch.stack([gen_idx, con_idx])
    data["gen", "supplies", "con"].edge_attr = extract_edge_features(W, instance)

    # Reverse edges: con→gen (same features, same connectivity)
    data["con", "demands", "gen"].edge_index = torch.stack([con_idx, gen_idx])
    data["con", "demands", "gen"].edge_attr = extract_edge_features(W, instance)

    # --- Global features (stored as (1, 6) so PyG batches to (B, 6)) ---
    data.global_feat = extract_global_features(instance).unsqueeze(0)  # (1, 6)

    # --- Targets ---
    if targets is not None:
        data.y = torch.tensor(targets, dtype=torch.float32)

    # Store sizes for batch reconstruction
    data["gen"].num_nodes = m
    data["con"].num_nodes = n

    return data
