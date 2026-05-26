"""WheelingGNN: Heterogeneous GNN surrogate for wheeling allocation.

Architecture:
    Input projection (per node type) → 3× HeteroConv(GATv2) layers
    → Global mean pooling → Concat with global features → MLP → (profit, surplus, shortfall)

~140K parameters. Designed for fast inference on MPS/CPU.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import (
    GATConv, GATv2Conv, HeteroConv, TransformerConv, global_mean_pool,
)

class WheelingGNN(nn.Module):
    """Heterogeneous GNN surrogate predicting (profit_norm, surplus_norm, shortfall_norm).

    Args:
        gen_in: Generator node feature dimension (default: 8).
        con_in: Consumer node feature dimension (default: 11).
        edge_dim: Edge feature dimension (default: 3).
        global_dim: Global feature dimension (default: 6).
        hidden_dim: Hidden dimension for all layers (default: 64).
        n_layers: Number of HeteroConv layers (default: 3).
        heads: Number of attention heads in GATv2 (default: 4).
        dropout: Dropout rate in prediction MLP (default: 0.1).
        n_outputs: Number of output objectives (default: 3).
    """

    def __init__(
        self,
        gen_in: int = 8,
        con_in: int = 11,
        edge_dim: int = 3,
        global_dim: int = 6,
        hidden_dim: int = 64,
        n_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        n_outputs: int = 3,
        conv_type: str = "gatv2",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_outputs = n_outputs
        self.conv_type = conv_type

        # Input projections (per node type)
        self.gen_proj = nn.Sequential(
            nn.Linear(gen_in, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.con_proj = nn.Sequential(
            nn.Linear(con_in, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Edge-aware message-passing layers. conv_type selects the attention
        # backbone (all variants consume the edge features for a fair comparison).
        def _make_conv():
            kw = dict(heads=heads, concat=False, edge_dim=edge_dim)
            if conv_type == "gatv2":
                return GATv2Conv(hidden_dim, hidden_dim, add_self_loops=False, **kw)
            if conv_type == "gat":
                return GATConv(hidden_dim, hidden_dim, add_self_loops=False, **kw)
            if conv_type == "transformer":
                return TransformerConv(hidden_dim, hidden_dim, **kw)
            raise ValueError(f"unknown conv_type: {conv_type}")

        self.convs = nn.ModuleList()
        self.norms_gen = nn.ModuleList()
        self.norms_con = nn.ModuleList()

        for _ in range(n_layers):
            conv = HeteroConv(
                {
                    ("gen", "supplies", "con"): _make_conv(),
                    ("con", "demands", "gen"): _make_conv(),
                },
                aggr="sum",
            )
            self.convs.append(conv)
            self.norms_gen.append(nn.LayerNorm(hidden_dim))
            self.norms_con.append(nn.LayerNorm(hidden_dim))

        # Prediction MLP: 2 * hidden_dim (pooled) + global_dim → n_outputs objectives
        mlp_in = 2 * hidden_dim + global_dim
        self.predictor = nn.Sequential(
            nn.Linear(mlp_in, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_outputs),
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
        edge_attr_dict: dict[tuple, torch.Tensor],
        batch_dict: dict[str, torch.Tensor] | None = None,
        global_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x_dict: Node features {'gen': (N_gen, feat), 'con': (N_con, feat)}.
            edge_index_dict: Edge indices per edge type.
            edge_attr_dict: Edge features per edge type.
            batch_dict: Batch assignment per node type (for batched graphs).
            global_feat: Global features (B, global_dim) or (global_dim,).

        Returns:
            Predictions of shape (B, n_outputs): [profit_norm, surplus_norm, shortfall_norm].
        """
        # Input projection
        h = {
            "gen": self.gen_proj(x_dict["gen"]),
            "con": self.con_proj(x_dict["con"]),
        }

        # Message passing with residual connections
        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index_dict, edge_attr_dict)

            # Residual + LayerNorm + ReLU
            h_new["gen"] = self.norms_gen[i](h_new["gen"] + h["gen"])
            h_new["con"] = self.norms_con[i](h_new["con"] + h["con"])
            h_new["gen"] = torch.relu(h_new["gen"])
            h_new["con"] = torch.relu(h_new["con"])

            h = h_new

        # Global mean pooling
        gen_batch = batch_dict["gen"] if batch_dict else torch.zeros(
            h["gen"].size(0), dtype=torch.long, device=h["gen"].device
        )
        con_batch = batch_dict["con"] if batch_dict else torch.zeros(
            h["con"].size(0), dtype=torch.long, device=h["con"].device
        )

        gen_pool = global_mean_pool(h["gen"], gen_batch)  # (B, hidden_dim)
        con_pool = global_mean_pool(h["con"], con_batch)  # (B, hidden_dim)

        # Concat pooled + global features
        if global_feat is not None:
            if global_feat.dim() == 1:
                global_feat = global_feat.unsqueeze(0)
            graph_repr = torch.cat([gen_pool, con_pool, global_feat], dim=-1)
        else:
            graph_repr = torch.cat([gen_pool, con_pool], dim=-1)

        return self.predictor(graph_repr)  # (B, n_outputs)

    @staticmethod
    def from_heterodata(data, model):
        """Extract inputs from a HeteroData batch and run forward pass.

        Convenience method for use with PyG DataLoader batches.

        Args:
            data: HeteroData batch from PyG DataLoader.
            model: WheelingGNN instance.

        Returns:
            Predictions of shape (B, n_outputs).
        """
        x_dict = {
            "gen": data["gen"].x,
            "con": data["con"].x,
        }
        edge_index_dict = {
            key: data[key].edge_index for key in data.edge_types
        }
        edge_attr_dict = {
            key: data[key].edge_attr for key in data.edge_types
        }
        batch_dict = {
            "gen": data["gen"].batch,
            "con": data["con"].batch,
        }
        global_feat = data.global_feat if hasattr(data, "global_feat") else None

        return model(x_dict, edge_index_dict, edge_attr_dict, batch_dict, global_feat)
