"""WheelingMLP: Simple MLP surrogate baseline for ablation comparison.

Takes flattened features as input (no graph structure).
Used to demonstrate the value of GNN's graph-aware architecture.

Key difference from GNN:
- Input dimension depends on problem size (8m + 11n + 6 + mn)
- Cannot generalize across different problem sizes
- No message passing between generator and consumer nodes
"""

import torch
import torch.nn as nn

class WheelingMLP(nn.Module):
    """MLP surrogate baseline predicting (profit_norm, surplus_norm, shortfall_norm).

    Input: concatenation of flattened generator features, consumer features,
    global features, and allocation matrix W.

    Args:
        input_dim: Total input dimension (8m + 11n + 6 + mn).
        hidden_dim: Hidden layer dimension (default: 256).
        n_layers: Number of hidden layers (default: 3).
        dropout: Dropout rate (default: 0.1).
        n_outputs: Number of output objectives (default: 3).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        dropout: float = 0.1,
        n_outputs: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_outputs = n_outputs

        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, n_outputs))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Predictions of shape (batch_size, n_outputs).
        """
        return self.network(x)
