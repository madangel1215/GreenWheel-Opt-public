"""GNN surrogate model for wheeling allocation evaluation."""

from src.greenwheel.surrogate.model import WheelingGNN
from src.greenwheel.surrogate.graph import instance_to_heterodata
from src.greenwheel.surrogate.features import (
    extract_generator_features,
    extract_consumer_features,
    extract_edge_features,
    extract_global_features,
)
from src.greenwheel.surrogate.inference import BatchSurrogate

__all__ = [
    "WheelingGNN",
    "instance_to_heterodata",
    "extract_generator_features",
    "extract_consumer_features",
    "extract_edge_features",
    "extract_global_features",
    "BatchSurrogate",
]
