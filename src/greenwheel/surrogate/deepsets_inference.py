"""DeepSets surrogate with the same interface as BatchSurrogate.

The question is whether the DeepSets baseline, which ranks almost as well as the GNN on
the held-out test set, also delivers the GNN's gain inside SA-NSGA-II. To answer it the
two surrogates must be deployed identically, so this wrapper builds the same graph
for each candidate (same repair, same edge features) and only swaps the model.
"""

import numpy as np
import torch

from src.greenwheel.config import WheelingInstance
from src.greenwheel.surrogate.baselines import (
    DeepSetsNorm,
    WheelingDeepSets,
    _collate,
    build_deepsets_arrays,
)
from src.greenwheel.surrogate.graph import instance_to_heterodata


class DeepSetsBatchSurrogate:
    def __init__(self, model: WheelingDeepSets, norm: DeepSetsNorm, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.norm = norm
        self.device = torch.device(device)
        self._instance: WheelingInstance | None = None

    def set_instance(self, instance: WheelingInstance) -> None:
        self._instance = instance

    def evaluate_batch(self, W_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self._instance is not None, "Call set_instance() first"
        inst = self._instance
        m, n = inst.m, inst.n
        W_matrices = W_batch.reshape(-1, m, n)
        graphs = []
        for W in W_matrices:
            # Identical to BatchSurrogate.evaluate_batch, so both surrogates see the same input.
            W_rep = W.copy()
            W_rep[W_rep < 0.01] = 0.0
            row_sums = W_rep.sum(axis=1)
            over = row_sums > 1.0
            if over.any():
                W_rep[over] /= row_sums[over, np.newaxis]
            # The training graphs carry targets; a zero placeholder is normalized and discarded.
            graphs.append(instance_to_heterodata(inst, W_rep, targets=np.zeros(3)))
        samples = [self.norm.apply(s) for s in build_deepsets_arrays(graphs)]
        with torch.no_grad():
            gen, con, gm, cm, glob, _ = _collate(samples, self.device)
            pred = self.model(gen, con, gm, cm, glob).cpu().numpy()
        pred = self.norm.denorm_y(pred)
        return pred[:, 0], pred[:, 1], pred[:, 2]


def load_deepsets(path: str, device: str = "cpu") -> DeepSetsBatchSurrogate:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = WheelingDeepSets(**ck["dims"])
    model.load_state_dict(ck["state_dict"])
    return DeepSetsBatchSurrogate(model, ck["norm"], device=device)
