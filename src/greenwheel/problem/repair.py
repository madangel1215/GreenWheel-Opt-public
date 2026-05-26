"""
Repair operator for wheeling allocation.

Snaps small allocation weights to zero and normalizes rows
that exceed the generator capacity constraint (Σ_j w_{ij} ≤ 1).
"""

import numpy as np
from pymoo.core.repair import Repair

class AllocationRepair(Repair):
    """Repair operator applied after crossover/mutation.

    1. Snap w_{ij} < threshold to 0 (eliminates negligible allocations).
    2. If Σ_j w_{ij} > 1 for any generator i, normalize that row to sum to 1.
    """

    def __init__(self, threshold: float = 0.01):
        super().__init__()
        self.threshold = threshold

    def _do(self, problem, X, **kwargs):
        m = problem.instance.m
        n = problem.instance.n

        for k in range(len(X)):
            W = X[k].reshape(m, n)

            # Step 1: snap small weights to zero
            W[W < self.threshold] = 0.0

            # Step 2: normalize rows exceeding 1
            row_sums = W.sum(axis=1)
            overallocated = row_sums > 1.0
            if overallocated.any():
                W[overallocated] /= row_sums[overallocated, np.newaxis]

            X[k] = W.ravel()

        return X
