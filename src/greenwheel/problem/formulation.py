"""
pymoo problem definition for green energy wheeling allocation.

Decision variables: W ∈ R^{m×n}, w_{ij} ∈ [0, 1]
  Flattened to x = [w_{11}, w_{12}, ..., w_{mn}], dim = m × n

Objectives (minimize all three):
  f₁ = -profit (negative retailer profit)
  f₂ = surplus_total (total surplus electricity, kWh)
  f₃ = shortfall_total (total RE100 shortfall, kWh)

Constraints:
  C1: Σ_j w_{ij} ≤ 1         (generator allocation sum)     → m inequalities
  C3: |prop_{ij} - π_j| ≤ α  (fairness per generator-pair)  → 2mn inequalities
"""

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from src.greenwheel.config import WheelingInstance
from src.greenwheel.simulator.matching import simulate_matching

class WheelingProblem(ElementwiseProblem):
    """pymoo ElementwiseProblem for wheeling allocation optimization.

    Each individual encodes a flattened allocation matrix W (m×n)
    with continuous values in [0, 1]. Standard SBX crossover and
    polynomial mutation work directly on this encoding.
    """

    def __init__(self, instance: WheelingInstance):
        self.instance = instance
        m, n = instance.m, instance.n

        n_var = m * n
        n_obj = 3  # profit (negated), surplus, shortfall

        # Constraints: m (C1) + 2*m*n (C3 fairness)
        n_ieq_constr = m + 2 * m * n

        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_ieq_constr=n_ieq_constr,
            xl=np.zeros(n_var),
            xu=np.ones(n_var),
        )

        # Cache target proportions
        self._pi = instance.target_proportions
        self._alpha = instance.market.fairness_tolerance

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate a single allocation vector."""
        m, n = self.instance.m, self.instance.n
        W = x.reshape(m, n)

        # Repair: snap tiny weights to zero
        W[W < 0.01] = 0.0

        # Full simulation (profit + TMI)
        result = simulate_matching(W, self.instance)

        # f₁: minimize negative profit → maximize profit
        # f₂: minimize surplus electricity
        # f₃: minimize RE100 shortfall
        out["F"] = np.array([
            -result.profit,
            result.total_surplus,
            result.total_shortfall,
        ])

        # Constraints (g ≤ 0 means feasible)
        out["G"] = self._compute_constraints(W)

    def _compute_constraints(self, W: np.ndarray) -> np.ndarray:
        """Compute constraint violation vector.

        C1: Σ_j w_{ij} - 1 ≤ 0   for each generator i   (m constraints)
        C3: prop_{ij} - π_j - α ≤ 0  AND                 (mn constraints)
            π_j - prop_{ij} - α ≤ 0                       (mn constraints)

        where prop_{ij} = w_{ij} / Σ_{j'} w_{ij'} is the actual allocation
        proportion from generator i to consumer j.

        Returns:
            Constraint vector of length m + 2*m*n. Feasible when all ≤ 0.
        """
        m, n = self.instance.m, self.instance.n
        pi = self._pi
        alpha = self._alpha

        g = np.zeros(m + 2 * m * n)

        # C1: generator allocation sum ≤ 1
        row_sums = W.sum(axis=1)  # (m,)
        g[:m] = row_sums - 1.0

        # C3: fairness constraints
        idx = m
        for i in range(m):
            if row_sums[i] > 1e-6:
                actual_prop = W[i, :] / row_sums[i]
            else:
                # If generator not allocated, proportions are 0
                # Deviation from π is at most max(π), penalized
                actual_prop = np.zeros(n)

            for j in range(n):
                # Upper bound: prop_{ij} - π_j ≤ α
                g[idx] = actual_prop[j] - pi[j] - alpha
                idx += 1
                # Lower bound: π_j - prop_{ij} ≤ α
                g[idx] = pi[j] - actual_prop[j] - alpha
                idx += 1

        return g
