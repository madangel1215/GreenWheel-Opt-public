"""MOEA/D runner for green energy wheeling allocation optimization.

Wraps pymoo's MOEAD with the WheelingProblem and AllocationRepair operator.
Uses Tchebycheff decomposition with uniform reference directions.

Note: pymoo's MOEA/D does not support explicit constraints, so we use a
penalty-based formulation (WheelingProblemPenalty) that adds constraint
violations as penalty terms to the objectives.
"""

import time
import numpy as np

from pymoo.algorithms.moo.moead import MOEAD
from pymoo.core.problem import ElementwiseProblem
from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

from src.greenwheel.config import WheelingConfig, WheelingInstance
from src.greenwheel.optimization.nsga2 import HVCallback, OptimizationResult, ParetoSolution
from src.greenwheel.problem.repair import AllocationRepair
from src.greenwheel.simulator.matching import simulate_matching

class WheelingProblemPenalty(ElementwiseProblem):
    """Unconstrained penalty-based variant for MOEA/D compatibility.

    Moves constraint violations into penalty terms added to objectives.
    AllocationRepair already handles C1 (row sums ≤ 1), so penalty is
    mainly for fairness constraints (C3) that slip through repair.
    """

    def __init__(self, instance: WheelingInstance, penalty_weight: float = 100.0):
        self.instance = instance
        self.penalty_weight = penalty_weight
        m, n = instance.m, instance.n
        self._pi = instance.target_proportions
        self._alpha = instance.market.fairness_tolerance

        super().__init__(
            n_var=m * n,
            n_obj=3,
            n_ieq_constr=0,  # No explicit constraints
            xl=np.zeros(m * n),
            xu=np.ones(m * n),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        m, n = self.instance.m, self.instance.n
        W = x.reshape(m, n)
        W[W < 0.01] = 0.0

        result = simulate_matching(W, self.instance)

        # Compute fairness constraint violation penalty
        penalty = self._constraint_penalty(W)

        out["F"] = np.array([
            -result.profit + self.penalty_weight * penalty,
            result.total_surplus + self.penalty_weight * penalty,
            result.total_shortfall + self.penalty_weight * penalty,
        ])

    def _constraint_penalty(self, W: np.ndarray) -> float:
        """Sum of squared constraint violations for fairness (C3)."""
        m, n = self.instance.m, self.instance.n
        pi = self._pi
        alpha = self._alpha

        violation = 0.0
        row_sums = W.sum(axis=1)

        for i in range(m):
            if row_sums[i] > 1e-6:
                actual_prop = W[i, :] / row_sums[i]
            else:
                actual_prop = np.zeros(n)

            for j in range(n):
                upper = actual_prop[j] - pi[j] - alpha
                lower = pi[j] - actual_prop[j] - alpha
                if upper > 0:
                    violation += upper ** 2
                if lower > 0:
                    violation += lower ** 2

        return violation

def run_moead(
    instance: WheelingInstance,
    config: WheelingConfig | None = None,
    seed: int = 42,
    verbose: bool = True,
    track_convergence: bool = False,
    ref_point: np.ndarray | None = None,
    hv_interval: int = 1,
    n_partitions: int = 12,
    n_neighbors: int = 20,
    prob_neighbor_mating: float = 0.9,
) -> OptimizationResult:
    """Run MOEA/D for wheeling allocation optimization.

    Uses penalty-based formulation since pymoo's MOEA/D doesn't support
    explicit constraints.

    Args:
        instance: Problem instance with generators, consumers, profiles.
        config: Optimization hyperparameters.
        seed: Random seed.
        verbose: Print progress to stdout.
        track_convergence: If True, record per-generation fronts.
        ref_point: Reference point for HV computation.
        hv_interval: Record fronts every N generations (default 1).
        n_partitions: Number of weight vector partitions (pop_size = n_partitions + 1).
        n_neighbors: Number of neighbors for weight vector neighborhood.
        prob_neighbor_mating: Probability of mating within neighborhood.

    Returns:
        OptimizationResult with Pareto solutions.
    """
    if config is None:
        config = WheelingConfig()

    problem = WheelingProblemPenalty(instance)

    # Reference directions for 3 objectives
    ref_dirs = get_reference_directions("uniform", 3, n_partitions=n_partitions)

    callback = None
    if track_convergence:
        callback = HVCallback(interval=hv_interval)

    algorithm = MOEAD(
        ref_dirs=ref_dirs,
        n_neighbors=n_neighbors,
        decomposition=Tchebicheff(),
        prob_neighbor_mating=prob_neighbor_mating,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=config.crossover_prob, eta=config.crossover_eta),
        mutation=PM(eta=config.mutation_eta),
        repair=AllocationRepair(threshold=config.min_weight_threshold),
    )

    termination = get_termination("n_gen", config.n_generations)

    if verbose:
        m, n = instance.m, instance.n
        print(f"[MOEA/D] Problem: m={m} generators, n={n} consumers")
        print(f"[MOEA/D] Decision variables: {m * n}")
        print(f"[MOEA/D] Pop={len(ref_dirs)} (from {n_partitions} partitions), "
              f"Gen={config.n_generations}")

    t0 = time.time()
    minimize_kwargs = dict(seed=seed, verbose=verbose)
    if callback is not None:
        minimize_kwargs["callback"] = callback
    res = pymoo_minimize(
        problem,
        algorithm,
        termination,
        **minimize_kwargs,
    )
    elapsed = time.time() - t0

    # Extract Pareto solutions - re-evaluate with true objectives (no penalty)
    # to get clean F values for fair HV comparison
    pareto_solutions = []
    if res.F is not None:
        m, n = instance.m, instance.n
        for i in range(len(res.F)):
            x = res.X[i]
            W = x.reshape(m, n)
            W[W < 0.01] = 0.0

            # Re-evaluate with true simulator for clean objectives
            mr = simulate_matching(W, instance)

            pareto_solutions.append(ParetoSolution(
                neg_profit=float(-mr.profit),
                surplus=float(mr.total_surplus),
                shortfall=float(mr.total_shortfall),
                W=W.copy(),
                decision_vector=x.copy(),
            ))
        pareto_solutions.sort(key=lambda s: s.profit, reverse=True)

    n_eval = res.algorithm.evaluator.n_eval if res.algorithm else 0

    if verbose:
        print(f"\n[MOEA/D] Completed in {elapsed:.1f}s")
        print(f"[MOEA/D] Evaluations: {n_eval}")
        print(f"[MOEA/D] Pareto solutions: {len(pareto_solutions)}")
        if pareto_solutions:
            profits = [s.profit for s in pareto_solutions]
            surpluses = [s.surplus for s in pareto_solutions]
            shortfalls = [s.shortfall for s in pareto_solutions]
            print(f"[MOEA/D] Profit range: [{min(profits):.0f}, {max(profits):.0f}] NTD")
            print(f"[MOEA/D] Surplus range: [{min(surpluses):.0f}, {max(surpluses):.0f}] kWh")
            print(f"[MOEA/D] Shortfall range: [{min(shortfalls):.0f}, {max(shortfalls):.0f}] kWh")

    # HV history - recompute from stored fronts (which contain penalized values)
    # For post-hoc HV, the penalized F is close enough (repair handles most violations)
    hv_history: list[dict] = []
    fronts_history: list[tuple[int, np.ndarray]] = []
    if callback is not None:
        fronts_history = callback.fronts
        if ref_point is None and len(pareto_solutions) > 0:
            front = np.array([[s.neg_profit, s.surplus, s.shortfall] for s in pareto_solutions])
            ref_point = front.max(axis=0) * 1.1
        if ref_point is not None:
            hv_history = [
                {"generation": gen, "hv": _compute_hv_safe(F, ref_point)}
                for gen, F in fronts_history
            ]

    return OptimizationResult(
        pareto_solutions=pareto_solutions,
        n_generations=config.n_generations,
        n_evaluations=n_eval,
        elapsed_seconds=elapsed,
        hv_history=hv_history,
        fronts_history=fronts_history,
    )

def _compute_hv_safe(F: np.ndarray, ref_point: np.ndarray) -> float:
    """Compute HV with safety against penalized fronts exceeding ref_point."""
    from src.greenwheel.optimization.metrics import compute_hypervolume
    # Clip penalized values that exceed ref_point
    F_clipped = np.minimum(F, ref_point)
    return compute_hypervolume(F_clipped, ref_point)
