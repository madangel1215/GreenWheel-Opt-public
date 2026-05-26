"""NSGA-III runner for green energy wheeling allocation optimization.

Wraps pymoo's NSGA3 with the WheelingProblem and AllocationRepair operator.
Uses Das-Dennis reference directions for many-objective decomposition.
"""

import time
import numpy as np

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

from src.greenwheel.config import WheelingConfig, WheelingInstance
from src.greenwheel.optimization.nsga2 import HVCallback, OptimizationResult, ParetoSolution
from src.greenwheel.problem.formulation import WheelingProblem
from src.greenwheel.problem.repair import AllocationRepair

def run_nsga3(
    instance: WheelingInstance,
    config: WheelingConfig | None = None,
    seed: int = 42,
    verbose: bool = True,
    track_convergence: bool = False,
    ref_point: np.ndarray | None = None,
    hv_interval: int = 1,
    n_partitions: int = 12,
) -> OptimizationResult:
    """Run NSGA-III for wheeling allocation optimization.

    Args:
        instance: Problem instance with generators, consumers, profiles.
        config: Optimization hyperparameters.
        seed: Random seed.
        verbose: Print progress to stdout.
        track_convergence: If True, record per-generation fronts.
        ref_point: Reference point for HV computation.
        hv_interval: Record fronts every N generations (default 1).
        n_partitions: Number of Das-Dennis partitions (pop_size = n_partitions + 1).

    Returns:
        OptimizationResult with Pareto solutions.
    """
    if config is None:
        config = WheelingConfig()

    problem = WheelingProblem(instance)

    # Das-Dennis reference directions for 3 objectives
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=n_partitions)

    callback = None
    if track_convergence:
        callback = HVCallback(interval=hv_interval)

    algorithm = NSGA3(
        ref_dirs=ref_dirs,
        pop_size=len(ref_dirs),
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=config.crossover_prob, eta=config.crossover_eta),
        mutation=PM(eta=config.mutation_eta),
        repair=AllocationRepair(threshold=config.min_weight_threshold),
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", config.n_generations)

    if verbose:
        m, n = instance.m, instance.n
        print(f"[NSGA-III] Problem: m={m} generators, n={n} consumers")
        print(f"[NSGA-III] Decision variables: {m * n}")
        print(f"[NSGA-III] Pop={len(ref_dirs)} (from {n_partitions} Das-Dennis partitions), "
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

    # Extract Pareto solutions
    pareto_solutions = []
    if res.F is not None:
        m, n = instance.m, instance.n
        for i in range(len(res.F)):
            x = res.X[i]
            W = x.reshape(m, n)
            pareto_solutions.append(ParetoSolution(
                neg_profit=float(res.F[i, 0]),
                surplus=float(res.F[i, 1]),
                shortfall=float(res.F[i, 2]),
                W=W.copy(),
                decision_vector=x.copy(),
            ))
        pareto_solutions.sort(key=lambda s: s.profit, reverse=True)

    n_eval = res.algorithm.evaluator.n_eval if res.algorithm else 0

    if verbose:
        print(f"\n[NSGA-III] Completed in {elapsed:.1f}s")
        print(f"[NSGA-III] Evaluations: {n_eval}")
        print(f"[NSGA-III] Pareto solutions: {len(pareto_solutions)}")
        if pareto_solutions:
            profits = [s.profit for s in pareto_solutions]
            surpluses = [s.surplus for s in pareto_solutions]
            shortfalls = [s.shortfall for s in pareto_solutions]
            print(f"[NSGA-III] Profit range: [{min(profits):.0f}, {max(profits):.0f}] NTD")
            print(f"[NSGA-III] Surplus range: [{min(surpluses):.0f}, {max(surpluses):.0f}] kWh")
            print(f"[NSGA-III] Shortfall range: [{min(shortfalls):.0f}, {max(shortfalls):.0f}] kWh")

    # HV history
    hv_history: list[dict] = []
    fronts_history: list[tuple[int, np.ndarray]] = []
    if callback is not None:
        fronts_history = callback.fronts
        if ref_point is None and len(pareto_solutions) > 0:
            front = np.array([[s.neg_profit, s.surplus, s.shortfall] for s in pareto_solutions])
            ref_point = front.max(axis=0) * 1.1
        if ref_point is not None:
            hv_history = callback.compute_hv_history(ref_point)

    return OptimizationResult(
        pareto_solutions=pareto_solutions,
        n_generations=config.n_generations,
        n_evaluations=n_eval,
        elapsed_seconds=elapsed,
        hv_history=hv_history,
        fronts_history=fronts_history,
    )
