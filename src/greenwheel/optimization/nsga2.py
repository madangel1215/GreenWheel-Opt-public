"""
NSGA-II runner for green energy wheeling allocation optimization.

Wraps pymoo's NSGA2 with the WheelingProblem and AllocationRepair operator.
Produces a Pareto front of profit vs temporal mismatch solutions.
"""

import time
import numpy as np
from dataclasses import dataclass, field

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination

from src.greenwheel.config import WheelingConfig, WheelingInstance
from src.greenwheel.optimization.metrics import compute_hypervolume
from src.greenwheel.problem.formulation import WheelingProblem
from src.greenwheel.problem.repair import AllocationRepair
from src.greenwheel.simulator.matching import simulate_matching

class HVCallback(Callback):
    """Callback that records per-generation Pareto fronts for post-hoc HV.

    Stores the non-dominated front F at each generation. HV can be computed
    afterward with a consistent ref_point using ``compute_hv_history()``.
    """

    def __init__(self, interval: int = 1) -> None:
        super().__init__()
        self.interval = interval
        self.fronts: list[tuple[int, np.ndarray]] = []  # (gen, F)

    def notify(self, algorithm) -> None:
        gen = algorithm.n_gen
        if gen % self.interval != 0:
            return
        F = algorithm.opt.get("F")
        if F is not None and len(F) > 0:
            self.fronts.append((gen, F.copy()))

    def compute_hv_history(self, ref_point: np.ndarray) -> list[dict]:
        """Compute HV for all stored fronts using a given ref_point."""
        return [
            {"generation": gen, "hv": compute_hypervolume(F, ref_point)}
            for gen, F in self.fronts
        ]

@dataclass
class ParetoSolution:
    """A single solution on the Pareto front."""

    neg_profit: float          # f₁ = -profit
    surplus: float             # f₂ = total surplus (kWh)
    shortfall: float           # f₃ = total shortfall (kWh)
    W: np.ndarray              # allocation matrix (m, n)
    decision_vector: np.ndarray  # flattened W

    @property
    def profit(self) -> float:
        return -self.neg_profit

@dataclass
class OptimizationResult:
    """Result of NSGA-II optimization."""

    pareto_solutions: list[ParetoSolution]
    n_generations: int
    n_evaluations: int
    elapsed_seconds: float
    hv_history: list[dict] = field(default_factory=list)
    fronts_history: list[tuple[int, np.ndarray]] = field(default_factory=list)

    def compute_hv_history(self, ref_point: np.ndarray) -> list[dict]:
        """Recompute HV history with a new ref_point from stored fronts."""
        return [
            {"generation": gen, "hv": compute_hypervolume(F, ref_point)}
            for gen, F in self.fronts_history
        ]

    @property
    def pareto_front(self) -> np.ndarray:
        """Return (n_solutions, 3) array of [neg_profit, surplus, shortfall]."""
        return np.array([
            [s.neg_profit, s.surplus, s.shortfall] for s in self.pareto_solutions
        ])

    @property
    def profits(self) -> np.ndarray:
        """Profit values (positive = good)."""
        return np.array([s.profit for s in self.pareto_solutions])

    @property
    def surpluses(self) -> np.ndarray:
        """Surplus values (lower = better)."""
        return np.array([s.surplus for s in self.pareto_solutions])

    @property
    def shortfalls(self) -> np.ndarray:
        """Shortfall values (lower = better)."""
        return np.array([s.shortfall for s in self.pareto_solutions])

    def get_knee_point(self) -> ParetoSolution:
        """Find the knee point on the Pareto front.

        Uses minimum distance to utopia point in normalized objective space.
        Works for any number of objectives.
        """
        front = self.pareto_front
        if len(front) <= 1:
            return self.pareto_solutions[0]

        f_min = front.min(axis=0)
        f_max = front.max(axis=0)
        f_range = f_max - f_min
        f_range[f_range == 0] = 1.0
        norm = (front - f_min) / f_range

        # Distance to utopia (0,0,...,0) in normalized space
        distances = np.sqrt((norm ** 2).sum(axis=1))
        knee_idx = int(np.argmin(distances))

        return self.pareto_solutions[knee_idx]

def run_nsga2(
    instance: WheelingInstance,
    config: WheelingConfig | None = None,
    seed: int = 42,
    verbose: bool = True,
    track_convergence: bool = False,
    ref_point: np.ndarray | None = None,
    hv_interval: int = 1,
) -> OptimizationResult:
    """Run NSGA-II for wheeling allocation optimization.

    Args:
        instance: Problem instance with generators, consumers, profiles.
        config: Optimization hyperparameters.
        seed: Random seed.
        verbose: Print progress to stdout.
        track_convergence: If True, record per-generation fronts. HV is
            computed post-hoc using ref_point (or auto-computed from final front).
        ref_point: Reference point for HV. If None and track_convergence=True,
            auto-computed from the final Pareto front.
        hv_interval: Record fronts every N generations (default 1).

    Returns:
        OptimizationResult with Pareto solutions (and hv_history if tracked).
    """
    if config is None:
        config = WheelingConfig()

    # Problem
    problem = WheelingProblem(instance)

    # Callback for convergence tracking (stores fronts, HV computed post-hoc)
    callback = None
    if track_convergence:
        callback = HVCallback(interval=hv_interval)

    # Algorithm with repair operator
    algorithm = NSGA2(
        pop_size=config.population_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=config.crossover_prob, eta=config.crossover_eta),
        mutation=PM(eta=config.mutation_eta),
        repair=AllocationRepair(threshold=config.min_weight_threshold),
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", config.n_generations)

    if verbose:
        m, n = instance.m, instance.n
        print(f"[NSGA-II] Problem: m={m} generators, n={n} consumers")
        print(f"[NSGA-II] Decision variables: {m * n}")
        print(f"[NSGA-II] Constraints: {m + 2 * m * n}")
        print(f"[NSGA-II] Pop={config.population_size}, Gen={config.n_generations}")

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

        # Sort by profit (descending)
        pareto_solutions.sort(key=lambda s: s.profit, reverse=True)

    n_eval = res.algorithm.evaluator.n_eval if res.algorithm else 0

    if verbose:
        print(f"\n[NSGA-II] Completed in {elapsed:.1f}s")
        print(f"[NSGA-II] Evaluations: {n_eval}")
        print(f"[NSGA-II] Pareto solutions: {len(pareto_solutions)}")
        if pareto_solutions:
            profits = [s.profit for s in pareto_solutions]
            surpluses = [s.surplus for s in pareto_solutions]
            shortfalls = [s.shortfall for s in pareto_solutions]
            print(f"[NSGA-II] Profit range: [{min(profits):.0f}, {max(profits):.0f}] NTD")
            print(f"[NSGA-II] Surplus range: [{min(surpluses):.0f}, {max(surpluses):.0f}] kWh")
            print(f"[NSGA-II] Shortfall range: [{min(shortfalls):.0f}, {max(shortfalls):.0f}] kWh")

    # Compute HV history post-hoc with correct ref_point
    hv_history: list[dict] = []
    fronts_history: list[tuple[int, np.ndarray]] = []
    if callback is not None:
        fronts_history = callback.fronts
        if ref_point is None and len(pareto_solutions) > 0:
            # Auto-compute ref_point from final Pareto front
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
