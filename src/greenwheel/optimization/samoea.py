"""Surrogate-Assisted MOEA (SA-MOEA) for wheeling allocation.

Pre-screening strategy: GNN surrogate ranks offspring, then only the
top candidates are truly evaluated and passed back to the base MOEA.
This exploits the surrogate's excellent ranking accuracy (Spearman > 0.98)
while avoiding its absolute-value inaccuracy.

Supports multiple base optimizers:
  - NSGA-II: crowding distance selection (default)
  - MOEA/D: Tchebycheff decomposition (penalty-based constraints)
  - NSGA-III: reference direction-based selection

Key idea - no F-value mixing:
  1. Generate offspring via base MOEA operators
  2. GNN predicts objectives for ALL offspring (cheap batch inference)
  3. Non-dominated sort on GNN predictions → select top-k candidates
  4. True simulator evaluates only top-k
  5. ONLY truly-evaluated solutions are returned to the MOEA
  6. No calibration, no mixing - all F values come from true simulator
"""

import time
from dataclasses import dataclass, field

import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

from src.greenwheel.config import WheelingInstance
from src.greenwheel.optimization.metrics import compute_hypervolume
from src.greenwheel.optimization.moead import WheelingProblemPenalty
from src.greenwheel.optimization.nsga2 import ParetoSolution
from src.greenwheel.problem.formulation import WheelingProblem
from src.greenwheel.problem.repair import AllocationRepair
from src.greenwheel.simulator.matching import simulate_matching
from src.greenwheel.surrogate.inference import BatchSurrogate

@dataclass
class SAMOEAConfig:
    """Configuration for surrogate-assisted MOEA.

    Supports NSGA-II, MOEA/D, and NSGA-III as base optimizers.
    """

    # Base algorithm selection
    base_algorithm: str = "nsga2"  # "nsga2", "moead", "nsga3"

    # Pre-screening parameters
    correction_fraction: float = 0.5
    warmup_generations: int = 10
    full_eval_interval: int = 20
    random_exploration_ratio: float = 0.1

    # Common MOEA parameters
    population_size: int = 100
    n_generations: int = 200
    crossover_prob: float = 0.9
    crossover_eta: float = 20.0
    mutation_eta: float = 20.0
    min_weight_threshold: float = 0.01

    # MOEA/D-specific parameters
    n_neighbors: int = 20
    prob_neighbor_mating: float = 0.9
    penalty_weight: float = 100.0

    # NSGA-III / MOEA/D reference direction partitions
    # For 3 objectives: n_partitions=12 → C(14,2)=91 ref dirs
    n_partitions: int = 12

@dataclass
class SAMOEAStats:
    """Per-generation statistics for SA-MOEA tracking."""

    generation: int
    mode: str  # "warmup", "full_eval", "prescreened"
    n_true_evals: int
    n_surr_evals: int
    hv: float = 0.0

@dataclass
class SAMOEAResult:
    """Result of SA-MOEA optimization with diagnostics."""

    pareto_solutions: list[ParetoSolution]
    n_generations: int
    n_true_evaluations: int
    n_surrogate_evaluations: int
    elapsed_seconds: float
    base_algorithm: str = "nsga2"
    gen_stats: list[SAMOEAStats] = field(default_factory=list)
    fronts_history: list[tuple[int, np.ndarray]] = field(default_factory=list)

    @property
    def pareto_front(self) -> np.ndarray:
        return np.array([
            [s.neg_profit, s.surplus, s.shortfall] for s in self.pareto_solutions
        ])

    @property
    def eval_savings(self) -> float:
        """Fraction of evaluations saved vs vanilla MOEA."""
        total = self.n_true_evaluations + self.n_surrogate_evaluations
        if total == 0:
            return 0.0
        return 1.0 - (self.n_true_evaluations / total)

    def compute_hv_history(self, ref_point: np.ndarray) -> list[dict]:
        """Compute HV for all stored fronts using a given ref_point."""
        return [
            {"generation": gen, "hv": compute_hypervolume(F, ref_point)}
            for gen, F in self.fronts_history
        ]

def _evaluate_true_batch(
    X: np.ndarray,
    instance: WheelingInstance,
) -> np.ndarray:
    """Evaluate solutions with true simulator.

    Args:
        X: Decision vectors, shape (n_solutions, m*n).
        instance: Problem instance.

    Returns:
        F: Objective values shape (n_solutions, 3) as [-profit, surplus, shortfall].
    """
    m, n = instance.m, instance.n
    F = np.zeros((len(X), 3))

    for i in range(len(X)):
        W = X[i].reshape(m, n)
        W[W < 0.01] = 0.0
        row_sums = W.sum(axis=1)
        over = row_sums > 1.0
        if over.any():
            W[over] /= row_sums[over, np.newaxis]

        result = simulate_matching(W, instance)
        F[i, 0] = -result.profit
        F[i, 1] = result.total_surplus
        F[i, 2] = result.total_shortfall

    return F

def _compute_constraints_batch(
    X: np.ndarray,
    instance: WheelingInstance,
) -> np.ndarray:
    """Compute analytical constraints for a batch of solutions.

    Args:
        X: Decision vectors, shape (n_solutions, m*n).
        instance: Problem instance.

    Returns:
        G: Constraint violations shape (n_solutions, m + 2*m*n).
    """
    m, n = instance.m, instance.n
    pi = instance.target_proportions
    alpha = instance.market.fairness_tolerance
    n_constr = m + 2 * m * n
    G = np.zeros((len(X), n_constr))

    for k in range(len(X)):
        W = X[k].reshape(m, n)
        g = np.zeros(n_constr)

        row_sums = W.sum(axis=1)
        g[:m] = row_sums - 1.0

        idx = m
        for i in range(m):
            if row_sums[i] > 1e-6:
                actual_prop = W[i, :] / row_sums[i]
            else:
                actual_prop = np.zeros(n)
            for j in range(n):
                g[idx] = actual_prop[j] - pi[j] - alpha
                idx += 1
                g[idx] = pi[j] - actual_prop[j] - alpha
                idx += 1

        G[k] = g

    return G

def _compute_penalty_batch(
    X: np.ndarray,
    instance: WheelingInstance,
) -> np.ndarray:
    """Compute fairness constraint penalty for each solution (for MOEA/D).

    Args:
        X: Decision vectors, shape (n_solutions, m*n).
        instance: Problem instance.

    Returns:
        penalties: shape (n_solutions,), sum of squared fairness violations.
    """
    m, n = instance.m, instance.n
    pi = instance.target_proportions
    alpha = instance.market.fairness_tolerance
    penalties = np.zeros(len(X))

    for k in range(len(X)):
        W = X[k].reshape(m, n)
        row_sums = W.sum(axis=1)
        violation = 0.0

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

        penalties[k] = violation

    return penalties

def _select_for_true_eval(
    F_surr: np.ndarray,
    n_select: int,
    random_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select indices for true evaluation based on surrogate NDS ranking.

    Top candidates by non-dominated front rank are selected, with a fraction
    reserved for random exploration to prevent exploitation bias.

    Args:
        F_surr: Surrogate objective predictions, shape (pop_size, 3).
        n_select: Number of solutions to select for true evaluation.
        random_ratio: Fraction of n_select reserved for random picks.
        rng: Random number generator.

    Returns:
        Array of selected indices.
    """
    pop_size = len(F_surr)
    n_select = min(n_select, pop_size)

    # Random exploration slots
    n_random = max(1, int(n_select * random_ratio))
    n_ranked = n_select - n_random

    # Non-dominated sorting on surrogate predictions
    nds = NonDominatedSorting()
    fronts = nds.do(F_surr)

    # Collect top-ranked indices
    ranked_indices = []
    for front in fronts:
        if len(ranked_indices) + len(front) <= n_ranked:
            ranked_indices.extend(front.tolist())
        else:
            remaining = n_ranked - len(ranked_indices)
            ranked_indices.extend(front[:remaining].tolist())
            break

    # Random exploration from remaining pool
    remaining_pool = list(set(range(pop_size)) - set(ranked_indices))
    if remaining_pool and n_random > 0:
        n_random_actual = min(n_random, len(remaining_pool))
        random_indices = rng.choice(
            remaining_pool, size=n_random_actual, replace=False
        ).tolist()
    else:
        random_indices = []

    selected = np.array(ranked_indices + random_indices, dtype=int)
    return selected

def _create_base_algorithm(
    config: SAMOEAConfig,
    instance: WheelingInstance,
) -> tuple[Algorithm, object, int]:
    """Create base MOEA algorithm, problem, and effective population size.

    Args:
        config: SA-MOEA configuration.
        instance: Problem instance.

    Returns:
        Tuple of (algorithm, problem, effective_pop_size).
    """
    sampling = FloatRandomSampling()
    crossover = SBX(prob=config.crossover_prob, eta=int(config.crossover_eta))
    mutation = PM(eta=int(config.mutation_eta))
    repair = AllocationRepair(threshold=config.min_weight_threshold)

    if config.base_algorithm == "nsga2":
        problem = WheelingProblem(instance)
        algorithm = NSGA2(
            pop_size=config.population_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            repair=repair,
            eliminate_duplicates=True,
        )
        return algorithm, problem, config.population_size

    elif config.base_algorithm == "moead":
        problem = WheelingProblemPenalty(
            instance, penalty_weight=config.penalty_weight,
        )
        ref_dirs = get_reference_directions(
            "uniform", 3, n_partitions=config.n_partitions,
        )
        algorithm = MOEAD(
            ref_dirs=ref_dirs,
            n_neighbors=config.n_neighbors,
            decomposition=Tchebicheff(),
            prob_neighbor_mating=config.prob_neighbor_mating,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            repair=repair,
        )
        return algorithm, problem, len(ref_dirs)

    elif config.base_algorithm == "nsga3":
        problem = WheelingProblem(instance)
        ref_dirs = get_reference_directions(
            "das-dennis", 3, n_partitions=config.n_partitions,
        )
        algorithm = NSGA3(
            ref_dirs=ref_dirs,
            pop_size=len(ref_dirs),
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            repair=repair,
            eliminate_duplicates=True,
        )
        return algorithm, problem, len(ref_dirs)

    else:
        raise ValueError(
            f"Unknown base_algorithm: {config.base_algorithm!r}. "
            f"Must be 'nsga2', 'moead', or 'nsga3'."
        )

def run_samoea(
    instance: WheelingInstance,
    surrogate: BatchSurrogate,
    config: SAMOEAConfig | None = None,
    seed: int = 42,
    verbose: bool = True,
    ref_point: np.ndarray | None = None,
) -> SAMOEAResult:
    """Run surrogate-assisted MOEA with pre-screening strategy.

    Pre-screening approach (no F-value mixing):
    - During prescreened generations, the base MOEA generates offspring
    - Surrogate ranks all offspring via NDS
    - Only top-k are truly evaluated and returned to the algorithm
    - MOEA merges parents + (smaller) infills → survival selection
    - All F values in the system come from the true simulator

    Supports three base optimizers:
    - NSGA-II: explicit constraints, crowding distance
    - MOEA/D: penalty-based constraints, Tchebycheff decomposition
    - NSGA-III: explicit constraints, reference direction selection

    Args:
        instance: Problem instance.
        surrogate: Trained GNN surrogate with instance already set.
        config: SA-MOEA configuration.
        seed: Random seed.
        verbose: Print progress.
        ref_point: Reference point for per-generation HV tracking.
                   If None, HV is not computed (hv stays 0.0 in stats).

    Returns:
        SAMOEAResult with Pareto solutions and diagnostics.
    """
    if config is None:
        config = SAMOEAConfig()

    m, n = instance.m, instance.n
    rng = np.random.default_rng(seed)
    use_penalty = config.base_algorithm == "moead"
    algo_label = {
        "nsga2": "SA-NSGA-II",
        "moead": "SA-MOEA/D",
        "nsga3": "SA-NSGA-III",
    }.get(config.base_algorithm, "SA-MOEA")

    # Create base algorithm, problem, and determine effective pop size
    algorithm, problem, effective_pop_size = _create_base_algorithm(config, instance)

    # Initialize algorithm
    algorithm.setup(problem, seed=seed, termination=("n_gen", config.n_generations))

    n_true_total = 0
    n_surr_total = 0
    gen_stats = []
    fronts_history: list[tuple[int, np.ndarray]] = []
    n_select = max(1, int(effective_pop_size * config.correction_fraction))

    if verbose:
        print(f"[{algo_label}] Problem: m={m}, n={n} ({m*n} vars)")
        print(f"[{algo_label}] Base algorithm: {config.base_algorithm.upper()}")
        print(f"[{algo_label}] Pop={effective_pop_size}, Gen={config.n_generations}")
        print(f"[{algo_label}] Correction fraction={config.correction_fraction} "
              f"({n_select}/{effective_pop_size})")
        print(f"[{algo_label}] Warmup={config.warmup_generations}, "
              f"Full eval every {config.full_eval_interval} gens")

    t0 = time.time()
    gen = 0

    while algorithm.has_next():
        # Ask for the next population to evaluate
        pop = algorithm.ask()
        X = pop.get("X")

        # Ensure X is 2D - MOEA/D may return 1D (single offspring per subproblem)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        pop_size_actual = len(X)

        # Determine evaluation mode
        is_warmup = gen < config.warmup_generations
        is_full_eval = (
            not is_warmup
            and config.full_eval_interval > 0
            and gen % config.full_eval_interval == 0
        )

        # Skip pre-screening when candidate count <= n_select (nothing to filter)
        skip_prescreening = pop_size_actual <= n_select

        if is_warmup or is_full_eval or skip_prescreening:
            # Full true evaluation - all solutions
            mode = "warmup" if is_warmup else ("full_eval" if is_full_eval else "direct")
            F = _evaluate_true_batch(X, instance)
            n_true = pop_size_actual
            n_surr = 0

            # MOEA/D yields Individual (not Population): squeeze to 1D so
            # off.F stays (n_obj,) and PBI decomposition works correctly.
            if pop_size_actual == 1:
                F = F.squeeze(0)  # (1, n_obj) → (n_obj,)

            if use_penalty:
                penalties = _compute_penalty_batch(X, instance)
                if pop_size_actual == 1:
                    F_penalized = F + config.penalty_weight * penalties[0]
                else:
                    F_penalized = F + config.penalty_weight * penalties[:, np.newaxis]
                pop.set("F", F_penalized)
            else:
                G = _compute_constraints_batch(X, instance)
                if pop_size_actual == 1:
                    G = G.squeeze(0)
                pop.set("F", F)
                pop.set("G", G)

            algorithm.tell(infills=pop)
        else:
            # Pre-screening mode: surrogate filters, only best are truly evaluated
            mode = "prescreened"

            # Step 1: GNN predicts objectives for all
            profits_surr, surpluses_surr, shortfalls_surr = surrogate.evaluate_batch(X)
            F_surr = np.column_stack([-profits_surr, surpluses_surr, shortfalls_surr])
            n_surr = pop_size_actual

            # Step 2: Select top candidates for true evaluation
            selected = _select_for_true_eval(
                F_surr, n_select, config.random_exploration_ratio, rng,
            )

            # Step 3: True evaluation for selected only
            X_sel = X[selected]
            F_sel = _evaluate_true_batch(X_sel, instance)
            n_true = len(selected)

            # Step 4: Create a filtered population with only truly-evaluated
            # solutions. MOEA will merge parents + infills and do selection.
            infills = Population.new("X", X_sel)

            if use_penalty:
                penalties = _compute_penalty_batch(X_sel, instance)
                F_penalized = F_sel + config.penalty_weight * penalties[:, np.newaxis]
                infills.set("F", F_penalized)
            else:
                G_sel = _compute_constraints_batch(X_sel, instance)
                infills.set("F", F_sel)
                infills.set("G", G_sel)

            algorithm.tell(infills=infills)

        n_true_total += n_true
        n_surr_total += n_surr

        # Track stats and store per-generation front for post-hoc HV
        hv_val = 0.0
        cur_opt = algorithm.opt
        if cur_opt is not None and len(cur_opt) > 0:
            F_opt = cur_opt.get("F")
            if F_opt is not None and len(F_opt) > 0:
                fronts_history.append((gen, F_opt.copy()))
                if ref_point is not None:
                    hv_val = compute_hypervolume(F_opt, ref_point)

        gen_stats.append(SAMOEAStats(
            generation=gen,
            mode=mode,
            n_true_evals=n_true,
            n_surr_evals=n_surr,
            hv=hv_val,
        ))

        if verbose and (gen % 50 == 0 or gen == config.n_generations - 1):
            print(f"  Gen {gen:3d}/{config.n_generations}: "
                  f"mode={mode:12s}, true={n_true:3d}, surr={n_surr:3d}, "
                  f"total_true={n_true_total}")

        gen += 1

    # Final: re-evaluate entire final population with true simulator
    # (always use clean F, even for MOEA/D, so Pareto solutions have true values)
    opt = algorithm.opt
    if opt is not None and len(opt) > 0:
        X_final = opt.get("X")
        F_final = _evaluate_true_batch(X_final, instance)
        n_true_total += len(X_final)

        # Extract Pareto solutions
        pareto_solutions = []
        for i in range(len(F_final)):
            x = X_final[i]
            W = x.reshape(m, n)
            pareto_solutions.append(ParetoSolution(
                neg_profit=float(F_final[i, 0]),
                surplus=float(F_final[i, 1]),
                shortfall=float(F_final[i, 2]),
                W=W.copy(),
                decision_vector=x.copy(),
            ))
        pareto_solutions.sort(key=lambda s: s.profit, reverse=True)
    else:
        pareto_solutions = []

    elapsed = time.time() - t0

    if verbose:
        print(f"\n[{algo_label}] Completed in {elapsed:.1f}s")
        print(f"[{algo_label}] True evals: {n_true_total}, "
              f"Surrogate evals: {n_surr_total}")
        savings_pct = (
            n_surr_total / (n_true_total + n_surr_total) * 100
            if (n_true_total + n_surr_total) > 0
            else 0
        )
        print(f"[{algo_label}] Eval savings: {savings_pct:.1f}%")
        print(f"[{algo_label}] Pareto solutions: {len(pareto_solutions)}")
        if pareto_solutions:
            profits = [s.profit for s in pareto_solutions]
            surpluses = [s.surplus for s in pareto_solutions]
            shortfalls = [s.shortfall for s in pareto_solutions]
            print(f"[{algo_label}] Profit range: [{min(profits):.0f}, {max(profits):.0f}] NTD")
            print(f"[{algo_label}] Surplus range: [{min(surpluses):.0f}, {max(surpluses):.0f}] kWh")
            print(f"[{algo_label}] Shortfall range: [{min(shortfalls):.0f}, {max(shortfalls):.0f}] kWh")

    return SAMOEAResult(
        pareto_solutions=pareto_solutions,
        n_generations=config.n_generations,
        n_true_evaluations=n_true_total,
        n_surrogate_evaluations=n_surr_total,
        elapsed_seconds=elapsed,
        base_algorithm=config.base_algorithm,
        gen_stats=gen_stats,
        fronts_history=fronts_history,
    )
