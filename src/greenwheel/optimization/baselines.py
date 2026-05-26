"""
Baseline allocation strategies for comparison with NSGA-II.

Implements four heuristic methods:
1. Random: uniform random allocation (lower bound)
2. Proportional: allocate by contract proportion (industry standard)
3. Greedy: maximize profit only (single-objective upper bound)
4. ITRI-GA: hierarchical GA mimicking ITRI 2023 approach
"""

import numpy as np
from dataclasses import dataclass

from src.greenwheel.config import WheelingInstance
from src.greenwheel.simulator.matching import simulate_matching, MatchingResult

@dataclass
class BaselineResult:
    """Result from a baseline allocation strategy."""

    name: str
    W: np.ndarray              # allocation matrix (m, n)
    matching: MatchingResult   # full simulation result
    elapsed_seconds: float = 0.0
    extra: dict | None = None  # method-specific info

def random_allocation(
    instance: WheelingInstance,
    seed: int = 42,
) -> BaselineResult:
    """Random uniform allocation normalized per generator.

    Each generator's row is sampled uniformly and normalized to sum ≤ 1.
    This is the lower-bound baseline: any reasonable method should beat this.
    """
    import time

    t0 = time.time()
    rng = np.random.default_rng(seed)
    m, n = instance.m, instance.n

    W = rng.uniform(0, 1, size=(m, n))
    # Normalize rows to sum to 1 (fully allocate each generator)
    W /= W.sum(axis=1, keepdims=True)

    result = simulate_matching(W, instance)
    elapsed = time.time() - t0

    return BaselineResult(
        name="Random",
        W=W.copy(),
        matching=result,
        elapsed_seconds=elapsed,
    )

def proportional_allocation(
    instance: WheelingInstance,
    scale: float = 0.95,
) -> BaselineResult:
    """Proportional allocation by contract volume.

    Each generator allocates to consumers proportionally to their
    contracted volumes: w_{ij} = scale * π_j where π_j = D_j^contract / Σ D^contract.

    This is the industry-standard simple method.
    """
    import time

    t0 = time.time()
    m = instance.m
    pi = instance.target_proportions  # (n,)

    # Each generator allocates identically by contract proportion
    W = np.tile(pi, (m, 1)) * scale

    result = simulate_matching(W, instance)
    elapsed = time.time() - t0

    return BaselineResult(
        name="Proportional",
        W=W.copy(),
        matching=result,
        elapsed_seconds=elapsed,
    )

def greedy_allocation(
    instance: WheelingInstance,
    n_restarts: int = 50,
    seed: int = 42,
) -> BaselineResult:
    """Greedy allocation maximizing profit only.

    Uses multi-start local search: generate random allocations,
    then greedily adjust weights to improve profit.
    This represents the single-objective upper bound for profit.
    """
    import time

    t0 = time.time()
    rng = np.random.default_rng(seed)
    m, n = instance.m, instance.n

    best_W = np.zeros((m, n))
    best_profit = -np.inf

    for _ in range(n_restarts):
        # Start from random allocation
        W = rng.uniform(0, 1, size=(m, n))
        W /= W.sum(axis=1, keepdims=True)

        # Greedy improvement: for each generator, shift weight
        # toward the highest-paying consumer
        retail_prices = np.array([c.retail_price for c in instance.consumers])
        price_order = np.argsort(-retail_prices)  # highest price first

        for i in range(m):
            # Shift allocation toward high-price consumers
            # while respecting row sum ≤ 1
            improved = True
            step = 0.05
            while improved:
                improved = False
                current_result = simulate_matching(W, instance)
                current_profit = current_result.profit

                for j_high in price_order[:3]:  # top 3 consumers
                    for j_low in price_order[-3:]:  # bottom 3 consumers
                        if W[i, j_low] < step:
                            continue
                        W_trial = W.copy()
                        W_trial[i, j_low] -= step
                        W_trial[i, j_high] += step
                        trial_result = simulate_matching(W_trial, instance)
                        if trial_result.profit > current_profit:
                            W = W_trial
                            current_profit = trial_result.profit
                            improved = True
                            break
                    if improved:
                        break

        result = simulate_matching(W, instance)
        if result.profit > best_profit:
            best_profit = result.profit
            best_W = W.copy()

    final_result = simulate_matching(best_W, instance)
    elapsed = time.time() - t0

    return BaselineResult(
        name="Greedy",
        W=best_W.copy(),
        matching=final_result,
        elapsed_seconds=elapsed,
    )

def itri_ga_allocation(
    instance: WheelingInstance,
    pop_size: int = 100,
    n_generations: int = 200,
    seed: int = 42,
    verbose: bool = False,
) -> BaselineResult:
    """ITRI 2023-style GA with hierarchical (lexicographic) fitness.

    Mimics the ITRI approach: basic GA with three-tier fitness:
      Tier 1 (highest): Maximize number of consumers meeting minimum transfer
      Tier 2: Maximize RE100 achievement rate
      Tier 3 (lowest): Maximize retailer profit

    This is NOT Pareto optimization - it's lexicographic (hierarchical).
    The key difference from NSGA-II is that lower tiers are only
    considered when higher tiers are tied.

    Reference: Hong et al. (2023), ITRI ICT Journal.
    """
    import time

    t0 = time.time()
    rng = np.random.default_rng(seed)
    m, n = instance.m, instance.n

    # --- Initialize population ---
    pop = []
    for _ in range(pop_size):
        W = rng.uniform(0, 1, size=(m, n))
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        W /= row_sums
        pop.append(W)

    def evaluate(W: np.ndarray) -> tuple[float, float, float]:
        """Compute hierarchical fitness (all to be maximized)."""
        result = simulate_matching(W, instance)

        # Tier 1: fraction of consumers with delivered > 50% of contract
        contracts = np.array([c.contract_kwh for c in instance.consumers])
        delivered = np.array([
            result.re100_rates[j] * instance.consumers[j].total_monthly_kwh
            for j in range(n)
        ])
        tier1 = float(np.mean(delivered >= 0.5 * contracts))

        # Tier 2: average RE100 achievement rate
        tier2 = float(result.re100_rates.mean())

        # Tier 3: normalized profit (positive is better)
        tier3 = result.profit

        return (tier1, tier2, tier3)

    def lexicographic_compare(fit_a, fit_b) -> bool:
        """Return True if a is better than b (lexicographic)."""
        eps = 1e-6
        # Tier 1
        if fit_a[0] > fit_b[0] + eps:
            return True
        if fit_b[0] > fit_a[0] + eps:
            return False
        # Tier 2
        if fit_a[1] > fit_b[1] + eps:
            return True
        if fit_b[1] > fit_a[1] + eps:
            return False
        # Tier 3
        return fit_a[2] > fit_b[2]

    # --- Evolution loop ---
    fitness = [evaluate(W) for W in pop]

    for gen in range(n_generations):
        # Tournament selection
        offspring = []
        for _ in range(pop_size):
            i1, i2 = rng.choice(pop_size, size=2, replace=False)
            parent = pop[i1] if lexicographic_compare(fitness[i1], fitness[i2]) else pop[i2]
            offspring.append(parent.copy())

        # Crossover (uniform)
        for k in range(0, pop_size - 1, 2):
            if rng.random() < 0.9:
                mask = rng.random(size=(m, n)) < 0.5
                child1 = np.where(mask, offspring[k], offspring[k + 1])
                child2 = np.where(mask, offspring[k + 1], offspring[k])
                offspring[k] = child1
                offspring[k + 1] = child2

        # Mutation (Gaussian perturbation)
        for k in range(pop_size):
            if rng.random() < 0.1:
                noise = rng.normal(0, 0.05, size=(m, n))
                offspring[k] = np.clip(offspring[k] + noise, 0, 1)

            # Repair: normalize rows
            row_sums = offspring[k].sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            offspring[k] /= row_sums

        # Evaluate offspring
        offspring_fitness = [evaluate(W) for W in offspring]

        # Elitist selection: combine parent + offspring, keep best
        combined = list(zip(pop + offspring, fitness + offspring_fitness))
        combined.sort(key=lambda x: x[1], reverse=True)
        # Lexicographic sort
        combined_sorted = sorted(
            combined,
            key=lambda x: (x[1][0], x[1][1], x[1][2]),
            reverse=True,
        )

        pop = [x[0] for x in combined_sorted[:pop_size]]
        fitness = [x[1] for x in combined_sorted[:pop_size]]

        if verbose and (gen + 1) % 50 == 0:
            best = fitness[0]
            print(f"  [ITRI-GA] Gen {gen+1}: Tier1={best[0]:.2f}, "
                  f"Tier2={best[1]:.3f}, Profit={best[2]:,.0f}")

    # Best solution
    best_W = pop[0]
    final_result = simulate_matching(best_W, instance)
    elapsed = time.time() - t0

    return BaselineResult(
        name="ITRI-GA",
        W=best_W.copy(),
        matching=final_result,
        elapsed_seconds=elapsed,
        extra={
            "final_tier1": fitness[0][0],
            "final_tier2": fitness[0][1],
            "final_tier3": fitness[0][2],
        },
    )

def run_all_baselines(
    instance: WheelingInstance,
    seed: int = 42,
    ga_pop_size: int = 100,
    ga_generations: int = 200,
    verbose: bool = True,
) -> list[BaselineResult]:
    """Run all baseline strategies and return results.

    Args:
        instance: Problem instance.
        seed: Random seed.
        ga_pop_size: Population size for ITRI-GA.
        ga_generations: Number of generations for ITRI-GA.
        verbose: Print progress.

    Returns:
        List of BaselineResult for each method.
    """
    results = []

    if verbose:
        print("\n--- Running baselines ---")

    # 1. Random
    r = random_allocation(instance, seed=seed)
    results.append(r)
    if verbose:
        print(f"  {r.name:20s}: profit={r.matching.profit:>12,.0f} NTD, "
              f"TMI={r.matching.tmi:>8.2f}, time={r.elapsed_seconds:.2f}s")

    # 2. Proportional
    r = proportional_allocation(instance)
    results.append(r)
    if verbose:
        print(f"  {r.name:20s}: profit={r.matching.profit:>12,.0f} NTD, "
              f"TMI={r.matching.tmi:>8.2f}, time={r.elapsed_seconds:.2f}s")

    # 3. Greedy
    r = greedy_allocation(instance, seed=seed)
    results.append(r)
    if verbose:
        print(f"  {r.name:20s}: profit={r.matching.profit:>12,.0f} NTD, "
              f"TMI={r.matching.tmi:>8.2f}, time={r.elapsed_seconds:.2f}s")

    # 4. ITRI-GA
    r = itri_ga_allocation(
        instance,
        pop_size=ga_pop_size,
        n_generations=ga_generations,
        seed=seed,
        verbose=verbose,
    )
    results.append(r)
    if verbose:
        print(f"  {r.name:20s}: profit={r.matching.profit:>12,.0f} NTD, "
              f"TMI={r.matching.tmi:>8.2f}, time={r.elapsed_seconds:.2f}s")

    return results
