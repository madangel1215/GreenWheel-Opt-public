"""
MOEA performance metrics for evaluating Pareto fronts.

Standard indicators used in multi-objective optimization literature:
- Hypervolume (HV): area/volume dominated by the Pareto front
- Inverted Generational Distance (IGD): distance to reference front
- Spacing: uniformity of solution distribution
- Statistical tests: Wilcoxon signed-rank, Vargha-Delaney A12
"""

import numpy as np
from scipy import stats as scipy_stats
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD

def compute_hypervolume(
    pareto_front: np.ndarray,
    ref_point: np.ndarray | None = None,
) -> float:
    """Compute hypervolume indicator.

    Higher is better - measures the volume of objective space
    dominated by the Pareto front.

    Args:
        pareto_front: (n_solutions, n_obj) array of objective values.
                      All objectives are minimized (f1=-profit, f2=TMI).
        ref_point: Reference point (worst-case). Auto-computed if None.

    Returns:
        Hypervolume value.
    """
    if len(pareto_front) == 0:
        return 0.0

    if ref_point is None:
        # Use 110% of the worst value in each objective as reference
        ref_point = pareto_front.max(axis=0) * 1.1

    hv = HV(ref_point=ref_point)
    return float(hv(pareto_front))

def compute_igd(
    pareto_front: np.ndarray,
    reference_front: np.ndarray,
) -> float:
    """Compute Inverted Generational Distance (IGD).

    Lower is better - measures how close the obtained front is
    to the reference (ideal) front.

    Args:
        pareto_front: (n_solutions, n_obj) obtained Pareto front.
        reference_front: (n_ref, n_obj) reference Pareto front.

    Returns:
        IGD value.
    """
    if len(pareto_front) == 0:
        return float("inf")

    igd = IGD(reference_front)
    return float(igd(pareto_front))

def compute_spacing(pareto_front: np.ndarray) -> float:
    """Compute spacing metric (distribution uniformity).

    Lower is better - measures how uniformly solutions are
    distributed along the Pareto front.

    SP = sqrt(1/(N-1) * Σ(d_i - d_mean)²)
    where d_i = min distance from solution i to any other solution.

    Args:
        pareto_front: (n_solutions, n_obj) array.

    Returns:
        Spacing value. 0 means perfectly uniform.
    """
    n = len(pareto_front)
    if n <= 1:
        return 0.0

    # Normalize to [0, 1] for fair distance computation
    f_min = pareto_front.min(axis=0)
    f_max = pareto_front.max(axis=0)
    f_range = f_max - f_min
    f_range[f_range == 0] = 1.0
    norm = (pareto_front - f_min) / f_range

    # Compute minimum distance for each solution
    min_dists = np.zeros(n)
    for i in range(n):
        dists = np.sqrt(np.sum((norm - norm[i]) ** 2, axis=1))
        dists[i] = np.inf  # exclude self
        min_dists[i] = dists.min()

    d_mean = min_dists.mean()
    spacing = float(np.sqrt(np.mean((min_dists - d_mean) ** 2)))
    return spacing

def compute_spread(pareto_front: np.ndarray) -> float:
    """Compute spread (extent) of the Pareto front.

    Higher is better - measures the range of the front in each objective.

    Returns the product of ranges in each normalized objective dimension.
    """
    if len(pareto_front) <= 1:
        return 0.0

    f_min = pareto_front.min(axis=0)
    f_max = pareto_front.max(axis=0)
    ranges = f_max - f_min

    # Return geometric mean of ranges to handle different scales
    nonzero = ranges[ranges > 0]
    if len(nonzero) == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(nonzero))))

def compute_all_metrics(
    pareto_front: np.ndarray,
    ref_point: np.ndarray | None = None,
    reference_front: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute all standard MOEA metrics.

    Args:
        pareto_front: (n_solutions, n_obj) obtained front.
        ref_point: Reference point for hypervolume.
        reference_front: Reference front for IGD (optional).

    Returns:
        Dict with metric names and values.
    """
    metrics = {
        "n_solutions": len(pareto_front),
        "hypervolume": compute_hypervolume(pareto_front, ref_point),
        "spacing": compute_spacing(pareto_front),
        "spread": compute_spread(pareto_front),
    }

    if reference_front is not None and len(reference_front) > 0:
        metrics["igd"] = compute_igd(pareto_front, reference_front)

    return metrics

# ---------------------------------------------------------------------------
# Statistical comparison functions for SCI paper experiments
# ---------------------------------------------------------------------------

def wilcoxon_signed_rank_test(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
) -> dict[str, float]:
    """Paired Wilcoxon signed-rank test (non-parametric).

    Tests H0: median(a - b) = 0 against H1: median(a - b) != 0.
    Appropriate for paired samples (same instance, different methods).

    Args:
        a: Metric values from method A (e.g., vanilla NSGA-II).
        b: Metric values from method B (e.g., SA-MOEA).

    Returns:
        Dict with 'statistic', 'p_value', 'significant' (p < 0.05).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert len(a) == len(b), "Paired test requires equal-length arrays"

    # Handle degenerate case: all differences are zero
    diffs = a - b
    if np.all(diffs == 0):
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}

    result = scipy_stats.wilcoxon(a, b, alternative="two-sided")
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant": float(result.pvalue) < 0.05,
    }

def compute_effect_size_a12(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
) -> dict[str, float | str]:
    """Vargha-Delaney A12 effect size measure.

    A12 = P(a > b) + 0.5 * P(a == b).
    - A12 = 0.5: no effect (methods are equivalent)
    - A12 > 0.5: method A tends to be larger
    - A12 < 0.5: method B tends to be larger

    Magnitude thresholds (Vargha & Delaney 2000):
    - |A12 - 0.5| < 0.06: negligible
    - |A12 - 0.5| < 0.14: small
    - |A12 - 0.5| < 0.21: medium
    - |A12 - 0.5| >= 0.21: large

    Args:
        a: Values from method A.
        b: Values from method B.

    Returns:
        Dict with 'a12' value and 'magnitude' classification.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m, n = len(a), len(b)

    # Count pairwise comparisons
    count = 0.0
    for ai in a:
        for bj in b:
            if ai > bj:
                count += 1.0
            elif ai == bj:
                count += 0.5

    a12 = count / (m * n)

    # Classify magnitude
    delta = abs(a12 - 0.5)
    if delta < 0.06:
        magnitude = "negligible"
    elif delta < 0.14:
        magnitude = "small"
    elif delta < 0.21:
        magnitude = "medium"
    else:
        magnitude = "large"

    return {"a12": float(a12), "magnitude": magnitude}

def run_statistical_comparison(
    results_by_method: dict[str, list[float]],
    metric_name: str = "hypervolume",
    higher_is_better: bool = True,
) -> dict:
    """Run complete statistical comparison between methods.

    Performs pairwise Wilcoxon tests and A12 effect sizes for all
    method pairs. Designed for comparing vanilla NSGA-II vs SA-MOEA
    across multiple seeds on the same instances.

    Args:
        results_by_method: {method_name: [metric_values_per_seed]}.
        metric_name: Name of the metric being compared.
        higher_is_better: Whether higher values are better.

    Returns:
        Dict with descriptive stats and pairwise comparisons.
    """
    methods = list(results_by_method.keys())
    comparison = {
        "metric": metric_name,
        "higher_is_better": higher_is_better,
        "descriptive": {},
        "pairwise": [],
    }

    # Descriptive statistics per method
    for method, values in results_by_method.items():
        arr = np.asarray(values, dtype=float)
        comparison["descriptive"][method] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "median": float(np.median(arr)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n": len(arr),
        }

    # Pairwise comparisons
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            a_name, b_name = methods[i], methods[j]
            a_vals = results_by_method[a_name]
            b_vals = results_by_method[b_name]

            wilcoxon = wilcoxon_signed_rank_test(a_vals, b_vals)
            a12 = compute_effect_size_a12(a_vals, b_vals)

            comparison["pairwise"].append({
                "method_a": a_name,
                "method_b": b_name,
                "wilcoxon": wilcoxon,
                "a12": a12,
            })

    return comparison
