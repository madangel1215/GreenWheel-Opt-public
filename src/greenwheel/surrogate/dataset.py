"""Training data generation and dataset for GNN surrogate.

Three sampling strategies:
1. LHS random (60%): Latin Hypercube Sampling of W → repair → evaluate
2. NSGA-II snapshots (30%): Collect populations from gen 0/50/100/150/200
3. Pareto perturbations (10%): Gaussian noise on Pareto solutions
"""

import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from torch_geometric.data import InMemoryDataset, Data

from src.greenwheel.config import WheelingConfig, WheelingInstance
from src.greenwheel.data.synthetic import generate_instance, generate_semireal_instance
from src.greenwheel.simulator.matching import simulate_matching
from src.greenwheel.surrogate.graph import instance_to_heterodata

def _repair_allocation(W: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Standalone repair (no pymoo dependency needed for data gen)."""
    W = W.copy()
    W[W < threshold] = 0.0
    row_sums = W.sum(axis=1)
    overallocated = row_sums > 1.0
    if overallocated.any():
        W[overallocated] /= row_sums[overallocated, np.newaxis]
    return W

def _lhs_sample(m: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate one Latin Hypercube Sample for allocation matrix."""
    dim = m * n
    # Stratified sampling per dimension
    result = np.zeros(dim)
    for d in range(dim):
        perm = rng.permutation(1)
        result[d] = (perm[0] + rng.random()) / 1.0
    # Simple approach: uniform random (LHS benefit is minimal for single samples)
    W = rng.uniform(0, 1, size=(m, n))
    return _repair_allocation(W)

def _batch_lhs_samples(
    m: int, n: int, n_samples: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Generate multiple LHS allocation samples."""
    samples = []
    for _ in range(n_samples):
        W = rng.uniform(0, 1, size=(m, n))
        # Random sparsity: zero out some entries
        mask = rng.random(size=(m, n)) < 0.3  # ~30% sparsity
        W[mask] = 0.0
        samples.append(_repair_allocation(W))
    return samples

def evaluate_allocation(
    W: np.ndarray,
    instance: WheelingInstance,
) -> tuple[float, float, float]:
    """Evaluate one allocation and return (profit, surplus, shortfall)."""
    result = simulate_matching(W, instance)
    return result.profit, result.total_surplus, result.total_shortfall

def generate_lhs_samples(
    instance: WheelingInstance,
    n_samples: int,
    seed: int = 42,
) -> list[tuple[np.ndarray, float, float, float]]:
    """Generate LHS random samples with evaluations.

    Returns:
        List of (W, profit, surplus, shortfall) tuples.
    """
    rng = np.random.default_rng(seed)
    m, n = instance.m, instance.n
    allocations = _batch_lhs_samples(m, n, n_samples, rng)

    results = []
    for W in allocations:
        profit, surplus, shortfall = evaluate_allocation(W, instance)
        results.append((W, profit, surplus, shortfall))
    return results

def generate_nsga2_snapshots(
    instance: WheelingInstance,
    pop_size: int = 100,
    n_generations: int = 200,
    snapshot_gens: list[int] | None = None,
    seed: int = 42,
) -> list[tuple[np.ndarray, float, float, float]]:
    """Run NSGA-II and collect population snapshots at specified generations.

    Args:
        instance: Problem instance.
        pop_size: Population size.
        n_generations: Total generations.
        snapshot_gens: Generations to snapshot (default: [0, 50, 100, 150, 200]).
        seed: Random seed.

    Returns:
        List of (W, profit, surplus, shortfall) tuples from all snapshots.
    """
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize
    from pymoo.core.callback import Callback

    from src.greenwheel.problem.formulation import WheelingProblem
    from src.greenwheel.problem.repair import AllocationRepair

    if snapshot_gens is None:
        snapshot_gens = [0, 50, 100, 150, 200]

    class SnapshotCallback(Callback):
        def __init__(self):
            super().__init__()
            self.snapshots = []
            self.target_gens = set(snapshot_gens)

        def notify(self, algorithm):
            gen = algorithm.n_gen
            if gen in self.target_gens:
                pop = algorithm.pop
                for ind in pop:
                    W = ind.X.reshape(instance.m, instance.n)
                    F = ind.F  # (-profit, surplus, shortfall) in pymoo convention
                    self.snapshots.append((W.copy(), -F[0], F[1], F[2]))

    problem = WheelingProblem(instance)
    callback = SnapshotCallback()

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=20),
        mutation=PM(eta=20),
        repair=AllocationRepair(),
    )

    minimize(
        problem,
        algorithm,
        ("n_gen", n_generations),
        seed=seed,
        callback=callback,
        verbose=False,
    )

    # Also add final generation if not in snapshot list
    if n_generations not in snapshot_gens:
        final_pop = algorithm.pop if hasattr(algorithm, "pop") else []
        for ind in final_pop:
            W = ind.X.reshape(instance.m, instance.n)
            F = ind.F
            callback.snapshots.append((W.copy(), -F[0], F[1], F[2]))

    return callback.snapshots

def generate_pareto_perturbations(
    pareto_samples: list[tuple[np.ndarray, float, float, float]],
    instance: WheelingInstance,
    n_perturbations_per: int = 5,
    sigma_range: tuple[float, float] = (0.05, 0.15),
    seed: int = 42,
) -> list[tuple[np.ndarray, float, float, float]]:
    """Generate perturbations around Pareto-optimal solutions.

    Args:
        pareto_samples: List of (W, profit, surplus, shortfall) from Pareto front.
        instance: Problem instance for re-evaluation.
        n_perturbations_per: Number of perturbations per Pareto solution.
        sigma_range: Range of Gaussian noise sigma.
        seed: Random seed.

    Returns:
        List of (W, profit, surplus, shortfall) from perturbations.
    """
    rng = np.random.default_rng(seed)
    results = []

    for W_orig, _, _, _ in pareto_samples:
        for _ in range(n_perturbations_per):
            sigma = rng.uniform(*sigma_range)
            noise = rng.normal(0, sigma, size=W_orig.shape)
            W_perturbed = np.clip(W_orig + noise, 0, 1)
            W_perturbed = _repair_allocation(W_perturbed)
            profit, surplus, shortfall = evaluate_allocation(W_perturbed, instance)
            results.append((W_perturbed, profit, surplus, shortfall))

    return results

def _generate_for_instance(args):
    """Worker function for parallel instance data generation."""
    (config, seed, n_lhs, n_nsga2_per_snapshot, nsga2_pop_size,
     nsga2_gens, snapshot_gens, n_pareto_perturb) = args

    # Generate instance
    if config.data_source == "semireal":
        instance = generate_semireal_instance(config, seed=seed)
    else:
        instance = generate_instance(config, seed=seed)

    samples = []

    # 1. LHS random samples
    lhs = generate_lhs_samples(instance, n_lhs, seed=seed)
    samples.extend(lhs)

    # 2. NSGA-II snapshots
    nsga2 = generate_nsga2_snapshots(
        instance,
        pop_size=nsga2_pop_size,
        n_generations=nsga2_gens,
        snapshot_gens=snapshot_gens,
        seed=seed,
    )
    samples.extend(nsga2)

    # 3. Pareto perturbations (from last snapshot gen)
    if nsga2:
        # Use the last ~pop_size solutions as Pareto candidates
        pareto_candidates = nsga2[-nsga2_pop_size:]
        perturbations = generate_pareto_perturbations(
            pareto_candidates[:20],  # top 20 Pareto solutions
            instance,
            n_perturbations_per=n_pareto_perturb,
            seed=seed + 5000,
        )
        samples.extend(perturbations)

    # Convert to HeteroData
    graphs = []
    for W, profit, surplus, shortfall in samples:
        targets = np.array([profit, surplus, shortfall], dtype=np.float32)
        graph = instance_to_heterodata(instance, W, targets=targets)
        graphs.append(graph)

    return graphs

def generate_training_data(
    config: WheelingConfig,
    n_instances: int = 20,
    samples_per_instance: int = 400,
    seed: int = 42,
    n_workers: int = 1,
) -> list:
    """Generate full training dataset across multiple instances.

    Args:
        config: Wheeling configuration.
        n_instances: Number of problem instances to generate.
        samples_per_instance: Approximate samples per instance.
        seed: Base random seed.
        n_workers: Number of parallel workers.

    Returns:
        List of HeteroData graphs with targets.
    """
    # Split samples: 60% LHS, 30% NSGA-II, 10% Pareto perturbation
    n_lhs = int(samples_per_instance * 0.6)
    nsga2_pop_size = 50
    snapshot_gens = [0, 50, 100, 150, 200]
    # NSGA-II samples = pop_size * len(snapshot_gens)
    n_pareto_perturb = max(1, int(samples_per_instance * 0.10 / 20))

    args_list = [
        (config, seed + i * 100, n_lhs, 0, nsga2_pop_size,
         200, snapshot_gens, n_pareto_perturb)
        for i in range(n_instances)
    ]

    all_graphs = []

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for graphs in executor.map(_generate_for_instance, args_list):
                all_graphs.extend(graphs)
    else:
        for args in args_list:
            graphs = _generate_for_instance(args)
            all_graphs.extend(graphs)

    return all_graphs

class SurrogateDataset(InMemoryDataset):
    """PyG InMemoryDataset wrapping pre-generated surrogate training data.

    Args:
        root: Root directory for dataset storage.
        data_list: Pre-generated list of HeteroData graphs.
        split: 'train', 'val', or 'test'.
    """

    def __init__(
        self,
        root: str | Path,
        data_list: list | None = None,
        split: str = "train",
    ):
        self._data_list_input = data_list
        self._split = split
        super().__init__(str(root))

        path = Path(self.processed_dir) / f"{split}.pt"
        if path.exists():
            loaded = torch.load(path, weights_only=False)
            self.data, self.slices = loaded
        elif data_list is not None:
            self.data, self.slices = self.collate(data_list)
            torch.save((self.data, self.slices), path)

    @property
    def processed_file_names(self) -> list[str]:
        return [f"{self._split}.pt"]

    def process(self):
        pass  # Data is provided externally
