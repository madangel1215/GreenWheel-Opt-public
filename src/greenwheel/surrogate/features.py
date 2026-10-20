"""Feature extraction for GNN surrogate.

Converts WheelingInstance + allocation matrix W into numerical feature tensors
suitable for the GNN model.
"""

import numpy as np
import torch

from src.greenwheel.config import WheelingInstance

def extract_generator_features(instance: WheelingInstance) -> torch.Tensor:
    """Extract generator node features.

    Features (8 per generator):
        0: log(capacity_kw)
        1: ppa_price (NTD/kWh)
        2: is_solar (binary)
        3: capacity_factor [0, 1]
        4: variability (CV)
        5: log(total_gen)
        6: peak_gen (kWh/interval)
        7: daytime_ratio (fraction of gen during 6am-6pm)

    Args:
        instance: Wheeling problem instance.

    Returns:
        Tensor of shape (m, 8), dtype float32.
    """
    feats = []
    for g in instance.generators:
        total_gen = g.generation.sum()
        peak_gen = g.generation.max()

        # Daytime ratio: fraction of generation during hours 6-18
        n_days = instance.n_intervals // 96
        intervals_per_hour = 4
        daytime_mask = np.zeros(instance.n_intervals, dtype=bool)
        for d in range(n_days):
            start = d * 96 + 6 * intervals_per_hour   # 6am
            end = d * 96 + 18 * intervals_per_hour     # 6pm
            daytime_mask[start:end] = True
        daytime_gen = g.generation[daytime_mask].sum()
        daytime_ratio = daytime_gen / total_gen if total_gen > 0 else 0.0

        feats.append([
            np.log1p(g.capacity_kw),
            g.ppa_price,
            1.0 if g.technology == "solar" else 0.0,
            g.capacity_factor,
            g.variability,
            np.log1p(total_gen),
            peak_gen,
            daytime_ratio,
        ])

    return torch.tensor(feats, dtype=torch.float32)

def extract_consumer_features(instance: WheelingInstance) -> torch.Tensor:
    """Extract consumer node features.

    Features (11 per consumer):
        0: log(total_monthly_kwh)
        1: log(contract_kwh)
        2: re100_target [0, 1]
        3: retail_price (NTD/kWh)
        4: load_factor [0, 1]
        5: variability (CV)
        6: contract_coverage [0, 1]
        7-9: consumer_type one-hot (commercial, industrial, hightech)
        10: peak_demand (kWh/interval)

    Args:
        instance: Wheeling problem instance.

    Returns:
        Tensor of shape (n, 11), dtype float32.
    """
    type_map = {"commercial": [1, 0, 0], "industrial": [0, 1, 0], "hightech": [0, 0, 1]}

    feats = []
    for c in instance.consumers:
        one_hot = type_map.get(c.consumer_type, [0, 0, 0])
        feats.append([
            np.log1p(c.total_monthly_kwh),
            np.log1p(c.contract_kwh),
            c.re100_target,
            c.retail_price,
            c.load_factor,
            c.variability,
            c.contract_coverage,
            *one_hot,
            c.demand.max(),
        ])

    return torch.tensor(feats, dtype=torch.float32)

def temporal_pair_features(instance: WheelingInstance) -> np.ndarray:
    """Per-pair temporal coincidence features, independent of W.

    The objectives are settled interval by interval as min(sum_i W_ij G_it, D_jt),
    so what a generator is worth to a consumer depends on whether their profiles
    coincide in time. The monthly summaries in the node features cannot express
    this: on semi-real instances they explain only 87% of the variance in pairwise
    shape overlap, against 99% on synthetic ones. Computed once per instance and cached on it.

    Features (3 per edge, row-major like extract_edge_features):
        0: shape overlap, sum_t min(g_it/sum g_i, d_jt/sum d_j), in [0, 1]
        1: Pearson correlation of G_i and D_j (0 if either is constant)
        2: coverage, sum_t min(G_it, D_jt) / sum_t D_jt, in [0, 1]

    Returns:
        Array of shape (m*n, 3), float32.
    """
    cached = getattr(instance, "_temporal_pair_cache", None)
    if cached is not None:
        return cached

    G = np.array([g.generation for g in instance.generators], dtype=np.float64)
    D = np.array([c.demand for c in instance.consumers], dtype=np.float64)
    m, n = G.shape[0], D.shape[0]

    g_tot = G.sum(axis=1, keepdims=True)
    d_tot = D.sum(axis=1, keepdims=True)
    Gs = np.divide(G, g_tot, out=np.zeros_like(G), where=g_tot > 0)
    Ds = np.divide(D, d_tot, out=np.zeros_like(D), where=d_tot > 0)

    Gc = G - G.mean(axis=1, keepdims=True)
    Dc = D - D.mean(axis=1, keepdims=True)
    g_norm = np.linalg.norm(Gc, axis=1)
    d_norm = np.linalg.norm(Dc, axis=1)
    denom = np.outer(g_norm, d_norm)
    corr = np.divide(Gc @ Dc.T, denom, out=np.zeros((m, n)), where=denom > 0)

    overlap = np.empty((m, n))
    coverage = np.empty((m, n))
    for i in range(m):
        overlap[i] = np.minimum(Gs[i][None, :], Ds).sum(axis=1)
        covered = np.minimum(G[i][None, :], D).sum(axis=1)
        coverage[i] = np.divide(covered, d_tot[:, 0], out=np.zeros(n), where=d_tot[:, 0] > 0)

    feats = np.stack([overlap, corr, coverage], axis=-1).reshape(m * n, 3).astype(np.float32)
    instance._temporal_pair_cache = feats
    return feats

def extract_edge_features(
    W: np.ndarray,
    instance: WheelingInstance,
    temporal: bool = False,
) -> torch.Tensor:
    """Extract edge features for all generator-consumer pairs.

    Features (3 per edge):
        0: w_ij (allocation weight, the decision variable)
        1: price_spread (retail_price_j - ppa_price_i)
        2: gen_demand_ratio (total_gen_i / total_demand_j)

    Edges are ordered row-major: (gen_0, con_0), (gen_0, con_1), ..., (gen_m-1, con_n-1).

    Args:
        W: Allocation matrix shape (m, n).
        instance: Wheeling problem instance.

    Returns:
        Tensor of shape (m*n, 3), dtype float32.
    """
    m, n = instance.m, instance.n
    feats = np.zeros((m * n, 3), dtype=np.float32)

    ppa_prices = np.array([g.ppa_price for g in instance.generators])
    retail_prices = np.array([c.retail_price for c in instance.consumers])
    total_gens = np.array([g.generation.sum() for g in instance.generators])
    total_demands = np.array([c.demand.sum() for c in instance.consumers])

    # Vectorized computation
    idx = 0
    for i in range(m):
        for j in range(n):
            feats[idx, 0] = W[i, j]
            feats[idx, 1] = retail_prices[j] - ppa_prices[i]
            feats[idx, 2] = total_gens[i] / total_demands[j] if total_demands[j] > 0 else 0.0
            idx += 1

    if temporal:
        feats = np.concatenate([feats, temporal_pair_features(instance)], axis=1)

    return torch.tensor(feats, dtype=torch.float32)

def extract_global_features(instance: WheelingInstance) -> torch.Tensor:
    """Extract global (graph-level) features.

    Features (6):
        0: surplus_price
        1: wheeling_fee
        2: re100_penalty_weight
        3: gen_demand_ratio (total gen / total demand)
        4: normalized_m (m / 50, assuming max ~50 generators)
        5: normalized_n (n / 200, assuming max ~200 consumers)

    Args:
        instance: Wheeling problem instance.

    Returns:
        Tensor of shape (6,), dtype float32.
    """
    total_gen = instance.total_generation
    total_dem = instance.total_demand
    gen_dem_ratio = total_gen / total_dem if total_dem > 0 else 0.0

    feats = [
        instance.market.surplus_price,
        instance.market.wheeling_fee,
        instance.market.re100_penalty_weight,
        gen_dem_ratio,
        instance.m / 50.0,
        instance.n / 200.0,
    ]

    return torch.tensor(feats, dtype=torch.float32)
