"""
15-minute interval matching simulator for wheeling allocation.

Given an allocation matrix W (m×n) and a WheelingInstance, simulates
the energy delivery over all time intervals and computes:
- Retailer profit (f₁, analytical)
- Temporal Mismatch Index (f₂, simulation-based)

This is the "expensive" evaluation that the GNN surrogate will replace.
"""

import numpy as np
from dataclasses import dataclass

from src.greenwheel.config import WheelingInstance

@dataclass
class MatchingResult:
    """Result of simulating a wheeling allocation over all intervals.

    All energy values in kWh.
    """

    # Per-entity aggregates
    surplus_per_gen: np.ndarray       # (m,) undelivered generation per generator
    over_alloc_per_con: np.ndarray    # (n,) over-allocation per consumer
    shortfall_per_con: np.ndarray     # (n,) RE100 shortfall per consumer
    re100_rates: np.ndarray           # (n,) actual green energy fraction

    # Scalar metrics
    total_delivered: float
    total_surplus: float
    total_shortfall: float

    # Objectives
    profit: float   # retailer profit (to be maximized; f₁ = -profit)
    tmi: float      # temporal mismatch index (to be minimized; f₂)

    # Profit components (for analysis)
    revenue: float
    gen_cost: float
    wheeling_cost: float
    surplus_penalty: float

def simulate_matching(
    W: np.ndarray,
    instance: WheelingInstance,
) -> MatchingResult:
    """Simulate 15-min interval matching for a given allocation matrix.

    The allocation matrix W (m×n) assigns fractions of each generator's
    output to consumers. At each 15-min interval, delivery is capped
    by consumer demand using proportional curtailment.

    Args:
        W: Allocation weight matrix, shape (m, n), values in [0, 1].
        instance: Problem instance with generation/demand profiles.

    Returns:
        MatchingResult with all metrics computed.
    """
    T = instance.n_intervals
    market = instance.market

    # Build generation and demand matrices: (m, T) and (n, T)
    G = np.stack([g.generation for g in instance.generators])  # (m, T)
    D = np.stack([c.demand for c in instance.consumers])       # (n, T)

    # Raw allocation: e_{ij}(t) = w_{ij} * g_i(t)
    # E_raw[i, j, t] = W[i, j] * G[i, t]
    E_raw = W[:, :, np.newaxis] * G[:, np.newaxis, :]  # (m, n, T)

    # Total allocated to each consumer: (n, T)
    total_to_con = E_raw.sum(axis=0)  # (n, T)

    # Proportional capping: if total_to_con > demand, scale down proportionally
    # cap_ratio[j, t] = min(1, d_j(t) / total_to_con[j, t])
    with np.errstate(divide="ignore", invalid="ignore"):
        cap_ratio = np.where(
            total_to_con > 1e-9,
            np.minimum(1.0, D / total_to_con),
            0.0,
        )  # (n, T)

    # Effective delivery per (generator, consumer) pair
    E_eff = E_raw * cap_ratio[np.newaxis, :, :]  # (m, n, T)

    # Effective delivery per consumer: min(total_to_con, D)
    eff_to_con = E_eff.sum(axis=0)  # (n, T)

    # Effective delivery from each generator
    eff_from_gen = E_eff.sum(axis=1)  # (m, T)

    # --- Over-allocation per consumer (surplus at consumer side) ---
    over_alloc = np.maximum(0, total_to_con - D)  # (n, T)

    # --- Generator surplus (unallocated + curtailed) ---
    gen_surplus = G - eff_from_gen  # (m, T), always >= 0 by construction

    # --- RE100 shortfall ---
    re100_targets = np.array([c.re100_target for c in instance.consumers])
    target_green = D * re100_targets[:, np.newaxis]  # (n, T)
    shortfall = np.maximum(0, target_green - eff_to_con)  # (n, T)

    # --- TMI: Temporal Mismatch Index ---
    # f₂ = (1/T) Σ_t Σ_j [ over_alloc_{j,t} + λ * shortfall_{j,t} ]
    lambda_re100 = market.re100_penalty_weight
    tmi = float((over_alloc.sum() + lambda_re100 * shortfall.sum()) / T)

    # --- Profit calculation ---
    retail_prices = np.array([c.retail_price for c in instance.consumers])  # (n,)
    ppa_prices = np.array([g.ppa_price for g in instance.generators])      # (m,)

    # Revenue = Σ_j p_j^retail * Σ_t eff_to_con[j, t]
    revenue = float((retail_prices[:, np.newaxis] * eff_to_con).sum())

    # GenCost = Σ_i p_i^ppa * Σ_t g_i(t)  (fixed cost, independent of W)
    gen_cost = float((ppa_prices[:, np.newaxis] * G).sum())

    # WheelingCost = β * total effective delivery
    wheeling_cost = float(market.wheeling_fee * eff_to_con.sum())

    # SurplusPenalty = p^surplus * total surplus
    total_surplus_val = float(gen_surplus.sum())
    surplus_penalty = float(market.surplus_price * total_surplus_val)

    profit = revenue - gen_cost - wheeling_cost - surplus_penalty

    # --- RE100 achievement rates ---
    total_delivered_per_con = eff_to_con.sum(axis=1)  # (n,)
    total_demand_per_con = D.sum(axis=1)               # (n,)
    re100_rates = np.where(
        total_demand_per_con > 0,
        total_delivered_per_con / total_demand_per_con,
        0.0,
    )

    return MatchingResult(
        surplus_per_gen=gen_surplus.sum(axis=1),
        over_alloc_per_con=over_alloc.sum(axis=1),
        shortfall_per_con=shortfall.sum(axis=1),
        re100_rates=re100_rates,
        total_delivered=float(eff_to_con.sum()),
        total_surplus=total_surplus_val + float(over_alloc.sum()),
        total_shortfall=float(shortfall.sum()),
        profit=profit,
        tmi=tmi,
        revenue=revenue,
        gen_cost=gen_cost,
        wheeling_cost=wheeling_cost,
        surplus_penalty=surplus_penalty,
    )

def compute_profit_only(
    W: np.ndarray,
    instance: WheelingInstance,
) -> float:
    """Compute retailer profit analytically (cheap, ~1ms).

    Same as simulate_matching but only returns profit, slightly faster
    by skipping RE100/TMI computation.
    """
    market = instance.market

    G = np.stack([g.generation for g in instance.generators])
    D = np.stack([c.demand for c in instance.consumers])

    E_raw = W[:, :, np.newaxis] * G[:, np.newaxis, :]
    total_to_con = E_raw.sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        cap_ratio = np.where(
            total_to_con > 1e-9,
            np.minimum(1.0, D / total_to_con),
            0.0,
        )

    E_eff = E_raw * cap_ratio[np.newaxis, :, :]
    eff_to_con = E_eff.sum(axis=0)
    eff_from_gen = E_eff.sum(axis=1)
    gen_surplus = G - eff_from_gen

    retail_prices = np.array([c.retail_price for c in instance.consumers])
    ppa_prices = np.array([g.ppa_price for g in instance.generators])

    revenue = float((retail_prices[:, np.newaxis] * eff_to_con).sum())
    gen_cost = float((ppa_prices[:, np.newaxis] * G).sum())
    wheeling_cost = float(market.wheeling_fee * eff_to_con.sum())
    surplus_penalty = float(market.surplus_price * gen_surplus.sum())

    return revenue - gen_cost - wheeling_cost - surplus_penalty
