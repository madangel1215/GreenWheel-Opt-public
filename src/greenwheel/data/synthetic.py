"""
Synthetic instance generator for wheeling allocation optimization.

Creates a WheelingInstance with randomly parameterized generators
and consumers, using profile generators from profiles.py.
"""

import numpy as np

from src.greenwheel.config import (
    ConsumerSite,
    GeneratorPlant,
    MarketParams,
    WheelingConfig,
    WheelingInstance,
)
from src.greenwheel.data.profiles import (
    generate_demand_profiles,
    generate_solar_profiles,
    generate_wind_profiles,
)
from src.greenwheel.data.real import build_generation_profiles, load_taipower

def generate_instance(
    config: WheelingConfig | None = None,
    seed: int = 42,
) -> WheelingInstance:
    """Generate a synthetic wheeling allocation instance.

    Args:
        config: Configuration (uses defaults if None).
        seed: Random seed for reproducibility.

    Returns:
        A WheelingInstance ready for optimization.
    """
    if config is None:
        config = WheelingConfig()

    rng = np.random.default_rng(seed)
    m = config.n_generators
    n = config.n_consumers
    n_days = config.n_days
    T = 96 * n_days

    # --- Generators ---
    n_solar = max(1, int(m * config.solar_fraction))
    n_wind = m - n_solar

    # Solar plants
    solar_caps = rng.uniform(*config.solar_capacity_range_kw, size=n_solar)
    solar_ppas = rng.uniform(*config.solar_ppa_range, size=n_solar)
    solar_profiles = generate_solar_profiles(
        n_plants=n_solar,
        capacities_kw=solar_caps,
        n_days=n_days,
        latitude=config.latitude,
        longitude=config.longitude,
        cloud_beta_a=config.cloud_beta_a,
        cloud_beta_b=config.cloud_beta_b,
        seed=seed,
    )

    # Wind plants
    wind_caps = rng.uniform(*config.wind_capacity_range_kw, size=n_wind) if n_wind > 0 else np.array([])
    wind_ppas = rng.uniform(*config.wind_ppa_range, size=n_wind) if n_wind > 0 else np.array([])
    wind_profiles = (
        generate_wind_profiles(
            n_plants=n_wind,
            capacities_kw=wind_caps,
            n_days=n_days,
            weibull_k=config.weibull_k,
            weibull_c_range=config.weibull_c_range,
            autocorr=config.wind_autocorr,
            seed=seed + 1000,
        )
        if n_wind > 0
        else np.zeros((0, T))
    )

    generators = []
    for i in range(n_solar):
        generators.append(
            GeneratorPlant(
                id=i,
                capacity_kw=float(solar_caps[i]),
                technology="solar",
                ppa_price=float(solar_ppas[i]),
                generation=solar_profiles[i],
            )
        )
    for i in range(n_wind):
        generators.append(
            GeneratorPlant(
                id=n_solar + i,
                capacity_kw=float(wind_caps[i]),
                technology="wind",
                ppa_price=float(wind_ppas[i]),
                generation=wind_profiles[i],
            )
        )

    # --- Consumers ---
    peak_demands = rng.uniform(*config.demand_peak_range_kw, size=n)
    contract_coverages = rng.uniform(*config.contract_coverage_range, size=n)
    re100_targets = rng.uniform(*config.re100_target_range, size=n)
    retail_prices = rng.uniform(*config.retail_price_range, size=n)

    # Assign consumer types (proportional to typical Taiwan corporate mix)
    type_choices = ["commercial", "industrial", "hightech"]
    type_weights = [0.5, 0.3, 0.2]
    consumer_types = rng.choice(type_choices, size=n, p=type_weights).tolist()

    demand_profiles = generate_demand_profiles(
        n_consumers=n,
        peak_demands_kw=peak_demands,
        consumer_types=consumer_types,
        n_days=n_days,
        seed=seed + 2000,
    )

    consumers = []
    for j in range(n):
        total_monthly = float(demand_profiles[j].sum())
        contract_kwh = total_monthly * float(contract_coverages[j])
        consumers.append(
            ConsumerSite(
                id=j,
                total_monthly_kwh=total_monthly,
                contract_kwh=contract_kwh,
                re100_target=float(re100_targets[j]),
                retail_price=float(retail_prices[j]),
                demand=demand_profiles[j],
                consumer_type=consumer_types[j],
            )
        )

    market = MarketParams(
        surplus_price=config.surplus_price,
        wheeling_fee=config.wheeling_fee,
        fairness_tolerance=config.fairness_tolerance,
        re100_penalty_weight=config.re100_penalty_weight,
    )

    return WheelingInstance(
        generators=generators,
        consumers=consumers,
        market=market,
        n_intervals=T,
    )

def generate_semireal_instance(
    config: WheelingConfig | None = None,
    seed: int = 42,
) -> WheelingInstance:
    """Generate a semi-real wheeling instance: real generation + synthetic demand.

    Uses real Taipower solar/wind generation profiles from dataset #37331
    combined with synthetic IEEE RTS-96 demand profiles.

    Args:
        config: Configuration with data_source fields set.
        seed: Random seed for reproducibility.

    Returns:
        A WheelingInstance with real generation and synthetic demand.
    """
    if config is None:
        config = WheelingConfig()

    rng = np.random.default_rng(seed)
    m = config.n_generators
    n = config.n_consumers
    n_days = config.n_days
    T = 96 * n_days

    # --- Real generation profiles ---
    n_solar = max(1, int(m * config.solar_fraction))
    n_wind = m - n_solar

    taipower_data = load_taipower(config.taipower_json_path)

    solar_profiles, wind_profiles, solar_caps_kw, wind_caps_kw, _meta = (
        build_generation_profiles(
            data=taipower_data,
            n_solar=n_solar,
            n_wind=n_wind,
            n_days=n_days,
            month=config.taipower_month,
            seed=seed,
        )
    )

    # PPA prices: sample from config ranges (real PPA data not public)
    solar_ppas = rng.uniform(*config.solar_ppa_range, size=n_solar)
    wind_ppas = rng.uniform(*config.wind_ppa_range, size=n_wind) if n_wind > 0 else np.array([])

    generators = []
    for i in range(n_solar):
        cap = solar_caps_kw[i] if config.use_real_capacities else rng.uniform(*config.solar_capacity_range_kw)
        generators.append(
            GeneratorPlant(
                id=i,
                capacity_kw=float(cap),
                technology="solar",
                ppa_price=float(solar_ppas[i]),
                generation=solar_profiles[i],
            )
        )
    for i in range(n_wind):
        cap = wind_caps_kw[i] if config.use_real_capacities else rng.uniform(*config.wind_capacity_range_kw)
        generators.append(
            GeneratorPlant(
                id=n_solar + i,
                capacity_kw=float(cap),
                technology="wind",
                ppa_price=float(wind_ppas[i]),
                generation=wind_profiles[i],
            )
        )

    # --- Synthetic demand (unchanged from generate_instance) ---
    peak_demands = rng.uniform(*config.demand_peak_range_kw, size=n)
    contract_coverages = rng.uniform(*config.contract_coverage_range, size=n)
    re100_targets = rng.uniform(*config.re100_target_range, size=n)
    retail_prices = rng.uniform(*config.retail_price_range, size=n)

    type_choices = ["commercial", "industrial", "hightech"]
    type_weights = [0.5, 0.3, 0.2]
    consumer_types = rng.choice(type_choices, size=n, p=type_weights).tolist()

    demand_profiles = generate_demand_profiles(
        n_consumers=n,
        peak_demands_kw=peak_demands,
        consumer_types=consumer_types,
        n_days=n_days,
        seed=seed + 2000,
    )

    consumers = []
    for j in range(n):
        total_monthly = float(demand_profiles[j].sum())
        contract_kwh = total_monthly * float(contract_coverages[j])
        consumers.append(
            ConsumerSite(
                id=j,
                total_monthly_kwh=total_monthly,
                contract_kwh=contract_kwh,
                re100_target=float(re100_targets[j]),
                retail_price=float(retail_prices[j]),
                demand=demand_profiles[j],
                consumer_type=consumer_types[j],
            )
        )

    market = MarketParams(
        surplus_price=config.surplus_price,
        wheeling_fee=config.wheeling_fee,
        fairness_tolerance=config.fairness_tolerance,
        re100_penalty_weight=config.re100_penalty_weight,
    )

    return WheelingInstance(
        generators=generators,
        consumers=consumers,
        market=market,
        n_intervals=T,
    )
