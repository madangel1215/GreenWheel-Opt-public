"""
Generation and demand profile generators for synthetic wheeling instances.

Solar: pvlib clear-sky + Beta-distributed cloud cover
Wind: Weibull speeds with AR(1) autocorrelation + cubic power curve
Demand: IEEE RTS-96-inspired hourly shapes, interpolated to 15-min
"""

import numpy as np
import pandas as pd
from scipy.stats import weibull_min, norm

# ---------------------------------------------------------------------------
# Demand profile templates (normalized hourly shapes, 24 values, peak = 1.0)
# Based on IEEE RTS-96 load shapes adapted for Taiwan corporate loads
# ---------------------------------------------------------------------------

DEMAND_SHAPES = {
    # Industrial: flat, high base load, slight daytime peak
    "industrial": np.array([
        0.80, 0.78, 0.76, 0.75, 0.76, 0.80, 0.88, 0.95,
        1.00, 1.00, 0.98, 0.97, 0.95, 0.97, 0.98, 1.00,
        0.98, 0.95, 0.90, 0.88, 0.85, 0.84, 0.83, 0.82,
    ]),
    # Commercial: strong daytime peak, low nighttime
    "commercial": np.array([
        0.45, 0.42, 0.40, 0.38, 0.38, 0.42, 0.55, 0.75,
        0.92, 0.98, 1.00, 0.99, 0.95, 0.97, 0.98, 0.99,
        0.97, 0.90, 0.78, 0.65, 0.55, 0.50, 0.48, 0.46,
    ]),
    # High-tech fab: very flat 24/7 operation
    "hightech": np.array([
        0.95, 0.94, 0.93, 0.93, 0.94, 0.95, 0.97, 0.98,
        1.00, 1.00, 1.00, 0.99, 0.99, 0.99, 1.00, 1.00,
        0.99, 0.98, 0.97, 0.96, 0.96, 0.96, 0.95, 0.95,
    ]),
}

# Weekend multiplier (lower demand on weekends)
WEEKEND_MULTIPLIER = 0.85

def generate_solar_profiles(
    n_plants: int,
    capacities_kw: np.ndarray,
    n_days: int = 30,
    latitude: float = 24.0,
    longitude: float = 120.5,
    cloud_beta_a: float = 2.0,
    cloud_beta_b: float = 5.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate 15-min solar generation profiles using pvlib clear-sky model.

    Args:
        n_plants: Number of solar plants.
        capacities_kw: Nameplate capacity per plant (kW), shape (n_plants,).
        n_days: Number of days in billing period.
        latitude: Site latitude (default: central Taiwan).
        longitude: Site longitude.
        cloud_beta_a: Beta distribution alpha for cloud cover.
        cloud_beta_b: Beta distribution beta for cloud cover.
        seed: Random seed.

    Returns:
        Array of shape (n_plants, T) with generation in kWh per 15-min interval.
    """
    try:
        from pvlib.location import Location
    except ImportError:
        return _generate_solar_simple(n_plants, capacities_kw, n_days, seed)

    rng = np.random.default_rng(seed)
    T = 96 * n_days  # 15-min intervals

    # Create time index (use a representative summer month for Taiwan)
    times = pd.date_range(
        start="2025-07-01",
        periods=T,
        freq="15min",
        tz="Asia/Taipei",
    )

    # Get clear-sky GHI from pvlib
    location = Location(latitude=latitude, longitude=longitude, tz="Asia/Taipei")
    clearsky = location.get_clearsky(times, model="ineichen")
    ghi_clearsky = clearsky["ghi"].values  # W/m²

    profiles = np.zeros((n_plants, T))
    for i in range(n_plants):
        # Per-plant cloud cover: AR(1) process with Beta marginals
        cloud = _ar1_beta(T, cloud_beta_a, cloud_beta_b, alpha=0.9, rng=rng)
        ghi_actual = ghi_clearsky * (1.0 - cloud * 0.8)

        # Convert GHI (W/m²) to power (kW): simplified P = GHI/1000 * cap
        power_kw = np.maximum(0, ghi_actual / 1000.0 * capacities_kw[i])

        # Convert power to energy per 15-min interval (kWh)
        profiles[i] = power_kw * 0.25

    return profiles

def _generate_solar_simple(
    n_plants: int,
    capacities_kw: np.ndarray,
    n_days: int,
    seed: int,
) -> np.ndarray:
    """Fallback solar profile without pvlib (simple sine model)."""
    rng = np.random.default_rng(seed)
    T = 96 * n_days
    intervals_per_day = 96
    profiles = np.zeros((n_plants, T))

    for i in range(n_plants):
        for d in range(n_days):
            for t in range(intervals_per_day):
                hour = t * 0.25
                if 6.0 <= hour <= 18.0:
                    solar_frac = np.sin(np.pi * (hour - 6.0) / 12.0)
                    cloud = rng.beta(2, 5)
                    power_kw = capacities_kw[i] * solar_frac * (1 - cloud * 0.6)
                    profiles[i, d * intervals_per_day + t] = max(0, power_kw * 0.25)
    return profiles

def generate_wind_profiles(
    n_plants: int,
    capacities_kw: np.ndarray,
    n_days: int = 30,
    weibull_k: float = 2.0,
    weibull_c_range: tuple[float, float] = (7.0, 9.0),
    cut_in: float = 3.0,
    rated_speed: float = 12.0,
    cut_out: float = 25.0,
    autocorr: float = 0.95,
    seed: int = 42,
) -> np.ndarray:
    """Generate 15-min wind generation profiles.

    Uses Weibull wind speed distribution with AR(1) temporal correlation
    and a cubic power curve.

    Args:
        n_plants: Number of wind plants.
        capacities_kw: Nameplate capacity per plant (kW), shape (n_plants,).
        n_days: Days in billing period.
        weibull_k: Weibull shape parameter (k=2 ≈ Rayleigh).
        weibull_c_range: Weibull scale parameter range (m/s).
        cut_in: Cut-in wind speed (m/s).
        rated_speed: Rated wind speed (m/s).
        cut_out: Cut-out wind speed (m/s).
        autocorr: AR(1) autocorrelation coefficient.
        seed: Random seed.

    Returns:
        Array of shape (n_plants, T) with generation in kWh per 15-min interval.
    """
    rng = np.random.default_rng(seed)
    T = 96 * n_days
    profiles = np.zeros((n_plants, T))

    for i in range(n_plants):
        # Per-plant Weibull scale parameter
        c = rng.uniform(*weibull_c_range)

        # Generate correlated wind speeds via Gaussian copula
        # 1. AR(1) Gaussian process
        z = np.zeros(T)
        z[0] = rng.standard_normal()
        innovation_std = np.sqrt(1.0 - autocorr**2)
        for t in range(1, T):
            z[t] = autocorr * z[t - 1] + innovation_std * rng.standard_normal()

        # 2. Transform to Weibull marginals via probability integral transform
        u = norm.cdf(z)  # uniform [0, 1]
        u = np.clip(u, 1e-6, 1.0 - 1e-6)
        wind_speeds = weibull_min.ppf(u, c=weibull_k, scale=c)

        # 3. Apply power curve
        power_frac = np.zeros(T)
        operating = (wind_speeds >= cut_in) & (wind_speeds <= cut_out)
        below_rated = operating & (wind_speeds < rated_speed)
        at_rated = operating & (wind_speeds >= rated_speed)

        power_frac[below_rated] = (
            (wind_speeds[below_rated] - cut_in) / (rated_speed - cut_in)
        ) ** 3
        power_frac[at_rated] = 1.0

        # Convert to energy (kWh per 15-min interval)
        profiles[i] = power_frac * capacities_kw[i] * 0.25

    return profiles

def generate_demand_profiles(
    n_consumers: int,
    peak_demands_kw: np.ndarray,
    consumer_types: list[str],
    n_days: int = 30,
    noise_std: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """Generate 15-min demand profiles based on IEEE RTS-96 load shapes.

    Args:
        n_consumers: Number of consumer sites.
        peak_demands_kw: Peak demand per consumer (kW), shape (n_consumers,).
        consumer_types: Load shape type per consumer ('industrial', 'commercial', 'hightech').
        n_days: Days in billing period.
        noise_std: Gaussian noise standard deviation (fraction of profile value).
        seed: Random seed.

    Returns:
        Array of shape (n_consumers, T) with demand in kWh per 15-min interval.
    """
    rng = np.random.default_rng(seed)
    T = 96 * n_days
    intervals_per_day = 96
    profiles = np.zeros((n_consumers, T))

    for j in range(n_consumers):
        ctype = consumer_types[j]
        hourly_shape = DEMAND_SHAPES.get(ctype, DEMAND_SHAPES["commercial"])

        # Interpolate hourly shape to 15-min resolution
        hours = np.arange(24)
        minutes_15 = np.arange(96) * 0.25
        shape_15min = np.interp(minutes_15, hours, hourly_shape, period=24)

        for d in range(n_days):
            day_of_week = d % 7  # 0=Mon, 6=Sun
            is_weekend = day_of_week >= 5

            # Apply weekend multiplier
            mult = WEEKEND_MULTIPLIER if is_weekend else 1.0

            # Base profile + noise
            base = shape_15min * peak_demands_kw[j] * mult
            noise = rng.normal(0, noise_std, intervals_per_day) * base
            demand_kw = np.maximum(0, base + noise)

            # Convert power (kW) to energy per 15-min (kWh)
            start_idx = d * intervals_per_day
            profiles[j, start_idx : start_idx + intervals_per_day] = demand_kw * 0.25

    return profiles

def _ar1_beta(
    n: int,
    a: float,
    b: float,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate AR(1)-correlated samples with Beta(a, b) marginals.

    Uses Gaussian copula: AR(1) Gaussian → uniform → Beta quantile.
    """
    z = np.zeros(n)
    z[0] = rng.standard_normal()
    innovation_std = np.sqrt(1.0 - alpha**2)
    for t in range(1, n):
        z[t] = alpha * z[t - 1] + innovation_std * rng.standard_normal()

    u = norm.cdf(z)
    u = np.clip(u, 1e-6, 1.0 - 1e-6)

    from scipy.stats import beta as beta_dist

    return beta_dist.ppf(u, a, b)
