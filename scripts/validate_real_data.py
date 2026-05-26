#!/usr/bin/env python3
"""
Validate real Taipower generation data and compare with synthetic profiles.

Produces:
  - Data quality report per plant (completeness, gaps, capacity factors)
  - Capacity factor distributions (real vs synthetic)
  - Daily generation pattern comparison
  - Temporal autocorrelation comparison

Usage:
    python scripts/validate_real_data.py
    python scripts/validate_real_data.py --json data/raw/taipower_37331.json --output results/validation
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.greenwheel.config import WheelingConfig
from src.greenwheel.data.real import load_taipower, build_generation_profiles
from src.greenwheel.data.profiles import generate_solar_profiles, generate_wind_profiles

def data_quality_report(data: dict, output_dir: Path) -> None:
    """Print and save data quality statistics per plant."""
    print("\n" + "=" * 70)
    print("DATA QUALITY REPORT")
    print("=" * 70)

    for tech in ["solar", "wind"]:
        df = data.get(tech)
        if df is None or df.empty:
            print(f"\n  {tech.upper()}: No data")
            continue

        print(f"\n  {tech.upper()} Plants ({len(df.columns)} total)")
        print(f"  {'─' * 60}")
        print(f"  {'Plant':<20s} {'Days':>5s} {'Complete%':>10s} {'MaxGap(h)':>10s} {'MeanMW':>8s} {'PeakMW':>8s} {'CF':>6s}")

        for col in df.columns:
            series = df[col]
            n_total = len(series)
            n_valid = series.notna().sum()
            completeness = n_valid / n_total * 100 if n_total > 0 else 0

            # Max gap
            isna = series.isna()
            max_gap_intervals = 0
            if isna.any():
                groups = (isna != isna.shift()).cumsum()
                gap_lengths = isna.groupby(groups).sum()
                max_gap_intervals = int(gap_lengths.max())
            max_gap_hours = max_gap_intervals * 10 / 60  # 10-min intervals

            n_days = n_total / 144  # 144 intervals per day at 10-min
            mean_mw = series.mean() if n_valid > 0 else 0
            peak_mw = series.max() if n_valid > 0 else 0
            cf = mean_mw / peak_mw if peak_mw > 0 else 0

            print(f"  {col:<20s} {n_days:>5.0f} {completeness:>9.1f}% {max_gap_hours:>9.1f} {mean_mw:>8.2f} {peak_mw:>8.2f} {cf:>6.3f}")

def plot_capacity_factors(
    real_solar: np.ndarray,
    real_wind: np.ndarray,
    real_solar_caps: list[float],
    real_wind_caps: list[float],
    output_dir: Path,
    n_days: int = 30,
) -> None:
    """Compare capacity factor distributions: real vs synthetic."""
    T = 96 * n_days
    rng = np.random.default_rng(42)
    config = WheelingConfig()

    # Compute real CFs
    real_solar_cfs = []
    for i in range(real_solar.shape[0]):
        max_kwh = real_solar_caps[i] * 0.25 * T
        cf = real_solar[i].sum() / max_kwh if max_kwh > 0 else 0
        real_solar_cfs.append(cf)

    real_wind_cfs = []
    for i in range(real_wind.shape[0]):
        max_kwh = real_wind_caps[i] * 0.25 * T
        cf = real_wind[i].sum() / max_kwh if max_kwh > 0 else 0
        real_wind_cfs.append(cf)

    # Generate synthetic for comparison
    n_synth = 20
    synth_solar_caps = rng.uniform(*config.solar_capacity_range_kw, size=n_synth)
    synth_solar = generate_solar_profiles(n_synth, synth_solar_caps, n_days=n_days, seed=42)
    synth_solar_cfs = []
    for i in range(n_synth):
        max_kwh = synth_solar_caps[i] * 0.25 * T
        cf = synth_solar[i].sum() / max_kwh if max_kwh > 0 else 0
        synth_solar_cfs.append(cf)

    synth_wind_caps = rng.uniform(*config.wind_capacity_range_kw, size=n_synth)
    synth_wind = generate_wind_profiles(n_synth, synth_wind_caps, n_days=n_days, seed=1042)
    synth_wind_cfs = []
    for i in range(n_synth):
        max_kwh = synth_wind_caps[i] * 0.25 * T
        cf = synth_wind[i].sum() / max_kwh if max_kwh > 0 else 0
        synth_wind_cfs.append(cf)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Solar
    ax = axes[0]
    if real_solar_cfs:
        ax.hist(real_solar_cfs, bins=10, alpha=0.6, label="Real (Taipower)", color="tab:orange")
    ax.hist(synth_solar_cfs, bins=10, alpha=0.6, label="Synthetic (pvlib)", color="tab:blue")
    ax.set_xlabel("Capacity Factor")
    ax.set_ylabel("Count")
    ax.set_title("Solar Capacity Factors")
    ax.legend()

    # Wind
    ax = axes[1]
    if real_wind_cfs:
        ax.hist(real_wind_cfs, bins=10, alpha=0.6, label="Real (Taipower)", color="tab:orange")
    ax.hist(synth_wind_cfs, bins=10, alpha=0.6, label="Synthetic (Weibull)", color="tab:blue")
    ax.set_xlabel("Capacity Factor")
    ax.set_ylabel("Count")
    ax.set_title("Wind Capacity Factors")
    ax.legend()

    plt.tight_layout()
    path = output_dir / "capacity_factors.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

def plot_daily_patterns(
    real_solar: np.ndarray,
    real_wind: np.ndarray,
    output_dir: Path,
    n_days: int = 30,
) -> None:
    """Compare average daily generation patterns."""
    config = WheelingConfig()
    rng = np.random.default_rng(42)
    T = 96 * n_days

    # Compute average daily profile (96 intervals)
    def avg_daily(profiles: np.ndarray) -> np.ndarray:
        if profiles.size == 0:
            return np.zeros(96)
        # Average across plants, then reshape to (n_days, 96) and average across days
        mean_profile = profiles.mean(axis=0)
        n_full_days = len(mean_profile) // 96
        if n_full_days == 0:
            return np.zeros(96)
        daily = mean_profile[:n_full_days * 96].reshape(n_full_days, 96)
        return daily.mean(axis=0)

    # Real
    real_solar_daily = avg_daily(real_solar)
    real_wind_daily = avg_daily(real_wind)

    # Synthetic
    n_synth = 10
    synth_solar_caps = rng.uniform(*config.solar_capacity_range_kw, size=n_synth)
    synth_solar = generate_solar_profiles(n_synth, synth_solar_caps, n_days=n_days, seed=42)
    synth_solar_daily = avg_daily(synth_solar)

    synth_wind_caps = rng.uniform(*config.wind_capacity_range_kw, size=n_synth)
    synth_wind = generate_wind_profiles(n_synth, synth_wind_caps, n_days=n_days, seed=1042)
    synth_wind_daily = avg_daily(synth_wind)

    hours = np.arange(96) * 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Solar daily pattern
    ax = axes[0]
    if real_solar_daily.max() > 0:
        ax.plot(hours, real_solar_daily / real_solar_daily.max(), label="Real (Taipower)", color="tab:orange", linewidth=2)
    if synth_solar_daily.max() > 0:
        ax.plot(hours, synth_solar_daily / synth_solar_daily.max(), label="Synthetic (pvlib)", color="tab:blue", linewidth=2, linestyle="--")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Normalized Generation")
    ax.set_title("Solar: Average Daily Pattern")
    ax.legend()
    ax.set_xlim(0, 24)

    # Wind daily pattern
    ax = axes[1]
    if real_wind_daily.max() > 0:
        ax.plot(hours, real_wind_daily / real_wind_daily.max(), label="Real (Taipower)", color="tab:orange", linewidth=2)
    if synth_wind_daily.max() > 0:
        ax.plot(hours, synth_wind_daily / synth_wind_daily.max(), label="Synthetic (Weibull)", color="tab:blue", linewidth=2, linestyle="--")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Normalized Generation")
    ax.set_title("Wind: Average Daily Pattern")
    ax.legend()
    ax.set_xlim(0, 24)

    plt.tight_layout()
    path = output_dir / "daily_patterns.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

def plot_autocorrelation(
    real_solar: np.ndarray,
    real_wind: np.ndarray,
    output_dir: Path,
    max_lag: int = 96,
) -> None:
    """Compare temporal autocorrelation of real vs synthetic profiles."""
    config = WheelingConfig()
    rng = np.random.default_rng(42)

    def compute_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
        x = x - x.mean()
        var = np.var(x)
        if var == 0:
            return np.zeros(max_lag)
        acf = np.correlate(x, x, mode="full")
        acf = acf[len(x) - 1:]
        acf = acf[:max_lag] / (var * len(x))
        return acf

    # Pick representative plant from each
    real_solar_acf = compute_acf(real_solar[0], max_lag) if real_solar.shape[0] > 0 else np.zeros(max_lag)
    real_wind_acf = compute_acf(real_wind[0], max_lag) if real_wind.shape[0] > 0 else np.zeros(max_lag)

    # Synthetic
    synth_solar_caps = rng.uniform(*config.solar_capacity_range_kw, size=1)
    synth_solar = generate_solar_profiles(1, synth_solar_caps, n_days=30, seed=42)
    synth_solar_acf = compute_acf(synth_solar[0], max_lag)

    synth_wind_caps = rng.uniform(*config.wind_capacity_range_kw, size=1)
    synth_wind = generate_wind_profiles(1, synth_wind_caps, n_days=30, seed=1042)
    synth_wind_acf = compute_acf(synth_wind[0], max_lag)

    lags = np.arange(max_lag) * 0.25  # hours

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(lags, real_solar_acf, label="Real (Taipower)", color="tab:orange", linewidth=2)
    ax.plot(lags, synth_solar_acf, label="Synthetic (pvlib)", color="tab:blue", linewidth=2, linestyle="--")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Solar: Temporal Autocorrelation")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)

    ax = axes[1]
    ax.plot(lags, real_wind_acf, label="Real (Taipower)", color="tab:orange", linewidth=2)
    ax.plot(lags, synth_wind_acf, label="Synthetic (Weibull)", color="tab:blue", linewidth=2, linestyle="--")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Wind: Temporal Autocorrelation")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    path = output_dir / "autocorrelation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

def main():
    parser = argparse.ArgumentParser(description="Validate real Taipower generation data")
    parser.add_argument(
        "--json", type=str, default="data/raw/taipower_37331.json",
        help="Path to Taipower JSON file",
    )
    parser.add_argument(
        "--output", type=str, default="results/validation",
        help="Output directory for plots and reports",
    )
    parser.add_argument(
        "--n-solar", type=int, default=10,
        help="Number of solar plants to select",
    )
    parser.add_argument(
        "--n-wind", type=int, default=5,
        help="Number of wind plants to select",
    )
    parser.add_argument(
        "--n-days", type=int, default=30,
        help="Window length in days",
    )
    parser.add_argument(
        "--month", type=int, default=None,
        help="Restrict window to this month (1-12)",
    )
    args = parser.parse_args()

    json_path = Path(args.json)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
        print(f"ERROR: Taipower JSON not found: {json_path}", file=sys.stderr)
        print("  Run: python scripts/download_taipower.py", file=sys.stderr)
        sys.exit(1)

    # Load data
    print("Loading Taipower data (with caching)...")
    data = load_taipower(json_path)

    # Quality report
    data_quality_report(data, output_dir)

    # Build generation profiles
    print(f"\nBuilding generation profiles (n_solar={args.n_solar}, n_wind={args.n_wind}, n_days={args.n_days})...")
    solar_profiles, wind_profiles, solar_caps, wind_caps, meta = build_generation_profiles(
        data=data,
        n_solar=args.n_solar,
        n_wind=args.n_wind,
        n_days=args.n_days,
        month=args.month,
        seed=42,
    )

    print(f"  Window: {meta['window_start']} to {meta['window_end']}")
    print(f"  Solar plants: {len(meta['solar_plants'])}")
    print(f"  Wind plants: {len(meta['wind_plants'])}")
    print(f"  Solar CFs: {meta.get('solar_capacity_factors', [])}")
    print(f"  Wind CFs: {meta.get('wind_capacity_factors', [])}")

    # Generate comparison plots
    print("\nGenerating validation plots...")
    plot_capacity_factors(solar_profiles, wind_profiles, solar_caps, wind_caps, output_dir, args.n_days)
    plot_daily_patterns(solar_profiles, wind_profiles, output_dir, args.n_days)
    plot_autocorrelation(solar_profiles, wind_profiles, output_dir)

    print(f"\nValidation complete. Results in: {output_dir}")

if __name__ == "__main__":
    main()
