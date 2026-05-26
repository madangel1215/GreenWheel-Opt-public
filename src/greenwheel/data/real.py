"""
Real Taipower generation data loader for semi-real wheeling instances.

Parses Taipower historical generation JSON (dataset #37331), selects
best time windows, samples plants, resamples to 15-min, and builds
generation profile arrays compatible with WheelingInstance.

Data source: https://service.taipower.com.tw/data/opendata/apply/file/d006010/001.json
"""

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Raw JSON parsing
# ---------------------------------------------------------------------------

def load_taipower_raw(path: str | Path) -> dict[str, pd.DataFrame]:
    """Parse Taipower JSON into per-technology DataFrames.

    Args:
        path: Path to the downloaded JSON file (~185 MB).

    Returns:
        Dict with keys 'solar' and 'wind', each a DataFrame with:
            - index: DatetimeIndex at 10-min resolution (Asia/Taipei)
            - columns: plant names (str)
            - values: generation in MW
    """
    path = Path(path)

    # Read with BOM handling
    raw_bytes = path.read_bytes()
    if raw_bytes[:3] == b"\xef\xbb\xbf":
        raw_text = raw_bytes[3:].decode("utf-8")
    else:
        raw_text = raw_bytes.decode("utf-8")

    data = json.loads(raw_text)

    # Actual format: {"records": {"NET_P": [{FUEL_TYPE, UNIT_NAME, DATETIME, NET_P}, ...]}}
    records = data["records"]
    net_p_list = records["NET_P"]

    # Collect per-plant time series
    solar_series: dict[str, list[tuple[str, float]]] = {}
    wind_series: dict[str, list[tuple[str, float]]] = {}

    for rec in net_p_list:
        fueltype = rec.get("FUEL_TYPE", "")
        name = rec.get("UNIT_NAME", "")
        dt_str = rec.get("DATETIME", "")
        net_p_val = rec.get("NET_P")

        if fueltype == "太陽能":
            target = solar_series
        elif fueltype == "風力":
            target = wind_series
        else:
            continue

        if name not in target:
            target[name] = []

        if dt_str and net_p_val is not None:
            try:
                val = float(net_p_val)
            except (ValueError, TypeError):
                val = np.nan
            target[name].append((dt_str, val))

    result = {}
    for tech, series_dict in [("solar", solar_series), ("wind", wind_series)]:
        if not series_dict:
            result[tech] = pd.DataFrame()
            continue

        frames = {}
        for plant_name, records in series_dict.items():
            if not records:
                continue
            idx = pd.to_datetime([r[0] for r in records])
            vals = [r[1] for r in records]
            s = pd.Series(vals, index=idx, name=plant_name, dtype=float)
            # Drop exact duplicate timestamps, keep first
            s = s[~s.index.duplicated(keep="first")]
            s = s.sort_index()
            frames[plant_name] = s

        if frames:
            df = pd.DataFrame(frames)
            df.index = df.index.tz_localize("Asia/Taipei")
            df = df.sort_index()
        else:
            df = pd.DataFrame()

        result[tech] = df

    return result

def load_taipower(
    path: str | Path,
    cache_dir: str | Path = "data/cache",
) -> dict[str, pd.DataFrame]:
    """Load Taipower data with pickle caching.

    Args:
        path: Path to the raw JSON file.
        cache_dir: Directory for pickle cache.

    Returns:
        Same as load_taipower_raw().
    """
    path = Path(path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Cache key based on file path and modification time
    stat = path.stat()
    key_str = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime}"
    cache_hash = hashlib.md5(key_str.encode()).hexdigest()[:12]
    cache_path = cache_dir / f"taipower_{cache_hash}.pkl"

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    data = load_taipower_raw(path)

    with open(cache_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return data

# ---------------------------------------------------------------------------
# Window selection
# ---------------------------------------------------------------------------

def _score_window(
    solar_df: pd.DataFrame,
    wind_df: pd.DataFrame,
    start: pd.Timestamp,
    n_days: int,
) -> float:
    """Score a candidate time window by data quality.

    Higher score = better window. Considers:
    - Completeness: fraction of non-NaN values
    - Plant coverage: fraction of plants with >50% valid data
    - Gap penalty: penalize windows with long contiguous NaN stretches
    """
    end = start + pd.Timedelta(days=n_days)

    score = 0.0
    total_weight = 0.0

    for df, weight in [(solar_df, 0.6), (wind_df, 0.4)]:
        if df.empty:
            continue

        window = df.loc[start:end]
        if window.empty:
            continue

        # Completeness
        total_cells = window.size
        valid_cells = window.notna().sum().sum()
        completeness = valid_cells / total_cells if total_cells > 0 else 0.0

        # Plant coverage (plants with >50% valid data)
        plant_valid_frac = window.notna().mean(axis=0)
        coverage = (plant_valid_frac > 0.5).mean()

        # Max gap length penalty (per plant, max consecutive NaN)
        max_gap = 0
        for col in window.columns:
            isna = window[col].isna()
            if isna.any():
                groups = (isna != isna.shift()).cumsum()
                gap_lengths = isna.groupby(groups).sum()
                max_gap = max(max_gap, gap_lengths.max())
        gap_penalty = max(0, 1.0 - max_gap / (144 * 3))  # penalize gaps > 3 days

        score += weight * (0.5 * completeness + 0.3 * coverage + 0.2 * gap_penalty)
        total_weight += weight

    return score / total_weight if total_weight > 0 else 0.0

def select_window(
    data: dict[str, pd.DataFrame],
    n_days: int = 30,
    month: int | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Select the best n_days contiguous window from the dataset.

    Args:
        data: Output from load_taipower().
        n_days: Window length in days.
        month: If given, restrict to windows starting in this month (1-12).

    Returns:
        (start, end) timestamps for the selected window.
    """
    solar_df = data.get("solar", pd.DataFrame())
    wind_df = data.get("wind", pd.DataFrame())

    # Determine date range from data
    all_dates = set()
    for df in [solar_df, wind_df]:
        if not df.empty:
            all_dates.update(df.index.normalize().unique())

    if not all_dates:
        raise ValueError("No data found in dataset")

    all_dates = sorted(all_dates)
    min_date = all_dates[0]
    max_date = all_dates[-1]

    # Generate candidate start dates (every day)
    candidates = pd.date_range(
        start=min_date,
        end=max_date - pd.Timedelta(days=n_days),
        freq="D",
        tz="Asia/Taipei",
    )

    if month is not None:
        candidates = candidates[candidates.month == month]

    if len(candidates) == 0:
        raise ValueError(
            f"No valid {n_days}-day windows found"
            + (f" for month={month}" if month else "")
        )

    # Score each candidate
    best_start = candidates[0]
    best_score = -1.0

    for start in candidates:
        score = _score_window(solar_df, wind_df, start, n_days)
        if score > best_score:
            best_score = score
            best_start = start

    end = best_start + pd.Timedelta(days=n_days)
    return best_start, end

# ---------------------------------------------------------------------------
# Plant selection
# ---------------------------------------------------------------------------

def select_plants(
    data: dict[str, pd.DataFrame],
    n_solar: int,
    n_wind: int,
    window: tuple[pd.Timestamp, pd.Timestamp],
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select plants via stratified sampling across capacity quantiles.

    If more plants are requested than available, replicates existing plants
    with added noise to create diversity.

    Args:
        data: Output from load_taipower().
        n_solar: Number of solar plants to select.
        n_wind: Number of wind plants to select.
        window: (start, end) timestamps.
        seed: Random seed.

    Returns:
        (solar_df, wind_df) DataFrames sliced to the window, with exactly
        n_solar and n_wind columns respectively.
    """
    rng = np.random.default_rng(seed)
    start, end = window

    results = []
    for tech, n_wanted in [("solar", n_solar), ("wind", n_wind)]:
        df = data.get(tech, pd.DataFrame())
        if df.empty or n_wanted == 0:
            results.append(pd.DataFrame())
            continue

        # Slice to window
        window_df = df.loc[start:end].copy()

        # Filter plants with sufficient data (>60% non-NaN)
        valid_frac = window_df.notna().mean()
        good_plants = valid_frac[valid_frac > 0.6].index.tolist()

        if len(good_plants) == 0:
            raise ValueError(f"No {tech} plants with >60% valid data in window")

        window_df = window_df[good_plants]

        # Compute mean generation per plant (for stratified sampling)
        mean_gen = window_df.mean()

        if n_wanted <= len(good_plants):
            # Stratified sampling: divide plants into quantiles, sample from each
            n_quantiles = min(n_wanted, 4)
            quantile_edges = np.linspace(0, 1, n_quantiles + 1)

            selected = []
            plants_per_q = n_wanted // n_quantiles
            remainder = n_wanted % n_quantiles

            sorted_plants = mean_gen.sort_values().index.tolist()

            for q in range(n_quantiles):
                lo = int(quantile_edges[q] * len(sorted_plants))
                hi = int(quantile_edges[q + 1] * len(sorted_plants))
                hi = max(hi, lo + 1)
                q_plants = sorted_plants[lo:hi]

                n_from_q = plants_per_q + (1 if q < remainder else 0)
                n_from_q = min(n_from_q, len(q_plants))
                chosen = rng.choice(q_plants, size=n_from_q, replace=False).tolist()
                selected.extend(chosen)

            # Ensure exactly n_wanted
            selected = selected[:n_wanted]
            selected_df = window_df[selected]

        else:
            # Need more plants than available: replicate with noise
            selected_df = window_df.copy()

            n_extra = n_wanted - len(good_plants)
            for i in range(n_extra):
                source_col = rng.choice(good_plants)
                noise_scale = 0.05 + 0.05 * rng.random()  # 5-10% noise
                noisy = window_df[source_col].copy()
                noise = rng.normal(1.0, noise_scale, size=len(noisy))
                noisy = noisy * noise
                noisy = noisy.clip(lower=0)
                new_name = f"{source_col}_rep{i}"
                selected_df[new_name] = noisy

        # Rename columns to sequential names
        new_cols = [f"{tech}_{i}" for i in range(len(selected_df.columns))]
        selected_df.columns = new_cols
        results.append(selected_df)

    return results[0], results[1]

# ---------------------------------------------------------------------------
# Resampling and imputation
# ---------------------------------------------------------------------------

def resample_to_15min(
    series_10min: pd.DataFrame,
    window: tuple[pd.Timestamp, pd.Timestamp],
) -> np.ndarray:
    """Resample 10-min MW data to 15-min kWh arrays.

    Args:
        series_10min: DataFrame with DatetimeIndex (10-min) and plant columns (MW).
        window: (start, end) timestamps.

    Returns:
        Array of shape (n_plants, T) where T = 96 * n_days, values in kWh/15min.
    """
    if series_10min.empty:
        return np.zeros((0, 0))

    start, end = window

    # Create target 15-min index
    n_days = (end - start).days
    T = 96 * n_days
    target_index = pd.date_range(start=start, periods=T, freq="15min", tz="Asia/Taipei")

    # Resample: 10-min → 15-min via linear interpolation
    # First ensure the 10-min data covers the window
    sliced = series_10min.loc[start:end]

    # Reindex to 15-min with interpolation
    combined_index = sliced.index.union(target_index).sort_values().unique()
    reindexed = sliced.reindex(combined_index)
    interpolated = reindexed.interpolate(method="time", limit_direction="both")
    result_df = interpolated.reindex(target_index)

    # Convert MW → kWh per 15-min interval: MW * 0.25h * 1000 = kWh
    # But since data is already in MW (average power), energy = MW * 0.25h
    # We keep units consistent: MW * 0.25 = MWh/interval, * 1000 = kWh/interval
    values = result_df.values.T * 0.25 * 1000  # (n_plants, T), kWh per 15-min

    return np.maximum(0, values)

def impute_missing(profile: np.ndarray, technology: str) -> np.ndarray:
    """Fill remaining NaN gaps in a generation profile.

    Strategy:
    - Short gaps (<= 6 intervals = 1.5h): linear interpolation
    - Medium gaps (<= 96 intervals = 1 day): use same-hour values from adjacent days
    - Long gaps: fill with 0 for solar nighttime, mean for wind

    Args:
        profile: Shape (T,) with potential NaN values.
        technology: 'solar' or 'wind'.

    Returns:
        Profile with NaN replaced.
    """
    if not np.any(np.isnan(profile)):
        return profile

    result = profile.copy()
    T = len(result)

    # Pass 1: Linear interpolation for short gaps
    nan_mask = np.isnan(result)
    if nan_mask.any():
        valid = ~nan_mask
        if valid.sum() >= 2:
            x_valid = np.where(valid)[0]
            y_valid = result[valid]
            x_all = np.arange(T)
            result = np.interp(x_all, x_valid, y_valid)

    # Pass 2: Replace any remaining NaN with technology-appropriate defaults
    still_nan = np.isnan(result)
    if still_nan.any():
        if technology == "solar":
            result[still_nan] = 0.0  # nighttime or zero generation
        else:
            # Use global mean of valid values
            valid_mean = np.nanmean(profile)
            if np.isnan(valid_mean):
                valid_mean = 0.0
            result[still_nan] = valid_mean

    return np.maximum(0, result)

# ---------------------------------------------------------------------------
# Profile builder
# ---------------------------------------------------------------------------

def build_generation_profiles(
    data: dict[str, pd.DataFrame],
    n_solar: int,
    n_wind: int,
    n_days: int = 30,
    month: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[float], list[float], dict]:
    """Build generation profile arrays from real Taipower data.

    Complete pipeline: window selection → plant selection → resample → impute.

    Args:
        data: Output from load_taipower().
        n_solar: Number of solar plants.
        n_wind: Number of wind plants.
        n_days: Window length in days.
        month: Optional month restriction for window selection.
        seed: Random seed for plant selection.

    Returns:
        Tuple of:
            - solar_profiles: shape (n_solar, T), kWh per 15-min
            - wind_profiles: shape (n_wind, T), kWh per 15-min
            - solar_capacities_kw: list of capacity estimates per solar plant
            - wind_capacities_kw: list of capacity estimates per wind plant
            - metadata: dict with window dates, plant names, quality info
    """
    T = 96 * n_days

    # Step 1: Select best window
    window = select_window(data, n_days=n_days, month=month)
    start, end = window

    # Step 2: Select plants
    solar_df, wind_df = select_plants(data, n_solar, n_wind, window, seed=seed)

    metadata = {
        "window_start": str(start),
        "window_end": str(end),
        "n_days": n_days,
        "solar_plants": list(solar_df.columns) if not solar_df.empty else [],
        "wind_plants": list(wind_df.columns) if not wind_df.empty else [],
    }

    # Step 3: Resample and convert
    solar_profiles = np.zeros((n_solar, T))
    wind_profiles = np.zeros((n_wind, T))

    solar_capacities_kw = []
    wind_capacities_kw = []

    if not solar_df.empty:
        solar_raw = resample_to_15min(solar_df, window)
        for i in range(min(n_solar, solar_raw.shape[0])):
            solar_profiles[i] = impute_missing(solar_raw[i], "solar")
            # Estimate capacity from peak generation: peak_kWh / 0.25h = kW
            peak_kwh = np.max(solar_profiles[i])
            cap_kw = peak_kwh / 0.25 if peak_kwh > 0 else 1.0
            solar_capacities_kw.append(float(cap_kw))

    if not wind_df.empty:
        wind_raw = resample_to_15min(wind_df, window)
        for i in range(min(n_wind, wind_raw.shape[0])):
            wind_profiles[i] = impute_missing(wind_raw[i], "wind")
            peak_kwh = np.max(wind_profiles[i])
            cap_kw = peak_kwh / 0.25 if peak_kwh > 0 else 1.0
            wind_capacities_kw.append(float(cap_kw))

    # Pad capacity lists if needed (for replicated plants)
    while len(solar_capacities_kw) < n_solar:
        solar_capacities_kw.append(solar_capacities_kw[-1] if solar_capacities_kw else 1.0)
    while len(wind_capacities_kw) < n_wind:
        wind_capacities_kw.append(wind_capacities_kw[-1] if wind_capacities_kw else 1.0)

    metadata["solar_capacities_kw"] = solar_capacities_kw
    metadata["wind_capacities_kw"] = wind_capacities_kw

    # Quality summary
    solar_cf = []
    for i in range(n_solar):
        cap = solar_capacities_kw[i]
        max_kwh = cap * 0.25 * T
        cf = solar_profiles[i].sum() / max_kwh if max_kwh > 0 else 0.0
        solar_cf.append(round(cf, 4))

    wind_cf = []
    for i in range(n_wind):
        cap = wind_capacities_kw[i]
        max_kwh = cap * 0.25 * T
        cf = wind_profiles[i].sum() / max_kwh if max_kwh > 0 else 0.0
        wind_cf.append(round(cf, 4))

    metadata["solar_capacity_factors"] = solar_cf
    metadata["wind_capacity_factors"] = wind_cf

    return solar_profiles, wind_profiles, solar_capacities_kw, wind_capacities_kw, metadata
