"""
Real-scenario instances: Taipower generation + EWELD corporate load, month by month.

Generation. Taipower publishes plant-level output older than three months only as
monthly totals (#29938 solar stations, #29961 wind turbines); 10-minute history exists
only as national totals per technology (Zenodo 7537890, CC BY 4.0). Each plant's month
is therefore the national shape of its technology scaled to the plant's reported
monthly energy: real monthly energy per plant, real 10-minute national dynamics, and
no plant-to-plant weather differences within a technology.

Load. EWELD (figshare 10.6084/m9.figshare.21893808, CC BY 4.0): 15-minute kWh of
386 industrial and commercial users in southern China. Missing readings were forward
filled by the authors, so runs of identical values of two hours or more are treated as
gaps when selecting users. Timestamps mark the end of each interval.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.greenwheel.config import ConsumerSite, GeneratorPlant, MarketParams, WheelingInstance

RAW = Path("data/raw")
ZENODO = RAW / "taipower_zenodo_7537890" / "powerRatio_10min_2017Jan_2022Jun.csv"
SOLAR_MONTHLY = RAW / "taipower_monthly" / "d693003.csv"
WIND_MONTHLY = RAW / "taipower_monthly" / "d693004.csv"
EWELD = RAW / "eweld" / "EWELD" / "Electricity Consumption"
FILL_RUN = 8          # identical readings in a row (2 h) taken as a forward-filled gap
MIN_GOOD = 0.95

def _month_index(year, month):
    start = pd.Timestamp(year=year, month=month, day=1)
    return pd.date_range(start, start + pd.offsets.MonthBegin(1), freq="15min", inclusive="left")

@lru_cache(maxsize=1)
def _national():
    z = pd.read_csv(ZENODO, parse_dates=["datetime"]).set_index("datetime")[["solar", "wind"]]
    return z

def national_shape(tech, year, month):
    """15-minute mean national output (MW) for the month, or None if the month has gaps.

    The 10-minute values are snapshots, and some carry off-grid stamps (e.g. 12:13 in
    July 2019), so every snapshot is kept: power at a, a+5 and a+10 min is linearly
    interpolated between the actual stamps and averaged for the interval [a, a+15).
    A month is used only if no two consecutive snapshots are more than 60 min apart: in
    2017-2022 the gaps are either under 50 min or a day and longer (2017-02, 2017-05,
    2018-05, 2019-04..06, 2021-05, 2022-05), which are excluded.
    """
    idx = _month_index(year, month)
    z = _national()[tech].dropna()
    lo, hi = idx[0], idx[-1] + pd.Timedelta(minutes=15)
    window = z[(z.index >= lo - pd.Timedelta(minutes=10)) & (z.index <= hi + pd.Timedelta(minutes=10))]
    if len(window) < 2 or window.index[0] > lo or window.index[-1] < hi:
        return None
    t = (window.index - lo).total_seconds().to_numpy()
    if np.diff(t).max() > 60 * 60:
        return None
    pts = np.arange(len(idx))[:, None] * 900 + np.array([0, 300, 600])[None, :]
    return np.interp(pts, t, window.to_numpy()).mean(axis=1)

@lru_cache(maxsize=1)
def _plant_months():
    s = pd.read_csv(SOLAR_MONTHLY, encoding="utf-8-sig")
    s.columns = ["year", "month", "plant", "cap_kw", "kwh", "kwh_per_kw_day"]
    s["tech"] = "solar"
    w = pd.read_csv(WIND_MONTHLY, encoding="utf-8-sig")
    w.columns = ["year", "month", "county", "code", "plant", "turbine", "cap_kw", "kwh", "hours", "idle"]
    # 393 turbine-months are "-" or blank; a station-month missing any turbine would be
    # understated by a plain sum, so such station-months are dropped instead.
    w["kwh"] = pd.to_numeric(w["kwh"], errors="coerce")
    w = w.groupby(["year", "month", "plant"], as_index=False).agg(
        cap_kw=("cap_kw", "sum"), kwh=("kwh", lambda x: x.sum() if x.notna().all() else np.nan))
    w = w.dropna(subset=["kwh"])
    w["tech"] = "wind"
    return pd.concat([s[["year", "month", "plant", "tech", "cap_kw", "kwh"]], w], ignore_index=True)

def plant_generation(year, month):
    """All plants with a reported monthly total and a complete national shape."""
    pm = _plant_months()
    rows = pm[(pm.year == year) & (pm.month == month) & (pm.kwh > 0)]
    out = []
    for tech in ("solar", "wind"):
        shape = national_shape(tech, year, month)
        if shape is None:
            continue
        total = shape.sum()
        for r in rows[rows.tech == tech].itertuples():
            out.append(dict(plant=r.plant, tech=tech, cap_kw=float(r.cap_kw), kwh=float(r.kwh),
                            generation=shape / total * float(r.kwh)))
    return out

@lru_cache(maxsize=None)
def _eweld_user(path):
    df = pd.read_csv(path, parse_dates=["Time"])
    # End-of-interval timestamps: the reading at t covers [t - 15 min, t).
    return pd.Series(df["Value"].to_numpy(float), index=df["Time"] - pd.Timedelta(minutes=15))

def eweld_users():
    return sorted(EWELD.glob("*/*/*.csv"))

def user_month(path, year, month):
    """(readings, good) for the month, or None if the month is incomplete."""
    s = _eweld_user(str(path))
    idx = _month_index(year, month)
    v = s.reindex(idx)
    if v.isna().any():
        return None
    a = v.to_numpy()
    chg = np.concatenate([[True], np.diff(a) != 0])
    run = np.cumsum(chg)
    filled = np.bincount(run)[run] >= FILL_RUN
    return a, float((~filled).mean())

def eligible_users(year, month):
    out = []
    for p in eweld_users():
        um = user_month(p, year, month)
        if um is None:
            continue
        a, good = um
        if good >= MIN_GOOD and np.median(a) > 1.0:
            out.append(p)
    return out

def build_month(year, month, n_consumers, seed, rho=0.7, plants=None, market=None):
    """WheelingInstance for one month. Prices are placeholders (NaN)."""
    rng = np.random.default_rng(seed)
    gens = plant_generation(year, month)
    if plants is not None:
        gens = [g for g in gens if g["plant"] in plants]
    users = eligible_users(year, month)
    if len(users) < n_consumers:
        raise ValueError(f"{year}-{month:02d}: only {len(users)} eligible users")
    chosen = sorted(rng.choice(len(users), size=n_consumers, replace=False))
    loads = np.stack([user_month(users[k], year, month)[0] for k in chosen])
    supply = sum(g["kwh"] for g in gens)
    kappa = supply / (rho * loads.sum())
    loads = loads * kappa

    generators = [GeneratorPlant(id=i, capacity_kw=g["cap_kw"], technology=g["tech"], ppa_price=float("nan"),
                                 generation=g["generation"]) for i, g in enumerate(gens)]
    consumers = [ConsumerSite(id=j, total_monthly_kwh=float(loads[j].sum()), contract_kwh=float(loads[j].sum()),
                              re100_target=float("nan"), retail_price=float("nan"), demand=loads[j])
                 for j in range(n_consumers)]
    inst = WheelingInstance(generators=generators, consumers=consumers, market=market or MarketParams(),
                            n_intervals=loads.shape[1])
    inst.meta = dict(year=year, month=month, seed=seed, rho=rho, kappa=kappa,
                     plants=[g["plant"] for g in gens], users=[str(users[k]) for k in chosen])
    return inst
