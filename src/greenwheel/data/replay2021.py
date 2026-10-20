"""
The 2026 wheeling regime replayed on 2021 weather and load (two cases).

Rules and prices are those in force in 2026; generation is the Taipower 2017-2021
record, load is EWELD (case 1, small and medium C&I) or VEA (case 2, large industry).
A retailer fixes the shares W before month t from scenarios built only from data before
t, and is judged on the month that actually happened.
"""

import datetime as dt
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.greenwheel.config import ConsumerSite, GeneratorPlant, MarketParams, WheelingInstance
from src.greenwheel.data import realcase as rc
from src.greenwheel.simulator.settlement import ConsumerContractSettlement, ScenarioBatchSettlement, tou_labels_114

YEAR = 2021
# 2021-02 and 2021-12 have too few EWELD users in common, 2021-05 has a gap in the
# national shape.
MONTHS = (1, 3, 4, 6, 7, 8, 9, 10, 11)
GEN_YEARS = (2017, 2018, 2019, 2020)
SCEN_DAYS = 28
VEA = Path("data/raw/vea/load_profiles_tabsep.csv")
VEA_MASTER = Path("data/raw/vea/master_data_tabsep.csv")
VEA_STATS = Path("data/interim/vea_site_stats.csv")
# 260 weeks: 2021 dates map onto 2016 dates with the same weekday.
VEA_SHIFT = dt.timedelta(days=1820)

# 115年輸配電業各項費率, renewable (zero emission), tax included; the retailer reclaims the
# 5% VAT, so costs are taken net.
FEE_BASE = (0.0364 + 0.0274) / 1.05
FEE_TRANS = 0.0778 / 1.05
FEE_DIST = 0.1788 / 1.05
PPA_PRICE = 4.552          # 臺中地院114全150: 轉供度數 × 4.552 (未稅)
GREEN_PRICE = (5.4, 6.3)   # 台電 RE30 Q&A "行情約 5.4"; RE30 115年 6.3 (未稅)
OBLIGATION_KW = 5000       # 一定契約容量…管理辦法 第3條
EHV_PLANT_KW = 20000       # 併聯技術要點 第三點: 20,000 kW and above may connect at transmission level
SOLAR_KWH_PER_KW = 1250    # 管理辦法 附件
WIND_WEIGHT = 1250 / 2500  # a wind kWh covers half as much obligation as a solar kWh
PENALTY_PER_SOLAR_KWH = 4 * 2500 / SOLAR_KWH_PER_KW   # 代金: 未履行容量 × 2,500 度/瓩 × 4 元/度
RE100_TARGET = (0.2, 0.6)
N_OBLIGATED_VEA = 3

def month_days(year, month):
    first = dt.date(year, month, 1)
    nxt = dt.date(year + (month == 12), month % 12 + 1, 1)
    return [first + dt.timedelta(d) for d in range((nxt - first).days)]

def shift_month(year, month, k):
    i = year * 12 + month - 1 + k
    return i // 12, i % 12 + 1

@lru_cache(maxsize=None)
def _gen(year, month):
    return {g["plant"]: g for g in rc.plant_generation(year, month)}

@lru_cache(maxsize=1)
def plant_pool():
    """Plants reported in every valid month of 2017-2021 (26: 16 solar, 10 wind)."""
    pool = None
    for y in range(2017, YEAR + 1):
        for m in range(1, 13):
            g = _gen(y, m)
            if g:
                pool = set(g) if pool is None else pool & set(g)
    return tuple(sorted(pool))

def plant_info():
    g = _gen(YEAR, 7)
    return {p: (g[p]["tech"], g[p]["cap_kw"]) for p in plant_pool()}

def supply_2021(plants):
    return sum(_gen(YEAR, m)[p]["kwh"] for m in MONTHS for p in plants)

def portfolio(seed, rho, demand_total):
    """The prefix of a seeded random plant order whose 2021 supply is closest to rho x demand.

    "Add until reaching rho" overshot by one large wind farm (0.94 for rho 0.7).
    """
    order = list(np.random.default_rng(seed + 20_000).permutation(plant_pool()))
    cum = np.cumsum([sum(_gen(YEAR, m)[p]["kwh"] for m in MONTHS) for p in order]) / demand_total
    k = int(np.argmin(np.abs(cum - rho))) + 1
    return sorted(order[:k]), float(cum[k - 1])

# ---------------------------------------------------------------- customers

class EweldCustomers:
    """Case 1: the 106 EWELD users with good data in every replayed month."""

    name = "eweld"

    def __init__(self):
        sets = [set(map(str, rc.eligible_users(YEAR, m))) for m in MONTHS]
        self.ids = sorted(set.intersection(*sets))
        self.C = np.array([rc._eweld_user(u)["2020-07-01":"2021-12-31"].quantile(0.999) * 4 for u in self.ids])

    def _ok(self, u, year, month):
        um = rc.user_month(Path(u), year, month)
        return um is not None and um[1] >= rc.MIN_GOOD

    def month(self, year, month):
        """(n, T) kWh per 15 min for a calendar month, or None for a user without good data."""
        out = []
        for u in self.ids:
            um = rc.user_month(Path(u), year, month)
            out.append(um[0] if um is not None and um[1] >= rc.MIN_GOOD else None)
        return out

    def scenario_loads(self, t):
        """Load scenarios for target month t: months t-1, t-2, t-3, t-12, per user.

        A user without good data in one of them takes its nearest earlier good month
        within 36 months before t. Returns [(days, (n, SCEN_DAYS*96))].
        """
        months = [shift_month(YEAR, t, k) for k in (-1, -2, -3, -12)]
        out = []
        for ym in months:
            rows = []
            for u in self.ids:
                src = ym
                if not self._ok(u, *src):
                    src = next((shift_month(YEAR, t, -k) for k in range(1, 37)
                                if self._ok(u, *shift_month(YEAR, t, -k))), None)
                if src is None:
                    raise ValueError(f"{u}: no good month before {YEAR}-{t:02d}")
                rows.append(rc.user_month(Path(u), *src)[0][: SCEN_DAYS * 96])
            out.append((month_days(*ym)[:SCEN_DAYS], np.stack(rows)))
        return out

    def realized(self, t):
        rows = self.month(YEAR, t)
        if any(r is None for r in rows):
            raise ValueError(f"missing realized load in {YEAR}-{t:02d}")
        return month_days(YEAR, t), np.stack(rows)

def _vea_stats():
    """Per-site 99.9th percentile (kW), energy (kWh) and byte offset, cached once."""
    if VEA_STATS.exists():
        return pd.read_csv(VEA_STATS)
    rows = []
    with open(VEA, "rb") as f:
        f.readline()
        while True:
            off = f.tell()
            line = f.readline()
            if not line:
                break
            parts = line.rstrip(b"\n").split(b"\t")
            v = np.array(parts[1:], dtype=float)
            rows.append((int(parts[0]), off, np.nanquantile(v, 0.999), np.nansum(v) / 4, int(np.isnan(v).sum())))
    s = pd.DataFrame(rows, columns=["id", "offset", "c_kw", "kwh", "n_nan"])
    m = pd.read_csv(VEA_MASTER, sep="\t")[["Id", "Sector_group_id"]].rename(columns={"Id": "id"})
    s = s.merge(m, on="id", how="left")
    VEA_STATS.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(VEA_STATS, index=False)
    return s

@lru_cache(maxsize=None)
def _vea_site(offset):
    with open(VEA, "rb") as f:
        f.seek(offset)
        parts = f.readline().rstrip(b"\n").split(b"\t")
    return np.array(parts[1:], dtype=float) / 4          # kW over 15 min -> kWh

class VeaCustomers:
    """Case 2: three obligation-size manufacturers plus smaller sites."""

    name = "vea"

    def __init__(self, seed, pool_total):
        s = _vea_stats()
        s = s[s.n_nan == 0]
        rng = np.random.default_rng(seed + 30_000)
        big = s[(s.c_kw >= 5000) & (s.c_kw < 10000) & (s.Sector_group_id == "C")]
        big = big.iloc[sorted(rng.choice(len(big), N_OBLIGATED_VEA, replace=False))]
        small = s[(s.c_kw >= 100) & (s.c_kw < 5000)]
        small = small.iloc[rng.permutation(len(small))]
        chosen = list(big.itertuples())
        demand = sum(self._demand(r.offset) for r in chosen)
        for r in small.itertuples():
            if demand >= pool_total / 1.6:
                break
            chosen.append(r)
            demand += self._demand(r.offset)
        self.ids = [int(r.id) for r in chosen]
        self.offsets = [int(r.offset) for r in chosen]
        self.C = np.array([r.c_kw for r in chosen])

    @staticmethod
    def _days_index(days):
        idx = [((d - VEA_SHIFT) - dt.date(2016, 1, 1)).days for d in days]
        # Late-December dates fall past the end of 2016; such a block moves one more week
        # back, which keeps the weekday.
        return [i - 7 for i in idx] if max(idx) > 365 else idx

    def _rows(self, days):
        idx = np.concatenate([np.arange(i * 96, i * 96 + 96) for i in self._days_index(days)])
        return np.stack([_vea_site(o)[idx] for o in self.offsets])

    def _demand(self, offset):
        days = [d for m in MONTHS for d in month_days(YEAR, m)]
        idx = np.concatenate([np.arange(i * 96, i * 96 + 96) for i in self._days_index(days)])
        return _vea_site(offset)[idx].sum()

    def scenario_loads(self, t):
        # Months t-1..t-3 within the 2021 calendar, wrapping January to Oct-Dec; VEA has a
        # single year, so these are shape samples, not the past.
        out = []
        for k in (1, 2, 3):
            m = (t - 1 - k) % 12 + 1
            days = month_days(YEAR, m)[:SCEN_DAYS]
            out.append((days, self._rows(days)))
        return out

    def realized(self, t):
        days = month_days(YEAR, t)
        return days, self._rows(days)

# ---------------------------------------------------------------- the retailer's month

class Market:
    """Seeded prices and targets for one case."""

    def __init__(self, customers, seed):
        n = len(customers.ids)
        rng = np.random.default_rng(seed + 10_000)
        self.price = rng.uniform(*GREEN_PRICE, n)
        self.target = rng.uniform(*RE100_TARGET, n)
        self.obligated = customers.C >= OBLIGATION_KW
        self.requirement = np.where(self.obligated, 0.1 * customers.C * SOLAR_KWH_PER_KW, 0.0)   # solar kWh / year
        self.ehv_customer = customers.C >= OBLIGATION_KW

def _instance(G, cap_kw, techs, D):
    gens = [GeneratorPlant(id=i, capacity_kw=float(cap_kw[i]), technology=techs[i], ppa_price=PPA_PRICE,
                           generation=G[i]) for i in range(len(G))]
    cons = [ConsumerSite(id=j, total_monthly_kwh=float(D[j].sum()), contract_kwh=0.0, re100_target=0.0,
                         retail_price=0.0, demand=D[j]) for j in range(len(D))]
    return WheelingInstance(generators=gens, consumers=cons, market=MarketParams(), n_intervals=D.shape[1])

def _labels(settle):
    out = np.zeros(settle.masks[0].shape, dtype=int)
    for p, mk in enumerate(settle.masks):
        out[mk] = p
    return out

class MonthProblem:
    """Scenarios (ex ante) and the realized month (ex post) for target month t."""

    def __init__(self, customers, market, plants, t):
        info = plant_info()
        self.plants, self.t, self.market = plants, t, market
        self.techs = [info[p][0] for p in plants]
        self.cap = np.array([info[p][1] for p in plants])
        self.solar = np.array([tc == "solar" for tc in self.techs])
        self.ehv_plant = self.cap >= EHV_PLANT_KW
        self.m, self.n = len(plants), len(customers.ids)
        self.summer = dt.date(YEAR, t, 15) >= dt.date(YEAR, 5, 16) and dt.date(YEAR, t, 15) <= dt.date(YEAR, 10, 15)
        self.days_t = len(month_days(YEAR, t))
        gen_months = [shift_month(y, t, k) for y in GEN_YEARS for k in (-1, 0, 1)]
        gen_months = [ym for ym in gen_months if _gen(*ym) and all(p in _gen(*ym) for p in plants)]
        loads = customers.scenario_loads(t)
        self.scen = []
        for gy in gen_months:
            G = np.stack([_gen(*gy)[p]["generation"][: SCEN_DAYS * 96] for p in plants])
            for days, D in loads:
                tou = tou_labels_114(days, summer=self.summer)
                self.scen.append(dict(settle=ConsumerContractSettlement(_instance(G, self.cap, self.techs, D), tou),
                                      demand=D.sum(axis=1), scale=self.days_t / SCEN_DAYS, gen=gy))
        R = np.stack([c["settle"].R for c in self.scen])
        D = np.stack([c["settle"].D for c in self.scen])
        tou = np.stack([_labels(c["settle"]) for c in self.scen])
        self.batch = ScenarioBatchSettlement(R, D, tou)
        self.batch_demand = np.stack([c["demand"] for c in self.scen])
        days, D = customers.realized(t)
        G = np.stack([_gen(YEAR, t)[p]["generation"] for p in plants])
        self.real = dict(settle=ConsumerContractSettlement(_instance(G, self.cap, self.techs, D), tou_labels_114(days)),
                         demand=D.sum(axis=1), scale=1.0)

    def fees(self, W):
        inside = W >= 1e-4
        trans_only = self.market.ehv_customer & np.all(~inside | self.ehv_plant[:, None], axis=0)
        return FEE_BASE + FEE_TRANS + np.where(trans_only, 0.0, FEE_DIST)

    def _metrics(self, case, W, f):
        s = case["settle"].settle_quantities(W) * case["scale"]          # (m, n)
        mk = self.market
        green = s.sum(axis=0)
        profit = ((mk.price - PPA_PRICE - f) * green).sum()
        eq = (s * np.where(self.solar, 1.0, WIND_WEIGHT)[:, None]).sum(axis=0)
        penalty = (PENALTY_PER_SOLAR_KWH * np.maximum(0.0, mk.requirement / 12 - eq))[mk.obligated].sum()
        need = mk.target * case["demand"] * case["scale"]
        ach = np.minimum(1.0, green / np.maximum(need, 1e-9))[~mk.obligated]
        return profit, penalty, ach.min() if ach.size else 1.0, green, eq

    def evaluate(self, W):
        """Per-scenario arrays: profit, penalty, worst achievement (batched exact settlement)."""
        W = np.round(W, 4)
        f = self.fees(W)
        mk = self.market
        s = self.batch.settle_quantities(W) * (self.days_t / SCEN_DAYS)          # (S, m, n)
        green = s.sum(axis=1)
        profit = ((mk.price - PPA_PRICE - f)[None] * green).sum(axis=1)
        eq = np.einsum("smn,m->sn", s, np.where(self.solar, 1.0, WIND_WEIGHT))
        penalty = (PENALTY_PER_SOLAR_KWH * np.maximum(0.0, mk.requirement / 12 - eq))[:, mk.obligated].sum(axis=1)
        need = mk.target[None] * self.batch_demand * (self.days_t / SCEN_DAYS)
        ach = np.minimum(1.0, green / np.maximum(need, 1e-9))[:, ~mk.obligated]
        worst = ach.min(axis=1) if ach.shape[1] else np.ones(len(s))
        return dict(profit=profit, penalty=penalty, worst=worst)

    def evaluate_loop(self, W):
        """The same, one scenario at a time with ConsumerContractSettlement; the oracle."""
        W = np.round(W, 4)
        f = self.fees(W)
        r = np.array([self._metrics(c, W, f)[:3] for c in self.scen])
        return dict(profit=r[:, 0], penalty=r[:, 1], worst=r[:, 2])

    def evaluate_real(self, W):
        W = np.round(W, 4)
        p, pen, w, green, eq = self._metrics(self.real, W, self.fees(W))
        return dict(profit=p, penalty=pen, worst=w, green=green, obligation_eq=eq)
