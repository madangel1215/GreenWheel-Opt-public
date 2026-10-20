"""
Real-scenario retailers: three 2025 archetypes under the 2026 regime, replayed on 2021.

Supply is real: each site is a T-REC 2025 site of the archetype retailer at its real
annual volume, spread over 2021 like a Taipower station's monthly output; every station
follows the national intraday shape, so sites on the same station differ only in scale
and settlement is exact at station-group level, which is what the search
varies. Demand is a real EWELD load shape scaled so that each buyer's real contract volume
is rho of its load over the replayed months.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.greenwheel.data import realcase as rc
from src.greenwheel.data import replay2021 as rp
from src.greenwheel.simulator.settlement import (ConsumerContractSettlement, ScenarioBatchSettlement,
                                                 tou_labels_114)

# Built by scripts/build_trec_archetypes.py from the T-REC transaction records (see README).
TREC = Path("data/raw/trec/trec_archetypes_2025.tsv")
# Hydro sites are left out: Taipower publishes no intraday hydro record to give them a
# shape (台汽電 14.7% of 2025 MWh, 天能 0.1%).
TECH = {"太陽能": "solar", "風力能": "wind"}
ARCHETYPES = ("S", "M", "L")
# All sites are below 10 MW (T-REC 2025), so every contract pays transmission and
# distribution fees (115年費率).
FEE = rp.FEE_BASE + rp.FEE_TRANS + rp.FEE_DIST
# A Pmi change applies from the first day of the month after the change is completed
# (轉供契約 114.9.1 版 第十六條 五), so month t is decided with data through t-2.
LEAD = 2
LOAD_LAGS = (0, 1, 2, 11)          # load scenarios t-2, t-3, t-4, t-13
MAX_BACK = 36                      # fallback for a user without good data

def station_year_kwh(station, year=rp.YEAR):
    pm = rc._plant_months()
    rows = pm[(pm.year == year) & (pm.plant == station)]
    if len(rows) != 12:
        raise ValueError(f"{station}: {len(rows)} months in {year}")
    return float(rows.kwh.sum())

class ScaledEweld:
    """EWELD users, each multiplied by a fixed factor; load scenarios respect LEAD."""

    def __init__(self, ids, scale):
        self.ids, self.scale = list(ids), np.asarray(scale, float)

    @staticmethod
    def _ok(u, year, month):
        um = rc.user_month(Path(u), year, month)
        return um is not None and um[1] >= rc.MIN_GOOD

    def replay_kwh(self, u):
        return sum(rc.user_month(Path(u), rp.YEAR, m)[0].sum() for m in rp.MONTHS)

    def scenario_loads(self, t, year=rp.YEAR):
        """[(days, (n, SCEN_DAYS*96) kWh, source month per user)], all before month t-1."""
        out = []
        for lag in LOAD_LAGS:
            ym = rp.shift_month(year, t, -(LEAD + lag))
            rows, srcs = [], []
            for u, s in zip(self.ids, self.scale):
                src = ym if self._ok(u, *ym) else next(
                    (rp.shift_month(year, t, -k) for k in range(LEAD, MAX_BACK + 1)
                     if self._ok(u, *rp.shift_month(year, t, -k))), None)
                if src is None and year < rp.YEAR:
                    # Two EWELD users start in 2020-03 / 2020-04, so early-2020 training
                    # instances have no earlier load. A training instance only needs a
                    # plausible load, not the replay's information order, so look forward,
                    # never into the 2021 replay.
                    src = next((ym2 for k in range(1, 25) if (ym2 := rp.shift_month(*ym, k)) < (rp.YEAR, 1)
                                and self._ok(u, *ym2)), None)
                if src is None:
                    raise ValueError(f"{u}: no good month before {year}-{t:02d} within {MAX_BACK}")
                rows.append(rc.user_month(Path(u), *src)[0][: rp.SCEN_DAYS * 96] * s)
                srcs.append(src)
            out.append((rp.month_days(*ym)[: rp.SCEN_DAYS], np.stack(rows), srcs))
        return out

    def realized(self, t):
        rows = []
        for u, s in zip(self.ids, self.scale):
            um = rc.user_month(Path(u), rp.YEAR, t)
            if um is None or um[1] < rc.MIN_GOOD:
                raise ValueError(f"{u}: no realized load in {rp.YEAR}-{t:02d}")
            rows.append(um[0] * s)
        return rp.month_days(rp.YEAR, t), np.stack(rows)

class Retailer:
    """One archetype at supply ratio rho; seed draws stations, load shapes and prices."""

    def __init__(self, arch, rho, seed, customers=None):
        t = pd.read_csv(TREC, sep="\t")
        a = t[t.archetype == arch]
        sites = a[(a.kind == "site") & a.tech.isin(list(TECH))]
        buyers = np.sort(a[a.kind == "buyer"].mwh.to_numpy(float))[::-1]
        rng = np.random.default_rng(seed + 40_000)
        info = rp.plant_info()
        by_tech = {tc: [p for p in rp.plant_pool() if info[p][0] == tc] for tc in ("solar", "wind")}
        self.site_tech = sites.tech.map(TECH).to_numpy()
        self.site_kwh_year = sites.mwh.to_numpy(float) * 1000
        self.site_station = np.array([by_tech[tc][rng.integers(len(by_tech[tc]))] for tc in self.site_tech])

        # Site i produces its real T-REC volume over 2021, month by month like its station.
        self.stations = sorted(set(self.site_station))
        year = {p: station_year_kwh(p) for p in self.stations}
        self.site_weight = self.site_kwh_year / np.array([year[p] for p in self.site_station])
        self.group_weight = np.array([self.site_weight[self.site_station == p].sum() for p in self.stations])
        self.group_of_site = np.array([self.stations.index(p) for p in self.site_station])
        self.group_tech = [info[p][0] for p in self.stations]
        self.supply_replay = float(self.group_weight @ np.array([rp.supply_2021([p]) for p in self.stations]))

        # Buyer j holds its real share of the retailer's sales; its load over
        # the replayed months is that share of supply divided by rho.
        self.contract_share = buyers / buyers.sum()
        self.demand_replay = self.contract_share * self.supply_replay / rho
        cu = customers or rp.EweldCustomers()
        if len(buyers) > len(cu.ids):
            raise ValueError(f"{arch}: {len(buyers)} buyers but only {len(cu.ids)} EWELD users")
        ids = [cu.ids[i] for i in rng.choice(len(cu.ids), len(buyers), replace=False)]
        base = ScaledEweld(ids, np.ones(len(ids)))
        self.customers = ScaledEweld(ids, self.demand_replay / np.array([base.replay_kwh(u) for u in ids]))
        self.price = rng.uniform(*rp.GREEN_PRICE, len(buyers))
        self.arch, self.rho, self.seed = arch, rho, seed
        self.m, self.n = len(self.stations), len(buyers)

class RetailerMonth:
    """Scenarios (ex ante) and the realized month (ex post) for target month t of `year`.

    The replay is 2021; earlier years build surrogate training instances, which
    have scenarios only: generation from the GEN_YEARS before `year`, load as in 2021.
    """

    def __init__(self, ret: Retailer, t, year=rp.YEAR):
        self.ret, self.t, self.year = ret, t, year
        self.m, self.n = ret.m, ret.n
        self.days_t = len(rp.month_days(year, t))
        self.summer = rp.dt.date(year, 5, 16) <= rp.dt.date(year, t, 15) <= rp.dt.date(year, 10, 15)
        gen_months = [rp.shift_month(y, t, k) for y in rp.GEN_YEARS if y < year for k in (-1, 0, 1)]
        self.gen_months = [ym for ym in gen_months
                           if rp._gen(*ym) and all(p in rp._gen(*ym) for p in ret.stations)]
        loads = ret.customers.scenario_loads(t, year)
        R, D, tou, self.scen = [], [], [], []
        for gy in self.gen_months:
            G = self.station_gen(gy, rp.SCEN_DAYS * 96)
            for days, L, srcs in loads:
                R.append(G)
                D.append(L)
                tou.append(tou_labels_114(days, summer=self.summer))
                self.scen.append(dict(gen=gy, days=days, load_src=srcs))
        self.R, self.D, self.tou = np.stack(R), np.stack(D), np.stack(tou)
        self.batch = ScenarioBatchSettlement(self.R, self.D, self.tou)
        self.scale = self.days_t / rp.SCEN_DAYS
        self.real = None
        if year == rp.YEAR:
            days, L = ret.customers.realized(t)
            self.real = dict(R=self.station_gen((year, t), None), D=L, tou=tou_labels_114(days))

    def baseline(self):
        """The contract-share allocation as a retailer can write it.

        Pmi is filled in % to two decimals (轉供契約), so a share below 0.01% rounded
        cell by cell becomes 0 at every station and the buyer gets nothing. Instead
        each buyer's column is rounded with the remainder carried to the next station group,
        largest expected output first, so its expected total stays at its contract share and
        a small buyer sits on the fewest groups. Each row's rounding residue goes to the
        largest buyer so every group sells exactly 100%. Expected output is the scenario
        mean, which is all that is known when Pmi is decided.
        """
        share, g = self.ret.contract_share, self.R.mean(axis=0).sum(axis=1)
        W = np.zeros((self.m, self.n))
        order = np.argsort(-g, kind="stable")
        for j in range(self.n):
            got, cum = 0.0, 0.0
            for i in order:
                cum += g[i]
                W[i, j] = max(0.0, np.round((share[j] * cum - got) / g[i], 4))
                got += W[i, j] * g[i]
        big = int(np.argmax(share))
        W[:, big] = np.round(W[:, big] + 1 - W.sum(axis=1), 4)
        if (W < 0).any():
            raise ValueError("baseline: the largest buyer cannot absorb a row's rounding residue")
        return W

    def initial_population(self, size, rng):
        """The baseline plus size-1 uniform draws (the pymoo default): every search starts
        from the retailer's current allocation, as a retailer would."""
        return np.vstack([self.baseline().ravel(), rng.random((size - 1, self.m * self.n))])

    def station_gen(self, ym, T):
        g = rp._gen(*ym)
        return np.stack([self.ret.group_weight[i] * g[p]["generation"][:T] for i, p in enumerate(self.ret.stations)])

    def _metrics(self, s, R):
        """s (S, m, n) settled kWh, R (S, m, T) group generation, both for the whole month."""
        ret = self.ret
        green = s.sum(axis=1)                                                 # (S, n)
        profit = ((ret.price - rp.PPA_PRICE - FEE)[None] * green).sum(axis=1)
        due = ret.contract_share[None] * R.sum(axis=(1, 2))[:, None]          # (S, n)
        fulfil = np.minimum(1.0, green / np.maximum(due, 1e-9))
        return dict(profit=profit, worst=fulfil.min(axis=1), green=green, fulfil=fulfil)

    def evaluate(self, W, rounding=True):
        """Per-scenario metrics for a group decision W (m stations x n customers)."""
        W = np.round(W, 4)
        self.batch.rounding = rounding
        return self._metrics(self.batch.settle_quantities(W) * self.scale, self.R * self.scale)

    def evaluate_subset(self, W, idx):
        """The same metrics on scenarios `idx` only: the low-fidelity screen of the backtest."""
        key = tuple(int(i) for i in idx)
        cache = self.__dict__.setdefault("_subsets", {})
        if key not in cache:
            cache[key] = ScenarioBatchSettlement(self.R[list(key)], self.D[list(key)], self.tou[list(key)])
        W = np.round(W, 4)
        return self._metrics(cache[key].settle_quantities(W) * self.scale, self.R[list(key)] * self.scale)

    def evaluate_real(self, W):
        W = np.round(W, 4)
        r = self.real
        s = ScenarioBatchSettlement(r["R"][None], r["D"][None], r["tou"][None]).settle_quantities(W)
        return {k: v[0] for k, v in self._metrics(s, r["R"][None]).items()}

    def site_instance(self, scen):
        """Site-level generation (sites, T) of scenario `scen`, the oracle's input."""
        ret = self.ret
        g = rp._gen(*self.scen[scen]["gen"])
        return np.stack([ret.site_weight[i] * g[p]["generation"][: rp.SCEN_DAYS * 96]
                         for i, p in enumerate(ret.site_station)])

    def evaluate_sites(self, W, scens, rounding=True):
        """The same metrics settled site by site, each site taking its group's row of W.

        Uses ConsumerContractSettlement, the per-scenario path checked against
        settle_reference in check_consumer_contracts.py; independent of the batched code.
        """
        W = np.round(W, 4)
        Ws = W[self.ret.group_of_site]
        out = []
        for c in scens:
            G = self.site_instance(c)
            # Infinite capacity switches off the quarter-of-capacity clip of 第十三點: T-REC
            # gives each site's volume but not its kW, and the batched path has no clip either.
            inst = rp._instance(G, np.full(len(G), np.inf), list(self.ret.site_tech), self.D[c])
            settle = ConsumerContractSettlement(inst, self.tou[c], rounding=rounding)
            s = settle.settle_quantities(Ws) * self.scale                    # (sites, n)
            out.append(self._metrics(s.sum(axis=0, keepdims=True)[None].reshape(1, 1, -1),
                                     G[None] * self.scale))
        return {k: np.concatenate([o[k] for o in out]) for k in out[0]}
