"""
Wheeling settlement under Taipower Operating Rules point 13 (version 112-06-20).

Source: 台灣電力股份有限公司電能轉供及併網型直供營運規章, 第十三點 計費方式.
The rule settles per contract in three stages: 15-minute proportional matching
with monthly/annual caps, re-matching of leftovers within each time-of-use period
over the month, and per-period rounding to whole kWh.

`settle_reference` transcribes the rule step by step for any contract structure and
exists only as the oracle. `settle_pairs` is the fast path for the structure used in
this study, one contract per (generator, consumer) pair with share P = w_mn, so the
allocation matrix W keeps its meaning.
"""

from dataclasses import dataclass

import numpy as np

from src.greenwheel.config import WheelingInstance

PEAK, HALF_PEAK, OFF_PEAK, SAT_HALF_PEAK = 0, 1, 2, 3
N_PERIODS = 4

def tou_labels(n_days: int, summer: bool = True, start_weekday: int = 0,
               holidays: tuple[int, ...] = ()) -> np.ndarray:
    """Time-of-use period of every 15-minute interval.

    High-voltage three-period tariff with fixed peak hours, Taipower rate table
    effective 2024-04-01 (113-04-01): summer is 5/16-10/15. Weekday summer: peak
    16-22, half-peak 09-16 and 22-24, off-peak 00-09. Weekday non-summer: no peak,
    half-peak 06-11 and 14-24, off-peak 00-06 and 11-14. Saturday uses the weekday
    half-peak hours of its season as a separate period. Sundays and off-peak days
    are off-peak all day. `holidays` are day indices treated as off-peak days.
    """
    hour = np.repeat(np.arange(24), 4)
    if summer:
        weekday = np.where((hour >= 16) & (hour < 22), PEAK,
                           np.where(hour >= 9, HALF_PEAK, OFF_PEAK))
        saturday = np.where(hour >= 9, SAT_HALF_PEAK, OFF_PEAK)
    else:
        half = ((hour >= 6) & (hour < 11)) | (hour >= 14)
        weekday = np.where(half, HALF_PEAK, OFF_PEAK)
        saturday = np.where(half, SAT_HALF_PEAK, OFF_PEAK)
    days = []
    for d in range(n_days):
        dow = (start_weekday + d) % 7
        if d in holidays or dow == 6:
            days.append(np.full(96, OFF_PEAK))
        elif dow == 5:
            days.append(saturday)
        else:
            days.append(weekday)
    return np.concatenate(days)

def round_half_up(x):
    # 四捨五入; numpy's round is half-to-even.
    return np.floor(np.asarray(x) + 0.5)

@dataclass
class SettlementResult:
    settled: np.ndarray          # (m, n) whole kWh wheeled from m to n this month
    delivered_per_con: np.ndarray
    total_generation: float
    total_settled: float
    total_surplus: float
    total_shortfall: float
    profit: float
    revenue: float
    gen_cost: float
    wheeling_cost: float
    surplus_penalty: float

def _objectives(settled, instance, G_raw, D) -> SettlementResult:
    market = instance.market
    delivered = settled.sum(axis=0)
    retail = np.array([c.retail_price for c in instance.consumers])
    ppa = np.array([g.ppa_price for g in instance.generators])
    r = np.array([c.re100_target for c in instance.consumers])
    total_gen = float(G_raw.sum())
    total_settled = float(settled.sum())
    surplus = total_gen - total_settled
    # Monthly RE100 shortfall: settled wheeling cannot be traced back to intervals
    # once stage 2 re-matches leftovers across the month.
    shortfall = float(np.maximum(0.0, r * D.sum(axis=1) - delivered).sum())
    revenue = float(retail @ delivered)
    gen_cost = float(ppa @ G_raw.sum(axis=1))
    wheeling_cost = float(market.wheeling_fee * total_settled)
    surplus_penalty = float(market.surplus_price * surplus)
    return SettlementResult(
        settled=settled, delivered_per_con=delivered, total_generation=total_gen,
        total_settled=total_settled, total_surplus=surplus, total_shortfall=shortfall,
        profit=revenue - gen_cost - wheeling_cost - surplus_penalty, revenue=revenue,
        gen_cost=gen_cost, wheeling_cost=wheeling_cost, surplus_penalty=surplus_penalty,
    )

def _profiles(instance):
    G_raw = np.stack([g.generation for g in instance.generators])
    D = np.stack([c.demand for c in instance.consumers])
    cap_kw = np.array([g.capacity_kw for g in instance.generators])
    # 第十三點 (二)1 and (三): output above a quarter of capacity per 15 min is not wheeled.
    R = np.minimum(G_raw, cap_kw[:, None] * 0.25)
    return G_raw, D, cap_kw, R

def settle_reference(instance: WheelingInstance, contracts: list[dict], tou: np.ndarray,
                     stage2: bool = True, rounding: bool = True):
    """Literal transcription of point 13 for an arbitrary contract structure.

    Each contract is {"gens": {m: P_mi}, "cons": [n, ...], "cap": {n: min(M_ni, Y_ni)}}
    (missing cap means unlimited). Returns the (m, n) settled matrix and, when
    rounding is off, the unrounded one. Slow by design: it is the oracle.
    """
    G_raw, D, cap_kw, R = _profiles(instance)
    m, n = len(cap_kw), D.shape[0]
    T = D.shape[1]
    I = len(contracts)
    rem = [{c: k["cap"].get(c, np.inf) for c in k["cons"]} for k in contracts]
    contracts_of = {c: [i for i, k in enumerate(contracts) if c in k["cons"]] for c in range(n)}
    q_per = np.zeros((I, m, n, N_PERIODS))
    B_per = np.zeros((I, m, N_PERIODS))
    S_per = np.zeros((I, n, N_PERIODS))

    for t in range(T):
        p = tou[t]
        # Step 1: G_mi = R_m * P_mi
        Gmi = [{g: R[g, t] * P for g, P in k["gens"].items()} for k in contracts]
        Gsum = [sum(x.values()) for x in Gmi]
        # Step 2: C_ni by each contract's share of generation among the consumer's contracts
        Cni = [dict() for _ in range(I)]
        for c, ks in contracts_of.items():
            total = sum(Gsum[i] for i in ks)
            if total > 0:
                for i in ks:
                    Cni[i][c] = D[c, t] * Gsum[i] / total
            else:
                # All contracts at zero output: split by capacity times contract share.
                w = {i: sum(cap_kw[g] * P for g, P in contracts[i]["gens"].items()) for i in ks}
                wt = sum(w.values())
                for i in ks:
                    Cni[i][c] = D[c, t] * w[i] / wt if wt > 0 else 0.0
        for i, k in enumerate(contracts):
            U = {c: min(rem[i][c], Cni[i][c]) for c in k["cons"]}
            Usum = sum(U.values())
            Q = min(Gsum[i], Usum)  # step 3
            for c in k["cons"]:
                qn = Q * U[c] / Usum if Usum > 0 else 0.0  # step 4
                for g in k["gens"]:
                    qmn = qn * Gmi[i][g] / Gsum[i] if Gsum[i] > 0 else 0.0  # step 5
                    q_per[i, g, c, p] += qmn
                rem[i][c] -= qn  # step 6
                S_per[i, c, p] += Cni[i][c] - qn  # step 9
            for g in k["gens"]:
                delivered_g = sum(Q * U[c] / Usum * Gmi[i][g] / Gsum[i]
                                  for c in k["cons"]) if Usum > 0 and Gsum[i] > 0 else 0.0
                B_per[i, g, p] += Gmi[i][g] - delivered_g  # step 8

    qq_per = np.zeros_like(q_per)
    if stage2:
        for i, k in enumerate(contracts):
            for p in range(N_PERIODS):
                UU = {}
                for c in k["cons"]:
                    A = max(rem[i][c], 0.0)  # step 10: remaining min(M, Y)
                    Ssum = S_per[i, c].sum()
                    Ap = A * S_per[i, c, p] / Ssum if Ssum > 0 else 0.0
                    if np.isinf(A):
                        Ap = np.inf if S_per[i, c, p] > 0 else 0.0
                    UU[c] = min(Ap, S_per[i, c, p])  # step 11
                UUsum = sum(UU.values())
                Bsum = sum(B_per[i, g, p] for g in k["gens"])
                QQ = min(Bsum, UUsum)  # step 12
                for c in k["cons"]:
                    qqn = QQ * UU[c] / UUsum if UUsum > 0 else 0.0  # step 13
                    for g in k["gens"]:
                        qq_per[i, g, c, p] = qqn * B_per[i, g, p] / Bsum if Bsum > 0 else 0.0  # step 14

    per_period = q_per + qq_per
    unrounded = per_period.sum(axis=(0, 3))
    if rounding:
        # Step 15: round each contract-pair-period total, then add the periods.
        settled = round_half_up(per_period).sum(axis=(0, 3))
    else:
        settled = unrounded
    return settled, unrounded

def pair_contracts(W: np.ndarray, cap: np.ndarray | None = None) -> list[dict]:
    """One contract per pair with W > 0, in the form settle_reference expects."""
    out = []
    for g, c in zip(*np.nonzero(W)):
        k = {"gens": {int(g): float(W[g, c])}, "cons": [int(c)], "cap": {}}
        if cap is not None and np.isfinite(cap[g, c]):
            k["cap"][int(c)] = float(cap[g, c])
        out.append(k)
    return out

class PairSettlement:
    """Point-13 settlement with one contract per pair, vectorised over pairs.

    Per-instance quantities are computed once; each call costs a few (m x T)(T x n)
    products. With one generator and one consumer per contract, stage 1 reduces to
    q = min(w R_m, C, remaining cap), and C splits consumer demand in proportion to
    w R_m, so before caps bind q_mn(t) = w_mn R_m(t) gamma_n(t) as in eq. (cap_ratio).
    A cap only truncates the running total, so the time series of a pair is needed
    only for pairs whose uncapped monthly total exceeds their cap.
    """

    def __init__(self, instance: WheelingInstance, tou: np.ndarray,
                 stage2: bool = True, rounding: bool = True):
        self.instance = instance
        self.G_raw, self.D, self.cap_kw, self.R = _profiles(instance)
        self.tou = tou
        self.masks = [tou == p for p in range(N_PERIODS)]
        self.stage2 = stage2
        self.rounding = rounding
        self.R_p = np.stack([self.R[:, mk].sum(axis=1) for mk in self.masks], axis=1)  # (m, P)

    def settle(self, W: np.ndarray, cap: np.ndarray | None = None) -> SettlementResult:
        R, D, cap_kw = self.R, self.D, self.cap_kw
        m, n = W.shape
        P = N_PERIODS
        tot = W.T @ R                                    # (n, T) generation offered to n
        live = tot > 0
        gamma = np.where(live, np.minimum(1.0, D / np.where(live, tot, 1.0)), 0.0)
        ratio = np.where(live, D / np.where(live, tot, 1.0), 0.0)
        capW = cap_kw @ W                                # (n,) fallback weights
        dead_D = np.where(live, 0.0, D)                  # demand when no contract produces

        q_p = np.empty((m, n, P))
        C_p = np.empty((m, n, P))
        for p, mk in enumerate(self.masks):
            q_p[:, :, p] = W * (R[:, mk] @ gamma[:, mk].T)
            C_live = W * (R[:, mk] @ ratio[:, mk].T)
            with np.errstate(divide="ignore", invalid="ignore"):
                share = np.where(capW > 0, dead_D[:, mk].sum(axis=1) / np.where(capW > 0, capW, 1.0), 0.0)
            C_p[:, :, p] = C_live + W * cap_kw[:, None] * share[None, :]

        remaining = np.full((m, n), np.inf)
        if cap is not None:
            total = q_p.sum(axis=2)
            remaining = cap - total
            gs, cs = np.nonzero(total > cap)
            if len(gs):
                q_p[gs, cs], remaining[gs, cs] = self._capped(W, gamma, gs, cs, cap[gs, cs])

        per_period = q_p
        if self.stage2:
            B = W[:, :, None] * self.R_p[:, None, :] - q_p
            S = C_p - q_p
            S_sum = S.sum(axis=2, keepdims=True)
            A = np.maximum(remaining, 0.0)[:, :, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                A_p = np.where(np.isinf(A), np.where(S > 0, np.inf, 0.0),
                               np.where(S_sum > 0, A * S / np.where(S_sum > 0, S_sum, 1.0), 0.0))
            UU = np.minimum(A_p, S)
            per_period = q_p + np.minimum(B, UU)

        settled = round_half_up(per_period).sum(axis=2) if self.rounding else per_period.sum(axis=2)
        return _objectives(settled, self.instance, self.G_raw, D)

    def _capped(self, W, gamma, gs, cs, cap):
        """Stage-1 per-period totals for pairs whose cap binds during the month.

        A capped pair needs its running total, but only the day in which the cap is
        reached needs interval detail: whole days before it count in full and days
        after it count nothing. Day-by-period totals come from small matrix products.
        """
        R, tou = self.R, self.tou
        k, T = len(gs), R.shape[1]
        n_days = T // 96
        # One (m x |block|)(|block| x n) product per day-period block costs m*n*T in
        # total however many pairs are capped; per-pair elementwise sums cost k*T.
        Z = np.zeros((R.shape[0], gamma.shape[0], n_days, N_PERIODS))
        for d in range(n_days):
            sl = np.arange(96 * d, 96 * (d + 1))
            for p in range(N_PERIODS):
                blk = sl[tou[sl] == p]
                if len(blk):
                    Z[:, :, d, p] = R[:, blk] @ gamma[:, blk].T
        x_dp = Z[gs, cs] * W[gs, cs][:, None, None]
        cum_day = np.cumsum(x_dp.sum(axis=2), axis=1)
        cross = np.argmax(cum_day > cap[:, None], axis=1)        # day the cap is reached
        before = np.arange(n_days)[None, :] < cross[:, None]
        q = (x_dp * before[:, :, None]).sum(axis=1)
        prior = np.where(cross > 0, cum_day[np.arange(k), np.maximum(cross - 1, 0)], 0.0)
        idx = (96 * cross)[:, None] + np.arange(96)[None, :]
        x_t = W[gs, cs][:, None] * R[gs[:, None], idx] * gamma[cs[:, None], idx]
        cq = np.minimum(prior[:, None] + np.cumsum(x_t, axis=1), cap[:, None])
        qt = np.diff(cq, axis=1, prepend=prior[:, None])
        lab = tou[idx]
        for p in range(N_PERIODS):
            q[:, p] += np.where(lab == p, qt, 0.0).sum(axis=1)
        return q, cap - cq[:, -1]

def consumer_contracts(W: np.ndarray, cap: np.ndarray | None = None) -> list[dict]:
    """One contract per consumer holding every generator with W > 0, for settle_reference."""
    out = []
    for c in range(W.shape[1]):
        gens = {int(g): float(W[g, c]) for g in np.nonzero(W[:, c])[0]}
        if not gens:
            continue
        k = {"gens": gens, "cons": [c], "cap": {}}
        if cap is not None and np.isfinite(cap[c]):
            k["cap"][c] = float(cap[c])
        out.append(k)
    return out

class ConsumerContractSettlement:
    """Point-13 settlement with one contract per consumer, vectorised.

    A consumer belongs to a single contract, so its whole demand is that contract's
    C_n and stage 1 delivers Q_n(t) = min(sum_m W_mn R_m(t), D_n(t), remaining cap).
    Writing eta_n(t) = Q_n(t) / offered_n(t), each pair gets W_mn R_m(t) eta_n(t), and a
    cap only changes eta along one consumer's running total, so caps cost n x T, not
    m x n x T. Stage 2 pools the contract's leftover generation across its generators.
    """

    def __init__(self, instance: WheelingInstance, tou: np.ndarray, stage2: bool = True, rounding: bool = True):
        self.instance = instance
        self.G_raw, self.D, self.cap_kw, self.R = _profiles(instance)
        self.masks = [tou == p for p in range(N_PERIODS)]
        self.stage2, self.rounding = stage2, rounding
        self.R_p = np.stack([self.R[:, mk].sum(axis=1) for mk in self.masks], axis=1)   # (m, P)

    def settle_quantities(self, W: np.ndarray, cap: np.ndarray | None = None) -> np.ndarray:
        return self.settle_detail(W, cap)[0].sum(axis=2)

    def settle_detail(self, W: np.ndarray, cap: np.ndarray | None = None):
        """(settled per pair and period (m, n, P), stage-1 delivery per consumer (n, T))."""
        R, D = self.R, self.D
        offered = W.T @ R                                              # (n, T)
        Q = np.minimum(offered, D)
        if cap is not None:
            cum = np.minimum(np.cumsum(Q, axis=1), np.where(np.isfinite(cap), cap, np.inf)[:, None])
            Q = np.diff(cum, axis=1, prepend=0.0)
        live = offered > 0
        eta = np.where(live, Q / np.where(live, offered, 1.0), 0.0)
        q = np.stack([W * (R[:, mk] @ eta[:, mk].T) for mk in self.masks], axis=-1)   # (m, n, P)
        per_period = q
        if self.stage2:
            B = W[:, :, None] * self.R_p[:, None, :] - q
            Bsum = B.sum(axis=0)                                       # (n, P)
            S = np.stack([(D - Q)[:, mk].sum(axis=1) for mk in self.masks], axis=1)   # (n, P)
            if cap is None:
                UU = np.minimum(np.where(S > 0, np.inf, 0.0), S)
            else:
                A = np.maximum(np.where(np.isfinite(cap), cap - Q.sum(axis=1), np.inf), 0.0)[:, None]
                Ssum = S.sum(axis=1, keepdims=True)
                with np.errstate(divide="ignore", invalid="ignore"):
                    A_p = np.where(np.isinf(A), np.where(S > 0, np.inf, 0.0),
                                   np.where(Ssum > 0, A * S / np.where(Ssum > 0, Ssum, 1.0), 0.0))
                UU = np.minimum(A_p, S)
            QQ = np.minimum(Bsum, UU)
            with np.errstate(divide="ignore", invalid="ignore"):
                share = np.where(Bsum[None] > 0, B / np.where(Bsum[None] > 0, Bsum[None], 1.0), 0.0)
            per_period = q + QQ[None] * share
        return (round_half_up(per_period) if self.rounding else per_period), Q

class ScenarioBatchSettlement:
    """ConsumerContractSettlement for one W over S scenarios of equal length, without caps.

    Same arithmetic as the per-scenario class, batched over a leading scenario axis so a
    multi-scenario evaluation is a few batched matmuls instead of S Python loops.
    """

    def __init__(self, R: np.ndarray, D: np.ndarray, tou: np.ndarray, rounding: bool = True):
        # R (S, m, T) wheelable generation, D (S, n, T) demand, tou (S, T) periods.
        self.R, self.D, self.rounding = R, D, rounding
        self.mask = np.stack([(tou == p) for p in range(N_PERIODS)], axis=1).astype(R.dtype)   # (S, P, T)
        S, m, T = R.shape
        # R split by period, stacked as rows (S, m*P, T), so stage 1 per period is one batched matmul.
        self.RM = (R[:, :, None, :] * self.mask[:, None, :, :]).reshape(S, m * N_PERIODS, T)
        self.R_p = self.RM.sum(axis=2).reshape(S, m, N_PERIODS)
        self.Rt = np.ascontiguousarray(R)

    def settle_quantities(self, W: np.ndarray) -> np.ndarray:
        """Settled kWh per scenario and pair, (S, m, n)."""
        D = self.D
        S, m, T = self.Rt.shape
        offered = np.matmul(W.T[None], self.Rt)                                  # (S, n, T)
        Q = np.minimum(offered, D)
        live = offered > 0
        eta = np.where(live, Q / np.where(live, offered, 1.0), 0.0)
        # q[s, m, n, p] = W[m, n] * sum_t R[s, m, t] * eta[s, n, t] * mask[s, p, t]
        q = np.matmul(self.RM, eta.transpose(0, 2, 1)).reshape(S, m, N_PERIODS, -1).transpose(0, 1, 3, 2)
        q = W[None, :, :, None] * q
        B = W[None, :, :, None] * self.R_p[:, :, None, :] - q
        Bsum = B.sum(axis=1)                                                     # (S, n, P)
        Sl = np.matmul(D - Q, self.mask.transpose(0, 2, 1))                      # (S, n, P)
        QQ = np.minimum(Bsum, np.minimum(np.where(Sl > 0, np.inf, 0.0), Sl))
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(Bsum[:, None] > 0, B / np.where(Bsum[:, None] > 0, Bsum[:, None], 1.0), 0.0)
        per_period = q + QQ[:, None] * share
        if self.rounding:
            per_period = round_half_up(per_period)
        return per_period.sum(axis=-1)

# Taipower tariff table effective 2018-04-01 (107年4月1日), in force until 2022-06,
# chapter 5, high voltage three-period, fixed peak. Summer is 6/1-9/30. 離峰日: 1/1, 農曆除夕 to 1月5日,
# 2/28, 4/4, 清明, 5/1, 端午 (農曆5/5), 中秋 (農曆8/15), 10/10. Lunar dates resolved by hand.
OFF_PEAK_DAYS_2018_TABLE = {
    2018: ["01-01", "02-15", "02-16", "02-17", "02-18", "02-19", "02-20", "02-28", "04-04", "04-05",
           "05-01", "06-18", "09-24", "10-10"],
    2019: ["01-01", "02-04", "02-05", "02-06", "02-07", "02-08", "02-09", "02-28", "04-04", "04-05",
           "05-01", "06-07", "09-13", "10-10"],
    2020: ["01-01", "01-24", "01-25", "01-26", "01-27", "01-28", "01-29", "02-28", "04-04",
           "05-01", "06-25", "10-01", "10-10"],
    2021: ["01-01", "02-11", "02-12", "02-13", "02-14", "02-15", "02-16", "02-28", "04-04",
           "05-01", "06-14", "09-21", "10-10"],
}
# Energy charges (元/度) of the same table, high voltage: (summer, non-summer).
TOU_PRICE_2018_TABLE = {PEAK: (4.67, None), HALF_PEAK: (2.90, 2.82), OFF_PEAK: (1.32, 1.26),
                        SAT_HALF_PEAK: (1.78, 1.71)}

def tou_labels_2018_table(year: int, month: int) -> np.ndarray:
    """Period of each 15-minute interval of a month under the 2018-04-01 tariff table."""
    import datetime as _dt
    summer = 6 <= month <= 9
    minute = np.arange(96) * 15
    half = (minute >= 7 * 60 + 30) & (minute < 22 * 60 + 30)
    if summer:
        peak = ((minute >= 600) & (minute < 720)) | ((minute >= 780) & (minute < 1020))
        weekday = np.where(peak, PEAK, np.where(half, HALF_PEAK, OFF_PEAK))
    else:
        weekday = np.where(half, HALF_PEAK, OFF_PEAK)
    saturday = np.where(half, SAT_HALF_PEAK, OFF_PEAK)
    holidays = set(OFF_PEAK_DAYS_2018_TABLE[year])
    first = _dt.date(year, month, 1)
    days = ((first.replace(year=year + (month == 12), month=month % 12 + 1)) - first).days
    out = []
    for d in range(days):
        day = first + _dt.timedelta(days=d)
        if day.strftime("%m-%d") in holidays or day.weekday() == 6:
            out.append(np.full(96, OFF_PEAK))
        elif day.weekday() == 5:
            out.append(saturday)
        else:
            out.append(weekday)
    return np.concatenate(out)

# Off-peak days of the tariff table in force from 2025-10-01 (114年10月1日), 註1 of the high
# voltage three-period tariff: 1/1, 農曆除夕前1日 to 正月初五, 2/28, 4/4, 清明 (4/4 or
# 4/5), 5/1, 端午, 中秋, 9/28, 10/10, 10/25, 12/25. Lunar dates from lunardate 0.2.2, checked
# by hand; only the calendar years whose load is replayed are listed.
OFF_PEAK_DAYS_114_TABLE = {
    # 2018-2019 for load scenarios reaching back past 2020 under the two-month lead time.
    2018: ["01-01", "02-14", "02-15", "02-16", "02-17", "02-18", "02-19", "02-20", "02-28", "04-04",
           "04-05", "05-01", "06-18", "09-24", "09-28", "10-10", "10-25", "12-25"],
    2019: ["01-01", "02-03", "02-04", "02-05", "02-06", "02-07", "02-08", "02-09", "02-28", "04-04",
           "04-05", "05-01", "06-07", "09-13", "09-28", "10-10", "10-25", "12-25"],
    2020: ["01-01", "01-23", "01-24", "01-25", "01-26", "01-27", "01-28", "01-29", "02-28", "04-04",
           "05-01", "06-25", "09-28", "10-01", "10-10", "10-25", "12-25"],
    2021: ["01-01", "02-10", "02-11", "02-12", "02-13", "02-14", "02-15", "02-16", "02-28", "04-04",
           "05-01", "06-14", "09-21", "09-28", "10-10", "10-25", "12-25"],
}

def is_summer_114(day) -> bool:
    return (day.month, day.day) >= (5, 16) and (day.month, day.day) <= (10, 15)

def tou_labels_114(days, summer: bool | None = None) -> np.ndarray:
    """Period of each 15-minute interval of `days` (datetime.date) under the 2025-10-01 table.

    Weekday summer: peak 16-22, half-peak 09-16 and 22-24. Weekday non-summer: half-peak
    06-11 and 14-24. Saturday: the same half-peak hours, as Saturday half-peak. Sundays
    and off-peak days: off-peak. Summer (5/16-10/15) is decided per day unless `summer`
    forces it, which the replay uses to put a scenario's calendar into the target month's season.
    """
    hour = np.repeat(np.arange(24), 4)
    sw = np.where((hour >= 16) & (hour < 22), PEAK, np.where(hour >= 9, HALF_PEAK, OFF_PEAK))
    s_half = hour >= 9
    n_half = ((hour >= 6) & (hour < 11)) | (hour >= 14)
    out = []
    for day in days:
        hol = day.strftime("%m-%d") in OFF_PEAK_DAYS_114_TABLE[day.year]
        su = is_summer_114(day) if summer is None else summer
        if hol or day.weekday() == 6:
            out.append(np.full(96, OFF_PEAK))
        elif day.weekday() == 5:
            out.append(np.where(s_half if su else n_half, SAT_HALF_PEAK, OFF_PEAK))
        else:
            out.append(sw if su else np.where(n_half, HALF_PEAK, OFF_PEAK))
    return np.concatenate(out)

def tou_prices_2018_table(month: int) -> np.ndarray:
    col = 0 if 6 <= month <= 9 else 1
    return np.array([TOU_PRICE_2018_TABLE[p][col] if TOU_PRICE_2018_TABLE[p][col] is not None else np.nan
                     for p in range(N_PERIODS)])
