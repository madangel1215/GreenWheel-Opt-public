#!/usr/bin/env python3
"""Checks for the 2021 replay (replay2021.py): batched exact settlement against the per-scenario oracle.

B1 batched profit, penalty and worst achievement equal the one-scenario-at-a-time path
   (ConsumerContractSettlement, itself checked against settle_reference in
   check_consumer_contracts.py) for dense, sparse, zero and full-row W.
B2 W = 0 gives zero profit and the full penalty requirement.
B3 fees: a transmission-level customer pays the distribution fee as soon as one
   distribution-level plant is in its contract.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.greenwheel.data import replay2021 as rp  # noqa: E402

def random_W(rng, m, n, kind):
    if kind == "zero":
        return np.zeros((m, n))
    W = rng.random((m, n))
    if kind == "sparse":
        W *= rng.random((m, n)) < 0.05
    s = W.sum(axis=1, keepdims=True)
    W = np.where(s > 0, W / np.maximum(s, 1e-12), 0.0)
    if kind == "partial":
        W *= rng.uniform(0.2, 0.9, (m, 1))
    return W

def main():
    bad = 0
    cu = rp.EweldCustomers()
    dem = sum(np.stack(cu.month(2021, m)).sum() for m in rp.MONTHS)
    rng = np.random.default_rng(7)
    for t, rho, seed in [(7, 1.0, 1), (1, 0.7, 2), (10, 1.5, 3)]:
        plants, _ = rp.portfolio(seed, rho, dem)
        mp = rp.MonthProblem(cu, rp.Market(cu, seed), plants, t)
        for kind in ("dense", "sparse", "partial", "zero") * 3:
            W = random_W(rng, mp.m, mp.n, kind)
            a, b = mp.evaluate(W), mp.evaluate_loop(W)
            ok = all(np.allclose(a[k], b[k], rtol=1e-9, atol=1e-6) for k in a)
            bad += not ok
            if not ok:
                print("FAIL B1", t, rho, kind, {k: np.abs(a[k] - b[k]).max() for k in a})
        z = mp.evaluate(np.zeros((mp.m, mp.n)))
        req = (rp.PENALTY_PER_SOLAR_KWH * mp.market.requirement / 12)[mp.market.obligated].sum()
        ok = np.all(z["profit"] == 0) and np.allclose(z["penalty"], req)
        bad += not ok
        print(("PASS" if ok else "FAIL"), "B2", t, rho, seed, "scenarios", len(mp.scen))
    # B3 on a hand-made situation: customer 0 at transmission level, plant 0 at
    # transmission level, plant 1 at distribution level.
    mp.market.ehv_customer = np.zeros(mp.n, bool)
    mp.market.ehv_customer[0] = True
    mp.ehv_plant = np.zeros(mp.m, bool)
    mp.ehv_plant[0] = True
    W = np.zeros((mp.m, mp.n))
    W[0, 0] = 0.5
    f1 = mp.fees(W)[0]
    W[1, 0] = 0.0001
    f2 = mp.fees(W)[0]
    W[1, 0] = 0.00004            # rounds below one hundredth of a percent: not in the contract
    f3 = mp.fees(np.round(W, 4))[0]
    ok = (np.isclose(f1, rp.FEE_BASE + rp.FEE_TRANS) and np.isclose(f2, f1 + rp.FEE_DIST)
          and np.isclose(f3, f1) and np.isclose(mp.fees(W)[1], rp.FEE_BASE + rp.FEE_TRANS + rp.FEE_DIST))
    bad += not ok
    print(("PASS" if ok else "FAIL"), "B3 fees", round(f1, 4), round(f2, 4), round(f3, 4))
    print("ALL PASS" if not bad else f"{bad} FAIL")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
