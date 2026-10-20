#!/usr/bin/env python3
"""ConsumerContractSettlement == settle_reference with one contract per consumer.

The reference is the step-by-step transcription of point 13 already checked in
check_settlement.py; here it is driven with consumer contracts (several generators,
one consumer each), with and without monthly caps, on random small instances that
include night-time zero output and output above a quarter of capacity.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.greenwheel.config import ConsumerSite, GeneratorPlant, MarketParams, WheelingInstance
from src.greenwheel.simulator.settlement import (
    ConsumerContractSettlement, consumer_contracts, settle_reference, tou_labels)

def make_instance(G, D, cap_kw):
    gens = [GeneratorPlant(id=i, capacity_kw=float(cap_kw[i]), technology="solar", ppa_price=0.0,
                           generation=np.asarray(G[i], float)) for i in range(len(G))]
    cons = [ConsumerSite(id=j, total_monthly_kwh=float(np.sum(D[j])), contract_kwh=0.0, re100_target=0.5,
                         retail_price=5.0, demand=np.asarray(D[j], float)) for j in range(len(D))]
    return WheelingInstance(generators=gens, consumers=cons, market=MarketParams(), n_intervals=len(D[0]))

rng = np.random.default_rng(29)
worst_int = worst_raw = 0.0
n_cap = 0
for _ in range(300):
    m, n, days = rng.integers(1, 5), rng.integers(1, 6), 7
    T = 96 * days
    cap_kw = rng.uniform(50, 400, m)
    hour = np.tile(np.repeat(np.arange(24), 4), days)
    sun = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)
    G = cap_kw[:, None] * 0.25 * sun[None, :] * rng.uniform(0.3, 1.4, (m, T))
    D = rng.uniform(5, 120, (n, T))
    W = rng.random((m, n)) * (rng.random((m, n)) < 0.7)
    W = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12) * rng.uniform(0.5, 1.0, (m, 1))
    W[W < 0.01] = 0.0
    inst = make_instance(G, D, cap_kw)
    tou = tou_labels(days, summer=bool(rng.random() < 0.5), start_weekday=int(rng.integers(7)))
    cap = None
    if rng.random() < 0.5:
        uncapped = ConsumerContractSettlement(inst, tou, stage2=False, rounding=False).settle_quantities(W).sum(axis=0)
        cap = np.where(rng.random(n) < 0.6, uncapped * rng.uniform(0.2, 0.9, n), np.inf)
        n_cap += int((uncapped > cap).any())
    ref, _ = settle_reference(inst, consumer_contracts(W, cap), tou)
    _, ref_raw = settle_reference(inst, consumer_contracts(W, cap), tou, rounding=False)
    fast = ConsumerContractSettlement(inst, tou).settle_quantities(W, cap)
    fast_raw = ConsumerContractSettlement(inst, tou, rounding=False).settle_quantities(W, cap)
    worst_int = max(worst_int, np.abs(ref - fast).max())
    worst_raw = max(worst_raw, np.abs(ref_raw - fast_raw).max() / max(ref_raw.max(), 1.0))

ok = worst_int == 0.0 and worst_raw < 1e-9
print(f"[{'PASS' if ok else 'FAIL'}] consumer contracts == reference: max int diff {worst_int}, "
      f"max rel unrounded diff {worst_raw:.2e} ({n_cap} of 300 cases with a binding cap)")
sys.exit(0 if ok else 1)
