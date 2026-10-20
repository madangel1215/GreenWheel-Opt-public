#!/usr/bin/env python3
"""Acceptance checks for the point-13 settlement.

A1  fast pair settlement == literal transcription, 300 random small instances
A2  special case (pair contracts, no caps, stage 1 only, no rounding) == simulate_matching
A3  hand-computed examples
A4  cost per candidate on an xlarge semi-real instance (measurement only)

Exits non-zero on any failed check.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.greenwheel.config import ConsumerSite, GeneratorPlant, MarketParams, WheelingInstance
from src.greenwheel.simulator.matching import simulate_matching
from src.greenwheel.simulator.settlement import (
    PairSettlement, pair_contracts, settle_reference, tou_labels)

failures = []

def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)

def make_instance(G, D, cap_kw):
    gens = [GeneratorPlant(id=i, capacity_kw=float(cap_kw[i]), technology="solar",
                           ppa_price=3.0, generation=np.asarray(G[i], float)) for i in range(len(G))]
    cons = [ConsumerSite(id=j, total_monthly_kwh=float(np.sum(D[j])), contract_kwh=0.0,
                         re100_target=0.8, retail_price=5.0, demand=np.asarray(D[j], float))
            for j in range(len(D))]
    return WheelingInstance(generators=gens, consumers=cons, market=MarketParams(),
                            n_intervals=len(D[0]))

def random_case(rng):
    m, n, days = rng.integers(1, 5), rng.integers(1, 6), 7
    T = 96 * days
    cap_kw = rng.uniform(50, 400, m)
    hour = np.tile(np.repeat(np.arange(24), 4), days)
    sun = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)  # zero at night
    G = cap_kw[:, None] * 0.25 * sun[None, :] * rng.uniform(0.3, 1.4, (m, T))  # >1/4 cap at times
    D = rng.uniform(5, 120, (n, T))
    W = rng.random((m, n)) * (rng.random((m, n)) < 0.7)
    if n > 1 and rng.random() < 0.3:
        W[:, rng.integers(n)] = 0.0  # a consumer nobody serves
    W = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12) * rng.uniform(0.5, 1.0, (m, 1))
    W[W < 0.01] = 0.0
    inst = make_instance(G, D, cap_kw)
    cap = None
    if rng.random() < 0.6:
        uncapped = W * (np.minimum(G, cap_kw[:, None] * 0.25).sum(axis=1)[:, None])
        cap = np.where(rng.random((m, n)) < 0.5, uncapped * rng.uniform(0.1, 0.9, (m, n)), np.inf)
    tou = tou_labels(days, summer=bool(rng.random() < 0.5), start_weekday=int(rng.integers(7)))
    return inst, W, cap, tou

# ---- A1 ----------------------------------------------------------------------
rng = np.random.default_rng(20261015)
worst_int, worst_raw, n_binding = 0.0, 0.0, 0
for _ in range(300):
    inst, W, cap, tou = random_case(rng)
    ref, _ = settle_reference(inst, pair_contracts(W, cap), tou)
    _, ref_raw = settle_reference(inst, pair_contracts(W, cap), tou, rounding=False)
    fast = PairSettlement(inst, tou).settle(W, cap)
    fast_raw = PairSettlement(inst, tou, rounding=False).settle(W, cap)
    worst_int = max(worst_int, np.abs(ref - fast.settled).max())
    worst_raw = max(worst_raw, np.abs(ref_raw - fast_raw.settled).max() / max(ref_raw.max(), 1.0))
    if cap is not None:
        uncapped = PairSettlement(inst, tou, stage2=False, rounding=False).settle(W).settled
        n_binding += int((uncapped > cap).any())
check("A1 rounded settlement identical", worst_int == 0.0, f"max abs diff {worst_int}")
check("A1 unrounded settlement agrees", worst_raw < 1e-9, f"max rel diff {worst_raw:.2e}")
print(f"     ({n_binding} of 300 cases had at least one binding cap)")

# ---- A2 ----------------------------------------------------------------------
worst = 0.0
for _ in range(50):
    inst, W, _, tou = random_case(rng)
    for g in inst.generators:  # keep output within a quarter of capacity: the paper has no such cap
        g.generation = np.minimum(g.generation, g.capacity_kw * 0.25)
    fast = PairSettlement(inst, tou, stage2=False, rounding=False).settle(W)
    G = np.stack([g.generation for g in inst.generators])
    D = np.stack([c.demand for c in inst.consumers])
    tot = W.T @ G
    eff_paper = np.minimum(tot, D).sum(axis=1)  # from simulate_matching: min(total_to_con, D)
    sim = simulate_matching(W, inst)
    check_total = abs(sim.total_delivered - eff_paper.sum()) / max(sim.total_delivered, 1.0)
    worst = max(worst, np.abs(fast.delivered_per_con - eff_paper).max() / max(eff_paper.max(), 1.0),
                abs(fast.total_settled - sim.total_delivered) / max(sim.total_delivered, 1.0), check_total)
check("A2 stage-1 special case == simulate_matching", worst < 1e-9, f"max rel diff {worst:.2e}")

# ---- A3 ----------------------------------------------------------------------
# One generator, one consumer, two intervals in the same period (day 0 is Monday, 00:00
# and 00:15 are off-peak). Generation 10 then 0; demand 4 then 6.
# Stage 1: q = 4 then 0. Leftover B = 6, unmet S = 0 + 6 = 6. Stage 2: QQ = min(6, 6) = 6.
G = np.zeros((1, 96)); G[0, 0] = 10.0
D = np.zeros((1, 96)); D[0, 0], D[0, 1] = 4.0, 6.0
inst = make_instance(G, D, [1000.0])
tou = tou_labels(1)
r = PairSettlement(inst, tou).settle(np.array([[1.0]]))
check("A3a stage 2 re-matches the leftover", r.settled[0, 0] == 10.0, f"settled {r.settled[0, 0]}")
r1 = PairSettlement(inst, tou, stage2=False).settle(np.array([[1.0]]))
check("A3b stage 1 alone", r1.settled[0, 0] == 4.0, f"settled {r1.settled[0, 0]}")
# Same, but generation 10 at 00:00 and demand 6 at 16:00 (peak): different periods, no re-match.
D2 = np.zeros((1, 96)); D2[0, 0], D2[0, 64] = 4.0, 6.0
r2 = PairSettlement(make_instance(G, D2, [1000.0]), tou).settle(np.array([[1.0]]))
check("A3c no re-match across periods", r2.settled[0, 0] == 4.0, f"settled {r2.settled[0, 0]}")
# Monthly cap 3: stage 1 stops at 3, nothing left for stage 2.
r3 = PairSettlement(inst, tou).settle(np.array([[1.0]]), cap=np.array([[3.0]]))
check("A3d monthly cap binds", r3.settled[0, 0] == 3.0, f"settled {r3.settled[0, 0]}")
# Two generators (6 and 2 kWh) feed one consumer needing 4: curtailed in proportion 3 and 1.
G4 = np.zeros((2, 96)); G4[0, 0], G4[1, 0] = 6.0, 2.0
D4 = np.zeros((1, 96)); D4[0, 0] = 4.0
r4 = PairSettlement(make_instance(G4, D4, [1000.0, 1000.0]), tou, stage2=False).settle(np.ones((2, 1)))
check("A3e proportional split", list(r4.settled[:, 0]) == [3.0, 1.0], f"settled {r4.settled[:, 0]}")
# Output above a quarter of capacity is not wheeled: capacity 20 kW -> 5 kWh per 15 min.
r5 = PairSettlement(make_instance(G, D, [20.0]), tou).settle(np.array([[1.0]]))
check("A3f quarter-capacity cap", r5.settled[0, 0] == 5.0, f"settled {r5.settled[0, 0]}")

# Tariff calendar against the 113-04-01 rate table, read off by hand (day 0 = Monday).
def label(lab, day, hh, mm=0):
    return int(lab[day * 96 + hh * 4 + mm // 15])
su, ns = tou_labels(7, summer=True), tou_labels(7, summer=False)
expected = [(su, 0, 16, 0, 0), (su, 0, 21, 45, 0), (su, 0, 22, 0, 1), (su, 0, 9, 0, 1), (su, 0, 8, 45, 2),
            (su, 5, 9, 0, 3), (su, 5, 8, 45, 2), (su, 5, 20, 0, 3), (su, 6, 18, 0, 2),
            (ns, 0, 7, 0, 1), (ns, 0, 12, 0, 2), (ns, 0, 18, 0, 1), (ns, 0, 5, 45, 2),
            (ns, 5, 12, 0, 2), (ns, 5, 15, 0, 3), (ns, 6, 15, 0, 2)]
bad = [(d, h, mi, want, label(lab, d, h, mi)) for lab, d, h, mi, want in expected if label(lab, d, h, mi) != want]
check("A3g tariff calendar", not bad, f"mismatches {bad}" if bad else "")
holiday = tou_labels(7, summer=True, holidays=(2,))
check("A3h off-peak day", (holiday[2 * 96:3 * 96] == 2).all())

# ---- A4 ----------------------------------------------------------------------
if "--cost" in sys.argv:
    from src.greenwheel.config import WheelingConfig
    from src.greenwheel.data.synthetic import generate_semireal_instance
    cfg = WheelingConfig(); cfg.n_generators, cfg.n_consumers = 50, 100
    cfg.exclude_offline_plants = True
    inst = generate_semireal_instance(cfg, seed=42 + 74000)
    tou = tou_labels(cfg.n_days, summer=True)
    ps = PairSettlement(inst, tou)
    rng = np.random.default_rng(0)
    Ws = []
    for _ in range(50):
        W = rng.random((50, 100)) * (rng.random((50, 100)) < 0.3)
        Ws.append(W / W.sum(axis=1, keepdims=True))
    t = time.time(); [ps.settle(W) for W in Ws]; t_free = (time.time() - t) / len(Ws)
    caps = []
    for W in Ws:
        total = PairSettlement(inst, tou, stage2=False, rounding=False).settle(W).settled
        caps.append(np.where(rng.random(W.shape) < 0.5, total * 0.6, np.inf))
    t = time.time(); [ps.settle(W, c) for W, c in zip(Ws, caps)]; t_cap = (time.time() - t) / len(Ws)
    n_bind = int(sum(((PairSettlement(inst, tou, stage2=False, rounding=False).settle(W).settled > c)).sum()
                     for W, c in zip(Ws[:1], caps[:1])))
    t = time.time(); [simulate_matching(W, inst) for W in Ws[:20]]; t_old = (time.time() - t) / 20
    print(f"A4 xlarge cost per candidate: point-13 no caps {t_free*1000:.1f} ms | "
          f"half the contracts capped ({n_bind} binding pairs) {t_cap*1000:.1f} ms | "
          f"current simulate_matching {t_old*1000:.1f} ms")

sys.exit(1 if failures else 0)
