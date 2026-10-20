#!/usr/bin/env python3
"""Acceptance checks R1 to R6 for src/greenwheel/data/realcase.py.

Reference values are recomputed from the raw files with separate code (csv module,
numpy.interp), not taken from realcase itself.
"""
import csv
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.greenwheel.data import realcase

failures = []

def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)

YEAR, MONTH, N, SEED = 2019, 7, 100, 1
inst = realcase.build_month(YEAR, MONTH, N, SEED)
meta = inst.meta
print(f"{YEAR}-{MONTH:02d}: {len(inst.generators)} plants, {len(inst.consumers)} users, kappa {meta['kappa']:.3f}")

# R1: monthly totals equal Taipower's reported kWh (read with the csv module).
reported = defaultdict(float)
with open(realcase.SOLAR_MONTHLY, encoding="utf-8-sig") as f:
    for row in csv.reader(f):
        if row[0].isdigit() and int(row[0]) == YEAR and int(row[1]) == MONTH:
            reported[("solar", row[2])] += float(row[4])
with open(realcase.WIND_MONTHLY, encoding="utf-8-sig") as f:
    for row in csv.reader(f):
        if row[0].isdigit() and int(row[0]) == YEAR and int(row[1]) == MONTH:
            v = row[7].strip()
            reported[("wind", row[4])] += float(v) if v not in ("", "-") else float("nan")
err = max(abs(g.generation.sum() - reported[(g.technology, name)]) / reported[(g.technology, name)]
          for g, name in zip(inst.generators, meta["plants"]))
check("R1 plant month totals == reported", err <= 1e-9, f"max rel err {err:.1e}")

# R2: each plant's normalised curve == national 15-min mean shape, recomputed with np.interp.
z_t, z = {"solar": [], "wind": []}, {"solar": [], "wind": []}
t0 = datetime(YEAR, MONTH, 1)
t1 = datetime(YEAR, MONTH + 1, 1) + timedelta(minutes=15)
with open(realcase.ZENODO) as f:
    rd = csv.DictReader(f)
    for row in rd:
        t = datetime.fromisoformat(row["datetime"])
        if t0 <= t <= t1:
            for k in ("solar", "wind"):
                z_t[k].append((t - t0).total_seconds())
                z[k].append(float(row[k]))
T = inst.n_intervals
pts = np.arange(T)[:, None] * 900 + np.array([0, 300, 600])[None, :]
worst = 0.0
for k in ("solar", "wind"):
    ref = np.interp(pts, z_t[k], z[k]).mean(axis=1)
    ref = ref / ref.sum()
    for g in inst.generators:
        if g.technology == k:
            worst = max(worst, np.abs(g.generation / g.generation.sum() - ref).max())
check("R2 plant shape == national shape", worst <= 1e-12, f"max abs diff {worst:.1e}")

# R3: user readings equal the raw EWELD file (end-of-interval stamps), before scaling.
ok3 = True
for path in meta["users"][:3]:
    raw = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            raw[datetime.fromisoformat(row["Time"])] = float(row["Value"])
    j = meta["users"].index(path)
    expect = np.array([raw[t0 + timedelta(minutes=15 * (i + 1))] for i in range(T)])
    ok3 &= np.allclose(inst.consumers[j].demand / meta["kappa"], expect, rtol=1e-12, atol=0)
check("R3 user readings == raw EWELD", ok3)

# R4
days = (datetime(YEAR, MONTH + 1, 1) - t0).days
G = np.stack([g.generation for g in inst.generators]); D = np.stack([c.demand for c in inst.consumers])
check("R4 shape/NaN/negatives", T == days * 96 and G.shape[1] == T and np.isfinite(G).all() and np.isfinite(D).all()
      and (G >= 0).all() and (D >= 0).all(), f"T={T}")

# R5
ratio = G.sum() / D.sum()
check("R5 supply/demand == rho", abs(ratio - meta["rho"]) / meta["rho"] <= 1e-9, f"{ratio:.12f}")

# R6
again = realcase.build_month(YEAR, MONTH, N, SEED)
check("R6 deterministic", all(np.array_equal(a.demand, b.demand) for a, b in zip(inst.consumers, again.consumers))
      and all(np.array_equal(a.generation, b.generation) for a, b in zip(inst.generators, again.generators)))

over = np.maximum(G - np.array([g.capacity_kw for g in inst.generators])[:, None] * 0.25, 0).sum() / G.sum()
print(f"info: share of generation above a quarter of capacity (cut by the settlement) {over:.4%}")
print(f"info: plants by tech {dict(zip(*np.unique([g.technology for g in inst.generators], return_counts=True)))}")
sys.exit(1 if failures else 0)
