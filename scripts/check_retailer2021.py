#!/usr/bin/env python3
"""Checks for the real-scenario retailers (retailer2021.py).

    python scripts/check_retailer2021.py

R1 without rounding, group-level batched settlement equals site-by-site settlement
   (ConsumerContractSettlement, each site taking its station group's row) for dense,
   sparse, partial and zero W: settlement is exact at station-group level. The rounding difference is printed.
R2 real supply: each site's 2021 output (12 station months) = its T-REC 2025 volume.
R3 scaled demand: each buyer's load over the replayed months = contract share x supply / rho,
   and the shares sum to 1.
R4 W = 0 gives zero profit and zero fulfilment.
R5 lead time: no load scenario of month t uses data from month t-1 or later.
R6 baseline: entries are whole 0.01% steps, every row sells exactly 100%, every
   buyer gets a positive expected volume (cell-by-cell rounding gives the small ones 0),
   and each buyer's expected volume is its contract share to within the last
   carried step; the largest buyer, which takes the row residues, to within their sum.
   The initial population starts with it.
Also prints the build size and the time of one batched evaluation per archetype.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.check_replay2021 import random_W  # noqa: E402
from src.greenwheel.data import realcase as rc  # noqa: E402
from src.greenwheel.data import replay2021 as rp  # noqa: E402
from src.greenwheel.data import retailer2021 as rt  # noqa: E402

def main():
    bad = 0
    cu = rp.EweldCustomers()
    rng = np.random.default_rng(11)
    for arch, rho, seed, t in [("S", 0.1, 1, 7), ("M", 0.3, 2, 1), ("L", 0.05, 3, 10)]:
        ret = rt.Retailer(arch, rho, seed, customers=cu)
        mp = rt.RetailerMonth(ret, t)
        # R2: from the raw T-REC table and the raw monthly records, not from the module's
        # yearly totals.
        tsv = pd.read_csv(rt.TREC, sep="\t")
        trec = tsv[(tsv.archetype == arch) & (tsv.kind == "site") & tsv.tech.isin(list(rt.TECH))].mwh.to_numpy() * 1000
        pm = rc._plant_months()
        months = pm[pm.year == rp.YEAR].pivot_table(index="plant", columns="month", values="kwh", aggfunc="sum")
        out = ret.site_weight * np.array([months.loc[p].sum() for p in ret.site_station])
        real_month = mp.real["R"].sum()
        sites_month = sum(ret.site_weight[i] * rp._gen(rp.YEAR, t)[p]["kwh"] for i, p in enumerate(ret.site_station))
        ok = np.allclose(out, trec, rtol=1e-9) and np.isclose(real_month, sites_month, rtol=1e-9)
        bad += not ok
        print(("PASS" if ok else "FAIL"), f"R2 {arch} sites {len(out)}  supply {out.sum() / 1e6:,.1f} GWh/yr "
              f"(T-REC {trec.sum() / 1e6:,.1f})  2021-{t:02d} {real_month / 1e6:,.2f} GWh")
        # R3
        got = np.array([ret.customers.replay_kwh(u) for u in ret.customers.ids]) * ret.customers.scale
        ok = (np.allclose(got, ret.contract_share * ret.supply_replay / rho, rtol=1e-9)
              and np.isclose(ret.contract_share.sum(), 1))
        bad += not ok
        print(("PASS" if ok else "FAIL"), f"R3 {arch} demand over replay {got.sum() / 1e6:,.1f} GWh, "
              f"supply/demand {ret.supply_replay / got.sum():.4f}, largest share {ret.contract_share.max():.3f}")
        # R5
        first = rp.shift_month(rp.YEAR, t, -1)
        ok = all(src < first for c in mp.scen for src in c["load_src"])
        bad += not ok
        print(("PASS" if ok else "FAIL"), f"R5 {arch} latest load month used for {t}: "
              f"{max(src for c in mp.scen for src in c['load_src'])}")
        # R1: without rounding the two must agree to float precision; with
        # Taipower's per-pair integer rounding they differ, and the size is reported.
        scens = [0, len(mp.scen) // 2, len(mp.scen) - 1]
        for kind in ("dense", "sparse", "partial", "zero"):
            W = random_W(rng, mp.m, mp.n, kind)
            a, b = mp.evaluate(W, rounding=False), mp.evaluate_sites(W, scens, rounding=False)
            ok = all(np.allclose(a[k][scens], b[k], rtol=1e-9, atol=1e-6) for k in ("profit", "worst", "green"))
            bad += not ok
            ar, br = mp.evaluate(W), mp.evaluate_sites(W, scens)
            rel = np.abs(ar["profit"][scens] - br["profit"]) / np.maximum(1.0, np.abs(br["profit"]))
            print(("PASS" if ok else "FAIL"),
                  f"R1 {arch} {kind}: unrounded max |profit diff| {np.abs(a['profit'][scens] - b['profit']).max():.2e}; "
                  f"with rounding, group vs site profit differs by {rel.max():.2%}")
        # R6: expected volumes from the scenario means, recomputed here, not taken from baseline().
        B = mp.baseline()
        g = mp.R.mean(axis=0).sum(axis=1)
        vol, want = B.T @ g, ret.contract_share * g.sum()
        big = int(np.argmax(ret.contract_share))
        others = np.arange(mp.n) != big
        resid = np.abs(B.sum(axis=1) - 1).max()
        ok = (np.allclose(B * 1e4, np.round(B * 1e4), atol=1e-6) and resid < 1e-9 and (B >= 0).all()
              and (vol > 0).all() and (np.abs(vol - want)[others] <= 0.5e-4 * g.max() * (1 + 1e-9)).all()
              and np.array_equal(mp.initial_population(5, np.random.default_rng(0))[0], B.ravel()))
        bad += not ok
        rel = np.abs(vol - want) / want
        print(("PASS" if ok else "FAIL"), f"R6 {arch} baseline: smallest share {ret.contract_share.min():.2e}, "
              f"max relative volume error {rel[others].max():.2%} (largest buyer {rel[big]:.4%}), "
              f"worst fulfilment {mp.evaluate(B)['worst'].mean():.3f} "
              f"(cell rounding {mp.evaluate(np.tile(ret.contract_share, (mp.m, 1)))['worst'].mean():.3f})")
        # R4
        z = mp.evaluate(np.zeros((mp.m, mp.n)))
        ok = np.all(z["profit"] == 0) and np.all(z["worst"] == 0)
        bad += not ok
        print(("PASS" if ok else "FAIL"), f"R4 {arch}")
        W = random_W(rng, mp.m, mp.n, "dense")
        t0 = time.perf_counter()
        for _ in range(5):
            mp.evaluate(W)
        print(f"     {arch}: sites {len(ret.site_tech)} stations {mp.m} customers {mp.n} scenarios {len(mp.scen)} "
              f"evaluation {(time.perf_counter() - t0) / 5 * 1000:.0f} ms")
    print("ALL PASS" if not bad else f"{bad} FAIL")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
