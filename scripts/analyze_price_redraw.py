#!/usr/bin/env python3
"""Realized benefit of NSGA-II's selection under redrawn buyer prices.

Reads the backtests run with run_e4_backtest.py --price redraw1 / redraw2 and the
backtests of the same cases under the original prices, and prints, per price set and
retailer, the median realized gain over the contract-proportional baseline and the
number of cases in which the selection earned more.

    python scripts/analyze_price_redraw.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = {"drawn": ROOT / "results/retailer/e4", "redraw1": ROOT / "results/price_redraw/redraw1",
       "redraw2": ROOT / "results/price_redraw/redraw2"}
out = {}
for price, d in SRC.items():
    for arch in "SML":
        gain, rel = [], []
        for month in (1, 7, 11):
            for seed in (1, 2, 3):
                r = json.load(open(d / f"{arch}_rho0.1_m{month}_s{seed}_nsga2.json"))["realized"]
                g = r["pick"]["profit"] - r["baseline"]["profit"]
                gain.append(g); rel.append(g / r["baseline"]["profit"])
        out[f"{price}_{arch}"] = dict(median_gain_ntd=float(np.median(gain)),
                                      median_gain_pct=float(np.median(rel) * 100),
                                      wins=int((np.array(gain) > 0).sum()), n=len(gain))
        o = out[f"{price}_{arch}"]
        print(f"{price:8s} {arch}: median gain {o['median_gain_ntd']/1e6:.2f} M NTD "
              f"({o['median_gain_pct']:+.2f}%), wins {o['wins']}/{o['n']}")
json.dump(out, open(ROOT / "results/price_redraw/redraw_summary.json", "w"), indent=2)
