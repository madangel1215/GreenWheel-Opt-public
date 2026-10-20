#!/usr/bin/env python3
"""Judge E2f and fix each cell's backtest objectives.

    python scripts/judge_e2f.py

A test case conflicts when run_e2f_objectives.py marked it so (front >= 10 points, both
range gaps >= 5%). A pair holds for (archetype, rho) when >= 4 of its 6 cases conflict.
Objectives: both pairs hold -> profit, worst, cvar; one -> that pair; none -> no backtest.
Missing cases are listed and the cell is left undecided, never judged on fewer runs.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.greenwheel.data import retailer2021 as rt  # noqa: E402

E2F = Path("results/retailer/e2f")
RHOS, MONTHS, SEEDS, NEED = (0.05, 0.1, 0.3), (1, 7), (1, 2, 3), 4
PAIR_OBJS = {"B": ["profit", "worst"], "C": ["profit", "cvar"]}

def main():
    cells, missing = {}, []
    for arch in rt.ARCHETYPES:
        for rho in RHOS:
            runs = []
            for m in MONTHS:
                for s in SEEDS:
                    f = E2F / f"{arch}_2021{m:02d}_s{s}_rho{rho}.json"
                    if f.exists():
                        runs.append(json.load(open(f)))
                    else:
                        missing.append(f.name)
            key = f"{arch}_rho{rho}"
            if len(runs) < len(MONTHS) * len(SEEDS):
                cells[key] = dict(arch=arch, rho=rho, decided=False, runs=len(runs))
                print(f"{key}: {len(runs)}/6 runs, undecided")
                continue
            hold = {p: sum(r["pairs"][p]["conflict"] for r in runs) for p in PAIR_OBJS}
            ok = [p for p in PAIR_OBJS if hold[p] >= NEED]
            objs = ["profit", "worst", "cvar"] if len(ok) == 2 else (PAIR_OBJS[ok[0]] if ok else [])
            cells[key] = dict(arch=arch, rho=rho, decided=True, conflicts=hold, pairs_hold=ok, objs=objs)
            print(f"{key}: B {hold['B']}/6, C {hold['C']}/6 -> "
                  f"{'backtest on ' + ' '.join(objs) if objs else 'no pair holds, no backtest'}")
    out = dict(rule=">= 4 of 6 cases conflict per (archetype, rho, pair)", cells=cells, missing=missing)
    json.dump(out, open(E2F / "judgement.json", "w"), indent=2)
    decided = [c for c in cells.values() if c["decided"]]
    print(f"decided {len(decided)}/9 cells, backtest in {sum(bool(c['objs']) for c in decided)}; "
          f"missing runs {len(missing)}")

if __name__ == "__main__":
    main()
