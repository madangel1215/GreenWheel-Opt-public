#!/usr/bin/env python3
"""Backtest: one 2021 month, one method, cost-accounted search, realized settlement.

    python scripts/run_e4_backtest.py --arch L --rho 0.1 --month 7 --seed 1 --method sa_gnn \
        --objs profit worst cvar --ckpt checkpoints/retailer_buyer --costs results/retailer/unit_costs.json

Methods: nsga2, sa_gnn, sa_random, sa_low; ev and oracle are NSGA-II on one
scenario, the scenario mean (the expected-value solution) or the realized month (hindsight),
picked against the baseline evaluated on that same scenario. The SA loop is the published SA-MOEA
pre-screening with its published settings (retention 0.5, 10 warm-up generations, full
evaluation every 20, 10% random exploration, discard), reusing its candidate selection;
only the screener differs between SA variants. Cost is counted, not timed:
c_exact per exact evaluation, plus c_sur or c_low per screened candidate, from the idle
unit costs. The run stops once the budget B = 10,000 x c_exact is spent.

Writes the front (objectives and W) after every generation's cost, the baseline and the
realized settlement of the solution picked by pick(), so the analysis can compute
cost-to-target and the NT$ effect without rerunning. Snapshots are also taken at 25/50/75/100% of B
(last generation with cumulative cost <= f B, as for HV), each picked and settled the same
way, so that SA-GNN and NSGA-II can be compared in NT$ at equal cost.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.run_e2_conflict import RowCap  # noqa: E402
from src.greenwheel.data import replay2021 as rp  # noqa: E402
from src.greenwheel.data import retailer2021 as rt  # noqa: E402
from src.greenwheel.optimization.samoea import _select_for_true_eval  # noqa: E402
from src.greenwheel.simulator.settlement import ScenarioBatchSettlement  # noqa: E402
from src.greenwheel.surrogate import retailer_graph as rg  # noqa: E402
from src.greenwheel.surrogate.model import WheelingGNN  # noqa: E402
from src.greenwheel.surrogate.training import load_trained_model, normalize_features  # noqa: E402

INDEX = {"profit": 0, "worst": 1, "cvar": 2}
POP, RETAIN, WARMUP, FULL_EVERY, RANDOM_RATIO = 100, 0.5, 10, 20, 0.1
BUDGET_EVALS = 10_000
FRACTIONS = (0.25, 0.5, 0.75, 1.0)

class Box(Problem):
    def __init__(self, mp, n_obj):
        self.m, self.n = mp.m, mp.n
        super().__init__(n_var=mp.m * mp.n, n_obj=n_obj, xl=0.0, xu=1.0)

def pick(F_ind, base_ind):
    """Highest mean profit among solutions whose mean worst fulfilment is not below
    the baseline's; if none, the highest mean worst fulfilment. F_ind is (N, 3), maximised."""
    ok = F_ind[:, 1] >= base_ind[1]
    return int(np.argmax(np.where(ok, F_ind[:, 0], -np.inf))) if ok.any() else int(np.argmax(F_ind[:, 1]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=rt.ARCHETYPES, required=True)
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--method", choices=("nsga2", "sa_gnn", "sa_random", "sa_low", "ev", "oracle"), required=True)
    ap.add_argument("--objs", nargs="+", choices=list(INDEX), required=True)
    ap.add_argument("--costs", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="results/retailer/e4")
    ap.add_argument("--budget-evals", type=int, default=BUDGET_EVALS, help="the backtest uses 10,000")
    # Buyer prices are the one assumed input. A redraw replaces them for both the search
    # and the settlement, as a retailer that knew different contract prices would.
    ap.add_argument("--price", choices=("drawn", "redraw1", "redraw2"), default="drawn")
    a = ap.parse_args()
    t0 = time.time()
    unit = json.load(open(a.costs))[a.arch]
    c_exact = unit["c_exact"]
    c_screen = {"sa_gnn": unit["c_sur"], "sa_low": unit["c_low"]}.get(a.method, 0.0)
    budget = a.budget_evals * c_exact
    cols = [INDEX[o] for o in a.objs]

    ret = rt.Retailer(a.arch, a.rho, a.seed)
    if a.price != "drawn":
        k = {"redraw1": 1, "redraw2": 2}[a.price]
        ret.price = np.random.default_rng(70_000 + 1_000 * k + a.seed).uniform(*rp.GREEN_PRICE, ret.n)
    mp = rt.RetailerMonth(ret, a.month)
    rng = np.random.default_rng(a.seed)

    def exact(X):
        return np.array([rg.indicators(mp.evaluate(x.reshape(mp.m, mp.n))) for x in X])

    if a.method in ("ev", "oracle"):
        if a.method == "ev":
            # Load windows start on different weekdays, so the averaged load has no TOU
            # calendar of its own; the most recent window's is the one a forecaster would use.
            last = max(range(len(mp.scen)), key=lambda i: mp.scen[i]["days"][0])
            R1, D1, tou1, sc = mp.R.mean(axis=0), mp.D.mean(axis=0), mp.tou[last], mp.scale
        else:
            R1, D1, tou1, sc = mp.real["R"], mp.real["D"], mp.real["tou"], 1.0
        one = ScenarioBatchSettlement(R1[None], D1[None], tou1[None])

        def exact(X):
            return np.array([rg.indicators(mp._metrics(
                one.settle_quantities(np.round(x.reshape(mp.m, mp.n), 4)) * sc, R1[None] * sc)) for x in X])

    if a.method == "sa_gnn":
        torch.set_num_threads(1)
        model, norm = load_trained_model(Path(a.ckpt) / "best.pt", device="cpu")
        inst = rg.mean_instance(mp)

        def screen(X):
            graphs = [rg.to_graph(inst, np.round(x.reshape(mp.m, mp.n), 4)) for x in X]
            normalize_features(graphs, norm)
            with torch.no_grad():
                return WheelingGNN.from_heterodata(Batch.from_data_list(graphs), model).numpy()
    elif a.method == "sa_low":
        # k scenarios drawn once per run; k from the idle unit costs.
        idx = np.sort(np.random.default_rng(a.seed + 500).choice(len(mp.scen), unit["k"], replace=False))

        def screen(X):
            return np.array([rg.indicators(mp.evaluate_subset(x.reshape(mp.m, mp.n), idx)) for x in X])
    else:
        def screen(X):
            return rng.random((len(X), 3))

    base = mp.baseline()
    X0 = mp.initial_population(POP, np.random.default_rng(a.seed + 900))
    algo = NSGA2(pop_size=POP, sampling=X0, crossover=SBX(prob=0.9, eta=20),
                 mutation=PM(eta=20), repair=RowCap())
    algo.setup(Box(mp, len(cols)), seed=a.seed, termination=("n_gen", 10 ** 6))
    spent, gen, history, n_exact, n_screen = 0.0, 0, [], 0, 0
    snap = {}
    n_select = int(POP * RETAIN)
    while spent < budget:
        pop = algo.ask()
        X = pop.get("X")
        full = a.method in ("nsga2", "ev", "oracle") or gen < WARMUP or gen % FULL_EVERY == 0 or len(X) <= n_select
        if full:
            Y = exact(X)
            spent += len(X) * c_exact
            n_exact += len(X)
            pop.set("F", -Y[:, cols])
            algo.tell(infills=pop)
        else:
            S = screen(X)
            spent += len(X) * c_screen
            n_screen += len(X)
            sel = _select_for_true_eval(-S[:, cols], n_select, RANDOM_RATIO, rng)
            Y = exact(X[sel])
            spent += len(sel) * c_exact
            n_exact += len(sel)
            inf = Population.new("X", X[sel])
            inf.set("F", -Y[:, cols])
            algo.tell(infills=inf)
        opt = algo.opt
        history.append(dict(gen=gen, cost=spent, F=(-opt.get("F")).tolist()))
        for f in FRACTIONS:
            if spent <= f * budget * (1 + 1e-9):
                snap[f] = (gen, spent, opt.get("X").copy())
        gen += 1

    Xf = algo.opt.get("X")
    Yf = exact(Xf)                                        # all three indicators of the final front
    base_ind = exact(base.ravel()[None])[0]
    j = pick(Yf, base_ind)
    W_pick = Xf[j].reshape(mp.m, mp.n)
    real_pick, real_base = mp.evaluate_real(W_pick), mp.evaluate_real(base)
    at, W_at = {}, {}
    for f, (g, c, Xs) in snap.items():
        Ys = exact(Xs)
        js = pick(Ys, base_ind)
        W_at[f] = Xs[js].reshape(mp.m, mp.n)
        r = mp.evaluate_real(W_at[f])
        at[str(f)] = dict(gen=g, cost=c, minutes=c / 60, ex_ante=Ys[js].tolist(),
                          profit=float(r["profit"]), worst=float(r["worst"]))
    out = dict(arch=a.arch, rho=a.rho, month=a.month, seed=a.seed, method=a.method, objs=a.objs,
               price=a.price, buyer_prices=ret.price.tolist(),
               unit=unit, budget=budget, generations=gen, n_exact=n_exact, n_screen=n_screen,
               history=history, final_front=Yf.tolist(), baseline_ex_ante=base_ind.tolist(), pick=j,
               realized=dict(pick=dict(profit=float(real_pick["profit"]), worst=float(real_pick["worst"])),
                             baseline=dict(profit=float(real_base["profit"]), worst=float(real_base["worst"]))),
               realized_at=at,
               wall_seconds=time.time() - t0)
    # Names are built whole: with_suffix would cut at the dot in "rho0.1".
    stem = Path(a.out) / f"{a.arch}_rho{a.rho}_m{a.month}_s{a.seed}_{a.method}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(f"{stem}.json", "w"))
    pickle.dump(dict(X=Xf, W_pick=W_pick, W_at=W_at), open(f"{stem}.pkl", "wb"))
    print(f"{a.arch} rho {a.rho} 2021-{a.month:02d} s{a.seed} {a.method}: gens {gen} exact {n_exact} "
          f"screened {n_screen} | realized profit {real_pick['profit']:,.0f} vs baseline {real_base['profit']:,.0f}, "
          f"worst {real_pick['worst']:.3f} vs {real_base['worst']:.3f} ({time.time() - t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
