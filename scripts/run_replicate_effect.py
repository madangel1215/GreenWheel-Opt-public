#!/usr/bin/env python3
"""Does replicating generators change the surrogate's local ranking?

The xlarge semi-real instances need more solar plants than dataset #37331 has, so
about one sixth of their generators are noisy copies of real plants. This script
isolates that replication at a fixed instance size: for each seed it builds two
semi-real instances of identical size, one whose solar nodes are all real and one in
which k of them are noisy copies made exactly as in the xlarge instances, and
measures the Table 10 quantities (within-set Spearman and screening false-negative
rate on a Latin-hypercube set and on a converged NSGA-II population) on both.

The size is the largest at which every solar node can still be real, and k is
chosen so that replicates make up the same share of generators as at xlarge.

    python -u scripts/run_replicate_effect.py --n-seeds 10 --output results/replicate_effect
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_screening_diagnosis import _analyze  # Table 10's measurement, reused unchanged

from src.greenwheel.config import WheelingConfig
from src.greenwheel.data.real import load_taipower, select_window
from src.greenwheel.data.synthetic import generate_semireal_instance
from src.greenwheel.surrogate.dataset import generate_lhs_samples, generate_nsga2_snapshots
from src.greenwheel.surrogate.inference import BatchSurrogate
from src.greenwheel.surrogate.training import load_trained_model

XLARGE_M = 50
N_CONSUMERS = 100


def usable_solar(cfg):
    """Solar plants passing the same >60 % valid-data filter select_plants applies."""
    data = load_taipower(cfg.taipower_json_path)
    start, end = select_window(data, n_days=cfg.n_days, month=cfg.taipower_month)
    w = data["solar"].loc[start:end]
    return int((w.notna().mean() > 0.6).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/surrogate_phase7_semireal/best.pt")
    p.add_argument("--n-seeds", dest="n_seeds", type=int, default=10)
    p.add_argument("--conditions", default="real,replicated")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="results/replicate_effect")
    p.add_argument("--quick", action="store_true", help="small size, 2 seeds, 20 generations")
    args = p.parse_args()

    base = WheelingConfig()
    g = usable_solar(base)
    frac = base.solar_fraction
    xl_solar = max(1, int(XLARGE_M * frac))
    share = max(0, xl_solar - g) / XLARGE_M          # replicate share at xlarge
    m = max(mm for mm in range(2, XLARGE_M + 1) if int(mm * frac) <= g)
    k = int(round(share * m))
    n_gen = 200
    if args.quick:
        m, k, n_gen, args.n_seeds = 10, 2, 20, 2
    print(f"usable solar plants {g}; xlarge replicates {xl_solar - g}/{XLARGE_M} ({share:.1%})", flush=True)
    print(f"test size {m}x{N_CONSUMERS}: {int(m*frac)} solar, k={k} replicated in the 'replicated' condition", flush=True)

    model, norm = load_trained_model(args.ckpt, device=args.device)
    surrogate = BatchSurrogate(model, norm, device=args.device)
    conds = {"real": 0, "replicated": k}
    conds = {c: conds[c] for c in args.conditions.split(",")}

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    report = {"_meta": {"usable_solar": g, "xlarge_replicate_share": share, "m": m,
                        "n": N_CONSUMERS, "k": k, "n_generations": n_gen,
                        "seeds": [95000 + i * 100 for i in range(args.n_seeds)],
                        "ckpt": args.ckpt}}
    for cond, kk in conds.items():
        report[cond] = []
        for i in range(args.n_seeds):
            sd = 95000 + i * 100                       # Table 10's seeds
            cfg = WheelingConfig(); cfg.n_generators, cfg.n_consumers = m, N_CONSUMERS
            cfg.force_solar_replicates = kk
            inst = generate_semireal_instance(cfg, seed=sd)
            surrogate.set_instance(inst)
            t0 = time.time()
            glob = _analyze(generate_lhs_samples(inst, 100, seed=sd), surrogate)
            loc = _analyze(generate_nsga2_snapshots(inst, pop_size=100, n_generations=n_gen,
                                                    snapshot_gens=[n_gen], seed=sd), surrogate)
            report[cond].append({"seed": sd, "global": glob, "local": loc})
            print(f"  {cond:10s} seed {i+1}/{args.n_seeds}: global rho={glob['mean_local_rho']:.3f} "
                  f"FN={glob['fn_rate']:.2f} | local rho={loc['mean_local_rho']:.3f} "
                  f"FN={loc['fn_rate']:.2f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(report, open(out / f"results_{cond}.json", "w"), indent=2, default=str)

    if set(conds) == {"real", "replicated"}:
        report["_paired"] = paired(report)
        print(json.dumps(report["_paired"], indent=2), flush=True)
    json.dump(report, open(out / "results.json", "w"), indent=2, default=str)


def paired(report):
    res = {}
    for where in ("global", "local"):
        for key in ("mean_local_rho", "fn_rate"):
            a = np.array([r[where][key] for r in report["real"]])
            b = np.array([r[where][key] for r in report["replicated"]])
            d = b - a
            res[f"{where}_{key}"] = {
                "real_mean": float(a.mean()), "replicated_mean": float(b.mean()),
                "diff_mean": float(d.mean()), "diff_min": float(d.min()), "diff_max": float(d.max()),
                "wilcoxon_p": float(wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0,
            }
    return res


if __name__ == "__main__":
    main()
