#!/usr/bin/env python3
"""Per-generation convergence curves for vanilla vs surrogate-assisted MOEAs.

Records, per generation, the hypervolume (computed post hoc with one reference
point shared by every run on the same instance), the exact evaluations consumed
so far, and the wall-clock seconds so far. The last two make an anytime
comparison at equal cost possible: HV against exact evaluations, and HV against
wall-clock time.

Vanilla runs may be given more generations than the surrogate-assisted runs
(--vanilla-gens) so that the equal-time comparison has vanilla data beyond the
surrogate run's elapsed time. pymoo with a fixed seed makes the first N
generations of a longer run identical to an N-generation run, so the
equal-generation comparison is read off the same trajectory.

Published convergence figure, reproduced by the defaults:
    python -u scripts/run_convergence.py \
        --model checkpoints/surrogate_phase7_synthetic/best.pt \
        --scales medium,large --n-seeds 5 --device cuda \
        --output results/convergence

Anytime comparison at equal cost:
    python -u scripts/run_convergence.py \
        --model checkpoints/surrogate_phase7_synthetic/best.pt \
        --scales medium,large --n-seeds 10 --algos nsga2,nsga3 \
        --vanilla-gens 300 --device cuda --output results/anytime

Anytime comparison on the instances of the main comparison, all six variants:
    python -u scripts/run_convergence.py \
        --model checkpoints/surrogate_phase7_synthetic/best.pt \
        --scales medium,large --n-seeds 10 --algos nsga2,moead,nsga3 \
        --vanilla-gens 300 --seed-base 70042 --ref table5 --device cpu \
        --output results/anytime_table5

--ref table5 builds the reference point the way Table 5 did, from the six fronts
at --n-gen generations (a vanilla run's front at that generation equals the final
front of a run stopped there), so the generation-matched hypervolumes are directly
comparable with Table 5. NSGA-II and MOEA/D runs (vanilla and SA) are reproducible
from the seed; NSGA-III's are not, because pymoo 0.6.1.6 calls NSGA-III's niching
without the run's random state, so it draws from a fresh unseeded generator. The
NSGA-III pair, here and in Table 5, is therefore comparable only in distribution.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.greenwheel.config import WheelingConfig
from src.greenwheel.data.synthetic import generate_instance
from src.greenwheel.optimization.moead import run_moead
from src.greenwheel.optimization.nsga2 import run_nsga2
from src.greenwheel.optimization.nsga3 import run_nsga3
from src.greenwheel.optimization.samoea import SAMOEAConfig, run_samoea
from src.greenwheel.surrogate.inference import BatchSurrogate
from src.greenwheel.surrogate.training import load_trained_model
from src.greenwheel.optimization.metrics import compute_hypervolume

SIZE = {"small": (5, 10), "medium": (10, 20), "large": (20, 50), "xlarge": (50, 100)}
VANILLA = {"nsga2": ("NSGA-II", run_nsga2), "moead": ("MOEA/D", run_moead),
           "nsga3": ("NSGA-III", run_nsga3)}


def _ref_point(*fronts):
    valid = [np.asarray(f) for f in fronts if len(f) > 0]
    return np.max(np.vstack(valid), axis=0) * 1.1 if valid else np.ones(3) * 1e10


def _front_at(res, gen):
    return next(F for g, F in res.fronts_history if g == gen)


def _vanilla_record(seed, res, ref, n_gen):
    return {"seed": seed,
            "hv_at_n_gen": compute_hypervolume(_front_at(res, n_gen), ref),
            "hv_history": [{"generation": g, "hv": compute_hypervolume(F, ref),
                            "n_true_evals": ne, "elapsed_s": t}
                           for (g, F), (ne, t) in zip(res.fronts_history, res.cost_history)],
            "n_true_evals_total": res.n_evaluations,
            "elapsed_s_total": res.elapsed_seconds}


def _sa_record(seed, res, ref, n_gen):
    stats = {s.generation: s for s in res.gen_stats}
    cum, cum_by_gen = 0, {}
    for g in sorted(stats):
        cum += stats[g].n_true_evals
        cum_by_gen[g] = cum
    return {"seed": seed,
            "hv_at_n_gen": compute_hypervolume(res.pareto_front, ref),
            "hv_history": [{"generation": g, "hv": compute_hypervolume(F, ref),
                            "n_true_evals": cum_by_gen[g], "elapsed_s": stats[g].t_elapsed,
                            "mode": stats[g].mode}
                           for g, F in res.fronts_history],
            # Includes the final re-evaluation of the returned front.
            "n_true_evals_total": res.n_true_evaluations,
            "elapsed_s_total": res.elapsed_seconds}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--scales", default="medium,large")
    p.add_argument("--n-seeds", dest="n_seeds", type=int, default=5)
    p.add_argument("--n-gen", dest="n_gen", type=int, default=200)
    p.add_argument("--vanilla-gens", dest="vanilla_gens", type=int, default=None,
                   help="generations for vanilla runs (default: same as --n-gen)")
    p.add_argument("--algos", default="nsga2", help="comma list from: nsga2,moead,nsga3")
    p.add_argument("--seed-base", dest="seed_base", type=int, default=92000)
    p.add_argument("--ref", choices=("final", "table5"), default="final")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="results/convergence")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.scales, args.n_seeds, args.n_gen = "small", 2, 30
        if args.vanilla_gens is not None:
            args.vanilla_gens = 45
    van_gens = args.vanilla_gens or args.n_gen
    algos = args.algos.split(",")

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    model, norm = load_trained_model(args.model, device=args.device)
    surrogate = BatchSurrogate(model, norm, device=args.device)

    report = {"_meta": {"n_gen_sa": args.n_gen, "n_gen_vanilla": van_gens,
                        "algos": algos, "device": args.device,
                        "seed_base": args.seed_base,
                        "ref_point": ("1.1 x max over the final fronts of every run on the instance"
                                      if args.ref == "final" else
                                      "1.1 x max over each variant's front at n_gen_sa (Table 5)")}}
    for scale in args.scales.split(","):
        m, n = SIZE[scale]
        report[scale] = {}
        print(f"\n=== {scale} ({m}x{n}), {args.n_seeds} seeds, algos={algos} ===", flush=True)
        for i in range(args.n_seeds):
            seed = args.seed_base + i * 100
            inst = generate_instance(_cfg(m, n, args.n_gen), seed=seed)
            t0 = time.time()
            runs = {}
            for algo in algos:
                name, run_vanilla = VANILLA[algo]
                van = run_vanilla(inst, _cfg(m, n, van_gens), seed=seed, verbose=False,
                                  track_convergence=True)
                surrogate.set_instance(inst)
                sa = run_samoea(inst, surrogate,
                                SAMOEAConfig(base_algorithm=algo, correction_fraction=0.5,
                                             n_generations=args.n_gen),
                                seed=seed, verbose=False)
                runs[name] = van
                runs["SA-" + name] = sa
            if args.ref == "table5":
                ref = _ref_point(*(r.pareto_front if k.startswith("SA-") else _front_at(r, args.n_gen)
                                   for k, r in runs.items()))
            else:
                ref = _ref_point(*(r.pareto_front for r in runs.values()))
            for name, res in runs.items():
                rec = (_sa_record if name.startswith("SA-") else _vanilla_record)(seed, res, ref, args.n_gen)
                report[scale].setdefault(name, []).append(rec)
            print(f"  seed {i+1}/{args.n_seeds} done ({time.time()-t0:.0f}s)", flush=True)
            json.dump(report, open(out / "results.json", "w"), indent=2, default=str)

    json.dump(report, open(out / "results.json", "w"), indent=2, default=str)
    print(f"\nSaved -> {out}/results.json", flush=True)


def _cfg(m, n, n_gen):
    cfg = WheelingConfig()
    cfg.n_generators, cfg.n_consumers = m, n
    cfg.n_generations, cfg.population_size = n_gen, 100
    return cfg


if __name__ == "__main__":
    main()
