#!/usr/bin/env python3
"""Controlled degradation of the surrogate's ranking at fixed problem and scale.

The manuscript's hypothesis is that the benefit of pre-screening depends on how well the
surrogate orders the candidates of a single generation. The evidence for it compares two
instance families, which differ in more than surrogate quality. Here the instance, the
scale, the algorithm and the trained surrogate are held fixed and only the ranking
quality is changed: Gaussian noise is added to the surrogate's predictions, with a
standard deviation of lambda times the spread of the predictions in the batch being
screened (per objective), so the degradation stays proportionate as the population
converges. lambda = 0 is the published SA-NSGA-II; lambda = inf replaces the predictions
by noise, i.e. random screening.

Recorded per run: the final hypervolume (reference point shared by every arm on the
instance, vanilla NSGA-II included) and the in-run rank agreement between the noisy
predictions and the exact objectives on the full-evaluation generations (Spearman rho,
Kendall tau and the concordance sp), which is the "measured local rho" of the degraded
surrogate. Instances and seeds are those of the main comparison (Table 5).

Design and reading fixed before any run:
  levels lambda in {0, 0.5, 1, 2, 4, inf}; scales medium and large; 10 instances each.
  Primary test, per scale: Page's trend test on hypervolume across the six levels ordered
  from noisiest to cleanest, over the 10 instances (one-sided, p < 0.05).
  The relationship is called demonstrated at that scale only if that test is significant
  and the mean measured rho decreases monotonically with lambda. Whatever the outcome,
  the table of measured rho against hypervolume gain is reported for both scales.

    python -u scripts/run_noise_control.py --scale large --seed-index 0 --output results/noise_control
    python scripts/run_noise_control.py --analyze --output results/noise_control
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SIZE = {"small": (5, 10), "medium": (10, 20), "large": (20, 50)}
LEVELS = [0.0, 0.5, 1.0, 2.0, 4.0, float("inf")]


class NoisySurrogate:
    """Wraps a batch surrogate and perturbs what it returns; the model is untouched."""

    def __init__(self, base, lam: float, seed: int):
        self.base, self.lam = base, lam
        self.rng = np.random.default_rng(seed)

    def set_instance(self, instance):
        self.base.set_instance(instance)

    def evaluate_batch(self, W):
        out = [np.asarray(v, dtype=float) for v in self.base.evaluate_batch(W)]
        if self.lam == 0:
            return tuple(out)
        if np.isinf(self.lam):
            return tuple(self.rng.normal(size=v.shape) for v in out)
        return tuple(v + self.rng.normal(0.0, self.lam * (v.std() + 1e-12), size=v.shape) for v in out)


def run_seed(args):
    from src.greenwheel.config import WheelingConfig
    from src.greenwheel.data.synthetic import generate_instance
    from src.greenwheel.optimization.metrics import compute_hypervolume
    from src.greenwheel.optimization.nsga2 import run_nsga2
    from src.greenwheel.optimization.samoea import SAMOEAConfig, run_samoea
    from src.greenwheel.surrogate.inference import BatchSurrogate
    from src.greenwheel.surrogate.training import load_trained_model

    m, n = SIZE[args.scale]
    n_gen = 10 if args.quick else args.n_gen
    seed = 70042 + args.seed_index * 100                  # Table 5's instances
    cfg = WheelingConfig()
    cfg.n_generators, cfg.n_consumers = m, n
    cfg.n_generations, cfg.population_size = n_gen, 100
    inst = generate_instance(cfg, seed=seed)
    model, norm = load_trained_model(args.model, device="cpu")
    base = BatchSurrogate(model, norm, device="cpu")

    t0 = time.time()
    fronts = {"vanilla": run_nsga2(inst, cfg, seed=seed, verbose=False).pareto_front}
    arms = {}
    for lam in (LEVELS[:2] + LEVELS[-1:] if args.quick else LEVELS):
        sur = NoisySurrogate(base, lam, seed=10_000 + seed)
        sur.set_instance(inst)
        res = run_samoea(inst, sur, SAMOEAConfig(correction_fraction=0.5, n_generations=n_gen,
                                                 probe_enabled=True), seed=seed, verbose=False)
        # The manuscript's in-run measurement uses the periodic full-evaluation
        # generations only; the warm-up generations are probed too and kept separately.
        allp = [s for s in res.gen_stats if s.probe_rho is not None]
        probes = [s for s in allp if s.mode == "full_eval"]
        key = "inf" if np.isinf(lam) else str(lam)
        fronts[key] = res.pareto_front
        arms[key] = {"n_true_evals": res.n_true_evaluations, "n_probe_generations": len(probes),
                     "rho_by_objective": np.nanmean([s.probe_rho for s in probes], axis=0).tolist(),
                     "rho": float(np.nanmean([s.probe_rho for s in probes])),
                     "tau": float(np.nanmean([s.probe_tau for s in probes])),
                     "sp": float(np.nanmean([s.probe_sp for s in probes])),
                     "rho_incl_warmup": float(np.nanmean([s.probe_rho for s in allp])),
                     "probe": [{"generation": s.generation, "mode": s.mode, "rho": s.probe_rho,
                                "sp": s.probe_sp} for s in allp]}
        print(f"  {args.scale} seed {seed} lambda {key}: in-run rho {arms[key]['rho']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    valid = [f for f in fronts.values() if len(f) > 0]
    ref = np.vstack(valid).max(axis=0) * 1.1
    hv = {k: float(compute_hypervolume(f, ref)) for k, f in fronts.items()}
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    json.dump({"scale": args.scale, "seed": seed, "n_gen": n_gen, "hv": hv, "arms": arms,
               "seconds": time.time() - t0},
              open(out / f"{args.scale}_seed{args.seed_index}.json", "w"), indent=2)
    print(f"{args.scale} seed {seed} done: " + "  ".join(f"{k}={v/1e16:.1f}" for k, v in hv.items()), flush=True)


def analyze(args):
    from scipy.stats import page_trend_test, spearmanr, wilcoxon
    out = Path(args.output)
    keys = ["inf" if np.isinf(l) else str(l) for l in LEVELS]
    summary = {}
    for scale in ("medium", "large"):
        runs = [json.load(open(p)) for p in sorted(out.glob(f"{scale}_seed*.json"))]
        if len(runs) < 3:
            continue
        hv = np.array([[r["hv"][k] for k in keys] for r in runs])             # (instances, levels)
        van = np.array([r["hv"]["vanilla"] for r in runs])
        rho = np.array([[r["arms"][k]["rho"] for k in keys] for r in runs])
        sp = np.array([[r["arms"][k]["sp"] for k in keys] for r in runs])
        page = page_trend_test(hv[:, ::-1])       # columns noisiest -> cleanest; H1: increasing
        within = [spearmanr(rho[i], hv[i])[0] for i in range(len(runs))]
        s = {"n_instances": len(runs), "levels": keys,
             "mean_rho": rho.mean(0).tolist(), "mean_sp": sp.mean(0).tolist(),
             "rho_monotone_decreasing": bool(np.all(np.diff(rho.mean(0)) < 0)),
             "median_hv_ratio_to_vanilla": np.nanmedian(hv / np.where(van > 0, van, np.nan)[:, None], axis=0).tolist(),
             "n_vanilla_zero": int((van == 0).sum()),
             "mean_hv": hv.mean(0).tolist(), "mean_hv_vanilla": float(van.mean()),
             "mean_hv_ratio_of_means": (hv.mean(0) / van.mean()).tolist(),
             "wins_vs_vanilla": (hv > van[:, None]).sum(0).tolist(),
             "page_L": float(page.statistic), "page_p": float(page.pvalue),
             "within_instance_spearman_rho_hv": within,
             "within_instance_spearman_positive": int(np.sum(np.array(within) > 0)),
             "within_instance_wilcoxon_p": float(wilcoxon(within, alternative="greater").pvalue)}
        s["demonstrated"] = bool(s["page_p"] < 0.05 and s["rho_monotone_decreasing"])
        summary[scale] = s
        print(f"== {scale} ({len(runs)} instances)")
        print("lambda        " + "".join(f"{k:>8s}" for k in keys))
        print("in-run rho    " + "".join(f"{v:8.3f}" for v in s["mean_rho"]))
        print("in-run sp     " + "".join(f"{v:8.3f}" for v in s["mean_sp"]))
        print("HV/vanilla med" + "".join(f"{v:8.2f}" for v in s["median_hv_ratio_to_vanilla"]))
        print("HV/vanilla mean" + "".join(f"{v:7.2f} " for v in s["mean_hv_ratio_of_means"]))
        print("wins/vanilla  " + "".join(f"{v:8d}" for v in s["wins_vs_vanilla"]))
        print(f"Page trend p = {s['page_p']:.4g}; within-instance Spearman(rho, HV) positive on "
              f"{s['within_instance_spearman_positive']}/{len(runs)}, Wilcoxon p = "
              f"{s['within_instance_wilcoxon_p']:.4g}; demonstrated: {s['demonstrated']}")
    json.dump(summary, open(out / "summary.json", "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/surrogate_phase7_synthetic/best.pt")
    ap.add_argument("--scale", default="large")
    ap.add_argument("--seed-index", dest="seed_index", type=int, default=0)
    ap.add_argument("--n-gen", dest="n_gen", type=int, default=200)
    ap.add_argument("--output", default="results/noise_control")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    analyze(a) if a.analyze else run_seed(a)
