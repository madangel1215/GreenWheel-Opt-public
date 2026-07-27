#!/usr/bin/env python3
"""Which generator-consumer pairs does the surrogate attend to?

Opens up the deployed model to ask whether its attention carries interpretable
structure, or whether the encoding rather than the message passing is doing the
work.

The surrogate scores an allocation by passing messages over the bipartite
generator-consumer graph with GATv2 attention. The attention coefficient on each
edge is therefore a direct statement of which pairs the model considers when
forming its judgement. This script extracts those coefficients and asks whether
they align with quantities a market participant would regard as important: the
allocated fraction itself, the price spread between a consumer's retail tariff
and a generator's contract price, and the ratio of a generator's output to a
consumer's demand.

Runs on CPU in seconds; no GPU is required.

    python scripts/run_attention_analysis.py --scale medium --n-seeds 5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.greenwheel.config import WheelingConfig
from src.greenwheel.data.synthetic import generate_instance
from src.greenwheel.optimization.nsga2 import run_nsga2
from src.greenwheel.surrogate.graph import instance_to_heterodata
from src.greenwheel.surrogate.training import load_trained_model

SIZE = {"small": (5, 10), "medium": (10, 20), "large": (20, 50)}
EDGE_FEATURES = ["allocation weight", "price spread", "generation/demand ratio"]
REL = ("gen", "supplies", "con")


def attention_per_edge(model, data):
    """Replicate the forward pass, capturing GATv2 attention on gen->con edges.

    Returns (n_edges,) attention averaged over heads and layers, plus the
    per-layer array for the concentration analysis.
    """
    x_gen = data["gen"].x
    x_con = data["con"].x
    ei = data[REL].edge_index
    ea = data[REL].edge_attr

    h_gen = model.gen_proj(x_gen)
    h_con = model.con_proj(x_con)

    per_layer = []
    for i, hetero in enumerate(model.convs):
        conv = hetero.convs[REL]
        out, (_, alpha) = conv(
            (h_gen, h_con), ei, edge_attr=ea, return_attention_weights=True
        )
        per_layer.append(alpha.mean(dim=1).detach().cpu().numpy())

        # Advance both node types so later layers see the correct state.
        rev = hetero.convs[("con", "demands", "gen")]
        out_gen = rev(
            (h_con, h_gen),
            data["con", "demands", "gen"].edge_index,
            edge_attr=data["con", "demands", "gen"].edge_attr,
        )
        h_con_new = model.norms_con[i](out + h_con)
        h_gen_new = model.norms_gen[i](out_gen + h_gen)
        h_con = torch.relu(h_con_new)
        h_gen = torch.relu(h_gen_new)

    per_layer = np.stack(per_layer)          # (n_layers, n_edges)
    return per_layer.mean(axis=0), per_layer


def gini(x):
    """Concentration of a non-negative vector; 0 = uniform, 1 = all on one edge."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="checkpoints/surrogate_phase7_synthetic/best.pt")
    p.add_argument("--config", default="configs/surrogate.yaml")
    p.add_argument("--scale", default="medium")
    p.add_argument("--n-seeds", dest="n_seeds", type=int, default=5)
    p.add_argument("--output", default="results/attention_analysis")
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model, _ = load_trained_model(args.model, device="cpu")
    model.eval()

    m, n = SIZE[args.scale]
    rows, corr = [], {f: [] for f in EDGE_FEATURES}
    ginis, top_share = [], []

    for i in range(args.n_seeds):
        seed = 95000 + i * 100
        cfg = WheelingConfig()
        cfg.n_generators, cfg.n_consumers = m, n
        inst = generate_instance(cfg, seed=seed)

        # Attend to a solution the optimizer would actually reach, not a random one.
        res = run_nsga2(inst, cfg, seed=seed, verbose=False)
        W = res.pareto_solutions[0].W

        data = instance_to_heterodata(inst, W)
        with torch.no_grad():
            att, per_layer = attention_per_edge(model, data)

        ea = data[REL].edge_attr.numpy()
        for k, name in enumerate(EDGE_FEATURES):
            rho = spearmanr(att, ea[:, k])[0]
            corr[name].append(float(rho))

        g = gini(att)
        order = np.argsort(att)[::-1]
        share = float(att[order[: max(1, len(att) // 10)]].sum() / att.sum())
        ginis.append(g)
        top_share.append(share)

        rows.append({"seed": seed, "n_edges": int(len(att)), "gini": g,
                     "top10pct_share": share,
                     "spearman": {f: corr[f][-1] for f in EDGE_FEATURES},
                     "per_layer_gini": [gini(l) for l in per_layer]})
        print(f"  seed {i+1}: edges={len(att):5d}  gini={g:.3f}  "
              f"top-10% share={share:.3f}  "
              + "  ".join(f"{f.split()[0]}={corr[f][-1]:+.3f}" for f in EDGE_FEATURES),
              flush=True)

    print(f"\n=== {args.scale} ({m}x{n}), {args.n_seeds} seeds ===")
    print(f"attention concentration (Gini): {np.mean(ginis):.3f} +- {np.std(ginis):.3f}")
    print(f"share on top 10% of edges:      {np.mean(top_share):.3f} +- {np.std(top_share):.3f}")
    print(f"uniform attention would give:   gini 0.000, top-10% share 0.100")
    for f in EDGE_FEATURES:
        v = corr[f]
        print(f"Spearman(attention, {f:24s}) = {np.mean(v):+.3f} +- {np.std(v):.3f}")

    json.dump({"scale": args.scale, "m": m, "n": n, "runs": rows},
              open(out / f"{args.scale}.json", "w"), indent=2, default=str)
    print(f"\nSaved -> {out}/{args.scale}.json")


if __name__ == "__main__":
    main()
