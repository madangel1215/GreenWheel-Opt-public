#!/usr/bin/env python3
"""Anytime comparison of the six variants on the Table 5 instances.

Mean hypervolume over the ten instances against exact evaluations consumed and against
wall-clock time, at medium and large scale. Each run is a step function of its cost; a
variant's mean curve stops where its shortest run stops. Surrogate-assisted runs end at
200 generations (dot), base algorithms are run to 300.

    python scripts/generate_anytime_figure.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 13, "axes.labelsize": 14, "axes.titlesize": 14,
                     "legend.fontsize": 11.5, "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "figure.dpi": 150})
COLOR = {"NSGA-II": "tab:blue", "MOEA/D": "tab:green", "NSGA-III": "tab:orange"}
d = json.load(open("results/anytime_table5/results.json"))
scales = [k for k in d if k != "_meta"]
fig, axes = plt.subplots(len(scales), 2, figsize=(12, 4.3 * len(scales)))
for r, scale in enumerate(scales):
    for c, (key, total, label) in enumerate([("n_true_evals", "n_true_evals_total", "Exact evaluations (thousands)"),
                                             ("elapsed_s", "elapsed_s_total", "Wall-clock time (s)")]):
        ax = axes[r, c]
        unit = 1e3 if c == 0 else 1.0
        for name, runs in d[scale].items():
            sa = name.startswith("SA-")
            curves = []
            for run in runs:
                x = [h[key] for h in run["hv_history"]]
                y = [h["hv"] for h in run["hv_history"]]
                if sa:   # the returned front is re-evaluated once at the end
                    x.append(max(run[total], x[-1])); y.append(run["hv_at_n_gen"])
                curves.append((np.array(x, float), np.array(y, float)))
            end = min(x[-1] for x, _ in curves)
            grid = np.linspace(0, end, 400)
            mean = np.mean([np.where(grid >= x[0], y[np.clip(np.searchsorted(x, grid, side="right") - 1, 0, len(y) - 1)], 0.0)
                            for x, y in curves], axis=0) / 1e16
            ax.plot(grid / unit, mean, color=COLOR[name.replace("SA-", "")], lw=2.2 if sa else 1.8,
                    ls="-" if sa else "--", label=name)
            if sa:
                ax.plot(grid[-1] / unit, mean[-1], "o", color=COLOR[name.replace("SA-", "")], ms=6)
        ax.set_xlabel(label); ax.set_ylabel(r"Hypervolume ($\times 10^{16}$)")
        ax.set_title(f"({'abcd'[2 * r + c]}) {scale} scale, {'evaluations' if c == 0 else 'time'}")
        ax.grid(alpha=0.3)
handles, labels = axes[0, 0].get_legend_handles_labels()
order = [labels.index(k) for k in ("NSGA-II", "SA-NSGA-II", "MOEA/D", "SA-MOEA/D", "NSGA-III", "SA-NSGA-III")]
fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=6,
           frameon=False, bbox_to_anchor=(0.5, 1.03), handlelength=2.6)
fig.tight_layout()
fig.savefig("paper/figures/anytime.pdf", bbox_inches="tight")
print("saved paper/figures/anytime.pdf")
