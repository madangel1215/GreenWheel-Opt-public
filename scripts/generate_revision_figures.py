#!/usr/bin/env python3
"""Generate study figures: convergence curves and alpha sensitivity."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 13, "axes.labelsize": 14, "axes.titlesize": 14,
                     "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "figure.dpi": 150})

# ---- Convergence curves ----
d = json.load(open("results/convergence/results.json"))
fig, axes = plt.subplots(1, len(d), figsize=(5.5 * len(d), 4.2))
if len(d) == 1:
    axes = [axes]
for ax, scale in zip(axes, d):
    for meth, color in [("NSGA-II", "tab:blue"), ("SA-NSGA-II", "tab:red")]:
        curves = [[g["hv"] / 1e16 for g in run["hv_history"]] for run in d[scale][meth] if run["hv_history"]]
        L = min(len(c) for c in curves)
        arr = np.array([c[:L] for c in curves])
        gens = np.arange(1, L + 1)
        mean, std = arr.mean(0), arr.std(0)
        ax.plot(gens, mean, color=color, lw=2, label=meth)
        ax.fill_between(gens, mean - std, mean + std, color=color, alpha=0.2)
    ax.set_xlabel("Generation"); ax.set_ylabel(r"Hypervolume ($\times 10^{16}$)")
    ax.set_title(f"({'ab'[list(d).index(scale)]}) {scale} scale")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("paper/figures/convergence.pdf", bbox_inches="tight")
print("saved paper/figures/convergence.pdf")

# ---- Alpha sensitivity ----
d = json.load(open("results/alpha_sensitivity/results.json"))
alphas = sorted(d, key=float)
van = [np.median([r["hv"] for r in d[a]["NSGA-II"]]) / 1e16 for a in alphas]
sa = [np.median([r["hv"] for r in d[a]["SA-NSGA-II"]]) / 1e16 for a in alphas]
fig, ax = plt.subplots(figsize=(6, 4.2))
x = [float(a) for a in alphas]
ax.plot(x, van, "o-", color="tab:blue", lw=2, label="NSGA-II")
ax.plot(x, sa, "s-", color="tab:red", lw=2, label="SA-NSGA-II")
ax.axvline(0.15, ls="--", color="gray", lw=1.5, label=r"Default $\alpha=0.15$")
ax.set_xlabel(r"Fairness tolerance $\alpha$"); ax.set_ylabel(r"Median hypervolume ($\times 10^{16}$)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("paper/figures/alpha_sensitivity.pdf", bbox_inches="tight")
print("saved paper/figures/alpha_sensitivity.pdf")
