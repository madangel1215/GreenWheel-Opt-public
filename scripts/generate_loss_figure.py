#!/usr/bin/env python3
"""Generate the surrogate training/validation loss-curve figure.

Plots train vs validation loss and validation Spearman per objective across
epochs, marking the early-stopping (best-validation) epoch, as evidence that
overfitting is detected and controlled.

Usage:
    python scripts/generate_loss_figure.py \
        --history checkpoints/surrogate_phase7_synthetic/history.json \
        --out paper/figures/loss_curves.pdf
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 14, "axes.titlesize": 14,
    "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "figure.dpi": 150,
})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="checkpoints/surrogate_phase7_synthetic/history.json")
    ap.add_argument("--out", default="paper/figures/loss_curves.pdf")
    args = ap.parse_args()

    d = json.load(open(args.history))
    ep = np.arange(1, len(d["train_loss"]) + 1)
    best = int(np.argmin(d["val_loss"])) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(ep, d["train_loss"], label="Training loss", lw=2)
    ax1.plot(ep, d["val_loss"], label="Validation loss", lw=2)
    ax1.axvline(best, ls="--", color="gray", lw=1.5,
                label=f"Best epoch ({best})")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Normalized MSE loss")
    ax1.set_title("(a) Training vs validation loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    for obj in ["profit", "surplus", "shortfall"]:
        ax2.plot(ep, d[f"val_spearman_{obj}"], label=obj.capitalize(), lw=2)
    ax2.axvline(best, ls="--", color="gray", lw=1.5)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel(r"Validation Spearman $\rho$")
    ax2.set_title("(b) Validation ranking accuracy")
    ax2.legend(loc="lower right"); ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1.0)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"saved {args.out}; best epoch={best}, "
          f"final train={d['train_loss'][-1]:.4f} val={d['val_loss'][-1]:.4f}")

if __name__ == "__main__":
    main()
