#!/usr/bin/env python3
"""Generate publication-quality figures for the GreenWheel-Opt paper.

Produces:
  1. paper/figures/ablation.pdf       - correction fraction ablation (bar+line)
  2. paper/figures/critical_difference_diagram.pdf - Friedman-Nemenyi CD diagrams

Usage:
    python scripts/generate_paper_figures.py
"""

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from paper_figure_style import (
    ALGO_COLORS,
    ABLATION_BAR_COLOR,
    ABLATION_LINE_COLOR,
    ABLATION_OPTIMAL_COLOR,
    setup_style,
    save_pdf,
    SINGLE_COL_WIDTH_IN,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Figure 1: Ablation plot ──────────────────────────────────────────────────

def generate_ablation_figure():
    """Dual-axis bar+line chart for correction fraction ablation."""
    print("Generating ablation figure...")

    with open(ROOT / "results/phase7/7b_ablation/summary.json") as f:
        data = json.load(f)

    cfs = [d["cf"] for d in data]
    hv_ratios = [d["hv_ratio"] for d in data]
    savings = [d["eval_savings"] * 100 for d in data]

    fig, ax1 = plt.subplots(figsize=(SINGLE_COL_WIDTH_IN, SINGLE_COL_WIDTH_IN * 0.72))

    # Bars: HV ratio
    x = np.arange(len(cfs))
    bar_width = 0.55
    bars = ax1.bar(x, hv_ratios, bar_width, color=ABLATION_BAR_COLOR, alpha=0.85,
                   zorder=3, edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Correction fraction ($c_f$)")
    ax1.set_ylabel("HV ratio (vs. vanilla)", color=ABLATION_BAR_COLOR)
    ax1.tick_params(axis="y", colors=ABLATION_BAR_COLOR)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{c:.1f}" for c in cfs])
    ax1.set_ylim(0, max(hv_ratios) * 1.25)

    # Mark optimal cf=0.5
    opt_idx = cfs.index(0.5)
    ax1.plot(opt_idx, hv_ratios[opt_idx] + 0.08, marker="*", color=ABLATION_OPTIMAL_COLOR,
             markersize=12, zorder=5, markeredgecolor="black", markeredgewidth=0.4)

    # Line: Eval savings (second y-axis)
    ax2 = ax1.twinx()
    ax2.plot(x, savings, color=ABLATION_LINE_COLOR, marker="o", linewidth=1.5,
             markersize=5, zorder=4, markeredgecolor="white", markeredgewidth=0.5)
    ax2.set_ylabel("Evaluation savings (%)", color=ABLATION_LINE_COLOR)
    ax2.tick_params(axis="y", colors=ABLATION_LINE_COLOR)
    ax2.set_ylim(min(savings) - 5, max(savings) + 5)

    # Light grid on primary axis
    ax1.yaxis.grid(True, alpha=0.2, linewidth=0.4)
    ax1.set_axisbelow(True)

    fig.tight_layout()
    save_pdf(fig, FIG_DIR / "ablation.pdf")

# ── Figure 2: Critical difference diagrams (Demšar-style) ────────────────────

def _draw_cd_diagram(ax, mean_ranks, cd, nemenyi_p, scale_label):
    """Draw a Demšar-style CD diagram.

    Classic layout: rank axis at center, best-half names on the LEFT side,
    worst-half names on the RIGHT side. Thick bars connect non-significant
    groups. Uses the largest clique boundary to decide the left/right split.
    """
    n_algo = len(mean_ranks)
    sorted_algos = sorted(mean_ranks, key=lambda a: mean_ranks[a])
    ranks = [mean_ranks[a] for a in sorted_algos]

    # Find cliques first to determine optimal left/right split
    cliques = _find_nemenyi_cliques(sorted_algos, ranks, cd, nemenyi_p)

    # Determine left/right split based on clique structure.
    # Look for the largest gap in ranks - this usually separates "good" from
    # "bad" algorithms and gives the most natural visual split.
    n_left = n_algo // 2  # default: even split
    if n_algo > 2:
        gaps = [(ranks[i + 1] - ranks[i], i + 1) for i in range(n_algo - 1)]
        max_gap_val, max_gap_idx = max(gaps, key=lambda g: g[0])
        # Only use gap-based split if the gap is meaningful (> 0.5 rank units)
        # and doesn't put too many on one side
        if max_gap_val > 0.5 and 2 <= max_gap_idx <= n_algo - 2:
            n_left = max_gap_idx

    # Layout constants
    rank_min, rank_max = 0.5, n_algo + 0.5
    axis_y = 0.0
    name_x_left = rank_min - 0.15
    name_x_right = rank_max + 0.15

    ax.set_xlim(rank_min - 2.0, rank_max + 2.0)
    n_max_side = max(n_left, n_algo - n_left)
    y_spacing = 0.35
    y_bottom = -(0.45 + (n_max_side - 1) * y_spacing) - 0.3
    ax.set_ylim(y_bottom, 1.8)

    # ── Rank axis ──
    ax.hlines(axis_y, rank_min, rank_max, color="black", linewidth=0.7)
    for r in range(1, n_algo + 1):
        ax.vlines(r, axis_y - 0.07, axis_y + 0.07, color="black", linewidth=0.7)
        ax.text(r, axis_y + 0.15, str(r), ha="center", va="bottom", fontsize=7)

    # ── Scale label (top-left) ──
    ax.text(rank_min - 1.8, 1.5, scale_label, fontsize=8, fontweight="bold",
            va="top", ha="left")

    # ── CD bar indicator (top-right) ──
    cd_x_start = rank_max - cd
    cd_x_end = rank_max
    cd_y = 1.3
    ax.annotate("", xy=(cd_x_start, cd_y), xytext=(cd_x_end, cd_y),
                arrowprops=dict(arrowstyle="<->", color="black", lw=0.8))
    ax.text((cd_x_start + cd_x_end) / 2, cd_y + 0.12,
            f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=6)

    # ── Place algorithm names ──
    for i, algo in enumerate(sorted_algos):
        r = ranks[i]
        color = ALGO_COLORS.get(algo, "black")

        if i < n_left:
            name_y = -(0.45 + i * y_spacing)
            ax.text(name_x_left, name_y, algo, ha="right", va="center",
                    fontsize=6.5, color=color, fontweight="bold")
            ax.hlines(name_y, name_x_left + 0.05, r, color=color, linewidth=0.6)
            ax.vlines(r, name_y, axis_y, color=color, linewidth=0.6)
        else:
            j = i - n_left
            name_y = -(0.45 + j * y_spacing)
            ax.text(name_x_right, name_y, algo, ha="left", va="center",
                    fontsize=6.5, color=color, fontweight="bold")
            ax.hlines(name_y, r, name_x_right - 0.05, color=color, linewidth=0.6)
            ax.vlines(r, name_y, axis_y, color=color, linewidth=0.6)

    # ── Non-significant clique bars (only cliques with ≥3 members) ──
    # Filter out trivial pairs that add visual noise
    significant_cliques = [c for c in cliques if (c[1] - c[0]) >= 2]
    # Fall back to all cliques if no large ones exist
    if not significant_cliques:
        significant_cliques = cliques

    bar_y_base = 0.7
    bar_y_step = 0.25
    for ci, (c_start, c_end) in enumerate(significant_cliques):
        r_left = ranks[c_start]
        r_right = ranks[c_end]
        y = bar_y_base + ci * bar_y_step
        ax.plot([r_left, r_right], [y, y], color="#333333", linewidth=2.8,
                solid_capstyle="round", zorder=5)

    ax.axis("off")

def _find_nemenyi_cliques(sorted_algos, ranks, cd, nemenyi_p):
    """Find maximal cliques using actual Nemenyi p-values.

    A clique is a maximal set of algorithms where ALL pairs are
    not significantly different (p > 0.05).
    """
    n = len(sorted_algos)
    alpha = 0.05

    # Build adjacency: connected if NOT significantly different
    connected = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            a_i = sorted_algos[i]
            a_j = sorted_algos[j]
            if i == j:
                connected[i][j] = True
            else:
                p = nemenyi_p.get(a_i, {}).get(a_j, 1.0)
                connected[i][j] = p > alpha

    # Greedy maximal cliques: for each starting point, extend rightward
    # as long as ALL pairs in the group remain connected
    cliques = []
    for i in range(n):
        j = i + 1
        while j < n:
            # Check if adding j keeps all pairs connected
            all_ok = all(connected[k][j] for k in range(i, j))
            if all_ok:
                j += 1
            else:
                break
        j -= 1  # last valid index
        if j > i:
            clique = (i, j)
            # Only keep if not a strict subset of existing
            is_subset = any(s <= i and j <= e for s, e in cliques)
            if not is_subset:
                cliques.append(clique)

    return cliques

def generate_cd_figure():
    """Three-panel CD diagram (small, medium, large) stacked vertically."""
    print("Generating critical difference diagram...")

    with open(ROOT / "results/phase7/7f_statistics/friedman_nemenyi.json") as f:
        data = json.load(f)

    scales = ["small", "medium", "large"]
    scale_labels = [
        r"Small ($5 \times 10$)",
        r"Medium ($10 \times 20$)",
        r"Large ($20 \times 50$)",
    ]

    fig, axes = plt.subplots(3, 1, figsize=(SINGLE_COL_WIDTH_IN, SINGLE_COL_WIDTH_IN * 1.65))

    for ax, scale, label in zip(axes, scales, scale_labels):
        info = data[scale]
        _draw_cd_diagram(
            ax,
            mean_ranks=info["mean_ranks"],
            cd=info["critical_difference"],
            nemenyi_p=info["nemenyi_p_values"],
            scale_label=label,
        )

    fig.tight_layout(h_pad=0.8)
    save_pdf(fig, FIG_DIR / "critical_difference_diagram.pdf")

# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_style()
    generate_ablation_figure()
    generate_cd_figure()
    print("Done. All figures saved to paper/figures/")
