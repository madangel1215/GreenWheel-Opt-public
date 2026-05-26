#!/usr/bin/env python3
"""Generate ablation study figure for the paper.

Clean two-panel bar chart with error bars showing effect of
correction fraction on HV ratio and evaluation savings.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import json
from pathlib import Path

# ── Journal-quality settings ─────────────────────────────────────────
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif',
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8,
})

COLOR_NORMAL = '#2166AC'
COLOR_OPTIMAL = '#B2182B'
CF_KEYS = ['cf_0.3', 'cf_0.4', 'cf_0.5', 'cf_0.6', 'cf_0.7']
CF_LABELS = ['0.3', '0.4', '0.5', '0.6', '0.7']
OPTIMAL_IDX = 2

# Constants for converting stored eval_savings to true savings
# n_surr estimated from SA-NSGA-II main comparison (cf=0.5, n_true=11039, cs=0.621)
N_SURR_EST = 18082
N_VANILLA = 20000

def _true_savings(code_savings: float) -> float:
    """Convert stored eval_savings to standard metric: 1 - n_true / n_vanilla."""
    n_true = N_SURR_EST * (1.0 - code_savings) / code_savings
    return (1.0 - n_true / N_VANILLA) * 100

def load_data():
    p = (Path(__file__).resolve().parent.parent /
         'results' / 'phase7' / '7b_ablation' / 'results.json')
    raw = json.load(open(p))

    ratios, savings = [], []
    for key in CF_KEYS:
        runs = raw[key]
        ratios.append([r['hv_ratio'] for r in runs])
        savings.append([_true_savings(r['eval_savings']) for r in runs])
    return np.array(ratios), np.array(savings)

def main():
    ratios, savings = load_data()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.48, 2.6))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.20,
                        wspace=0.30)

    x = np.arange(len(CF_KEYS))
    bar_width = 0.50

    colors = [COLOR_OPTIMAL if i == OPTIMAL_IDX else COLOR_NORMAL
              for i in range(len(CF_KEYS))]
    alphas = [0.70 if i == OPTIMAL_IDX else 0.50
              for i in range(len(CF_KEYS))]

    # ── Panel (a): HV ratio ─────────────────────────────────────────
    means_r = np.mean(ratios, axis=1)
    stds_r = np.std(ratios, axis=1)

    bars_a = ax1.bar(x, means_r, width=bar_width,
                     color=colors, edgecolor=colors, linewidth=0.6)
    for bar, alpha in zip(bars_a, alphas):
        bar.set_alpha(alpha)

    # Error bars (separate call for clean styling)
    ax1.errorbar(x, means_r, yerr=stds_r, fmt='none',
                 ecolor='black', elinewidth=0.8, capsize=3, capthick=0.8)

    # Baseline
    ax1.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(CF_LABELS)
    ax1.set_xlabel(r'Correction fraction $c_f$')
    ax1.set_ylabel('HV ratio (SA / vanilla)')
    ax1.set_ylim(0, 4.5)
    ax1.set_title('(a) Hypervolume ratio', fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, axis='y', alpha=0.15, linewidth=0.4)

    # ── Panel (b): Evaluation savings ────────────────────────────────
    means_s = np.mean(savings, axis=1)

    bars_b = ax2.bar(x, means_s, width=bar_width,
                     color=colors, edgecolor=colors, linewidth=0.6)
    for bar, alpha in zip(bars_b, alphas):
        bar.set_alpha(alpha)

    ax2.set_xticks(x)
    ax2.set_xticklabels(CF_LABELS)
    ax2.set_xlabel(r'Correction fraction $c_f$')
    ax2.set_ylabel('Evaluation savings (%)')
    ax2.set_ylim(0, 75)
    ax2.set_title('(b) Evaluation savings', fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, axis='y', alpha=0.15, linewidth=0.4)

    out_dir = Path(__file__).resolve().parent.parent / 'paper' / 'figures'
    out_path = out_dir / 'ablation.pdf'
    fig.savefig(out_path, format='pdf', bbox_inches='tight',
                pad_inches=0.05, dpi=300)
    plt.close(fig)
    print(f'Saved: {out_path}')
    print(f'Size: {out_path.stat().st_size / 1024:.1f} KB')

if __name__ == '__main__':
    main()
