#!/usr/bin/env python3
"""Generate HV box plot for the paper.

Shows hypervolume distributions (10 seeds) for each algorithm across
three problem scales, directly supporting statistical comparison claims.
Grouped layout: each base algorithm has a vanilla (blue) and SA (red) box.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np
import json
from pathlib import Path

# ── Journal-quality settings ─────────────────────────────────────────
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif',
    'axes.labelsize': 9,
    'axes.titlesize': 11,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
})

# ── Paired groups: (base_name, vanilla_key, sa_key) ──────────────────
GROUPS = [
    ('NSGA-II',  'NSGA-II',  'SA-NSGA-II'),
    ('MOEA/D',   'MOEA/D',   'SA-MOEA/D'),
    ('NSGA-III', 'NSGA-III', 'SA-NSGA-III'),
]

COLOR_VANILLA = '#2166AC'
COLOR_SA      = '#B2182B'

SCALES = [
    ('small',  '(a) Small (5 × 10)'),
    ('medium', '(b) Medium (10 × 20)'),
    ('large',  '(c) Large (20 × 50)'),
]

def load_hv_data(results_path):
    """Load HV values per algorithm per scale."""
    with open(results_path) as f:
        data = json.load(f)

    all_algos = [a for _, v, s in GROUPS for a in (v, s)]
    result = {}
    for scale_key, _ in SCALES:
        result[scale_key] = {}
        for algo in all_algos:
            result[scale_key][algo] = [r['hv'] for r in data[scale_key][algo]]
    return result

def draw_panel(ax, scale_data, title, show_ylabel=True):
    """Draw one grouped HV box plot panel."""

    scale_factor = 1e16
    box_width = 0.32
    group_gap = 1.2  # center-to-center between groups
    pair_gap = 0.38  # center-to-center within pair

    all_data = []
    all_positions = []
    all_colors = []

    for gi, (base_name, vanilla_key, sa_key) in enumerate(GROUPS):
        center = gi * group_gap
        pos_v = center - pair_gap / 2
        pos_s = center + pair_gap / 2

        all_positions.extend([pos_v, pos_s])
        all_data.append(np.array(scale_data[vanilla_key]) / scale_factor)
        all_data.append(np.array(scale_data[sa_key]) / scale_factor)
        all_colors.extend([COLOR_VANILLA, COLOR_SA])

    bp = ax.boxplot(all_data, positions=all_positions, widths=box_width,
                    patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=1.2),
                    whiskerprops=dict(linewidth=0.8, color='gray'),
                    capprops=dict(linewidth=0.8, color='gray'),
                    flierprops=dict(marker='o', markersize=3, alpha=0.5,
                                    markerfacecolor='gray', markeredgecolor='none'))

    for patch, color in zip(bp['boxes'], all_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.60)
        patch.set_edgecolor(color)
        patch.set_linewidth(0.8)

    # X-axis: group labels (horizontal, centered)
    group_centers = [gi * group_gap for gi in range(len(GROUPS))]
    ax.set_xticks(group_centers)
    ax.set_xticklabels([name for name, _, _ in GROUPS])

    ax.set_title(title, fontweight='bold', pad=8)

    if show_ylabel:
        ax.set_ylabel(r'Hypervolume ($\times 10^{16}$)')

    ax.grid(True, axis='y', alpha=0.25, linewidth=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Ensure x-axis has some padding
    ax.set_xlim(all_positions[0] - 0.5, all_positions[-1] + 0.5)

def main():
    results_path = (Path(__file__).resolve().parent.parent /
                    'results' / 'phase7' / '7a_main_comparison' / 'results.json')
    data = load_hv_data(results_path)

    fig, axes = plt.subplots(1, 3, figsize=(7.48, 3.0))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.18,
                        wspace=0.28)

    for i, (ax, (scale_key, title)) in enumerate(zip(axes, SCALES)):
        draw_panel(ax, data[scale_key], title, show_ylabel=(i == 0))

    # Legend: two color patches
    legend_handles = [
        mpatches.Patch(facecolor=COLOR_VANILLA, alpha=0.60,
                       edgecolor=COLOR_VANILLA, label='Vanilla'),
        mpatches.Patch(facecolor=COLOR_SA, alpha=0.60,
                       edgecolor=COLOR_SA, label='SA-enhanced'),
    ]
    fig.legend(handles=legend_handles,
               loc='upper right', fontsize=8, frameon=True,
               fancybox=False, edgecolor='gray', framealpha=0.9,
               bbox_to_anchor=(0.98, 0.88))

    out_dir = Path(__file__).resolve().parent.parent / 'paper' / 'figures'
    out_path = out_dir / 'pareto_fronts.pdf'
    fig.savefig(out_path, format='pdf', bbox_inches='tight',
                pad_inches=0.05, dpi=300)
    plt.close(fig)
    print(f'Saved: {out_path}')
    print(f'Size: {out_path.stat().st_size / 1024:.1f} KB')

if __name__ == '__main__':
    main()
