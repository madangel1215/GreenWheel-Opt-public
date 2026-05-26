#!/usr/bin/env python3
"""Generate publication-quality Critical Difference diagrams using matplotlib.

Produces a Demšar-style CD diagram for Friedman–Nemenyi analysis,
with colorblind-friendly palette matching the paper's color scheme.

Layout (top to bottom per panel):
  - Panel title (left-aligned)
  - CD bar (right-aligned, above axis)
  - Rank axis with tick numbers below
  - Clique bars (thick horizontal lines below axis)
  - Algorithm labels on left/right sides with connector lines
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Color palette (RdBu diverging, colorblind-safe) ──────────────────
ALGO_COLORS = {
    'NSGA-II':     '#2166AC',
    'MOEA/D':      '#67A9CF',
    'NSGA-III':    '#4393C3',
    'SA-NSGA-II':  '#B2182B',
    'SA-MOEA/D':   '#EF8A62',
    'SA-NSGA-III': '#D6604D',
}

CD = 2.456

# ── Data ──────────────────────────────────────────────────────────────
panels = [
    {
        'title': '(a) Small (5 × 10)',
        'ranks': {
            'NSGA-II': 2.60, 'MOEA/D': 2.60, 'SA-MOEA/D': 3.00,
            'SA-NSGA-II': 3.40, 'NSGA-III': 4.60, 'SA-NSGA-III': 4.80,
        },
        'cliques': [(2.60, 4.80, '#999999')],
    },
    {
        'title': '(b) Medium (10 × 20)',
        'ranks': {
            'MOEA/D': 1.60, 'SA-MOEA/D': 1.60, 'SA-NSGA-II': 2.80,
            'SA-NSGA-III': 4.00, 'NSGA-II': 5.10, 'NSGA-III': 5.90,
        },
        'cliques': [
            (1.60, 2.80, '#B2182B'),
            (2.80, 5.10, '#999999'),
            (4.00, 5.90, '#2166AC'),
        ],
    },
    {
        'title': '(c) Large (20 × 50)',
        'ranks': {
            'SA-NSGA-II': 1.50, 'MOEA/D': 2.70, 'SA-NSGA-III': 2.80,
            'SA-MOEA/D': 3.00, 'NSGA-II': 5.30, 'NSGA-III': 5.70,
        },
        'cliques': [
            (1.50, 3.00, '#B2182B'),
            (5.30, 5.70, '#2166AC'),
        ],
    },
]

def draw_cd_panel(ax, panel_data, k=6):
    """Draw one CD panel on the given axes.

    Coordinate system:
      x: rank 1..6 mapped to 0..10
      y: 0 = axis line, positive = above, negative = below
    """
    ranks = panel_data['ranks']
    cliques = panel_data['cliques']
    title = panel_data['title']

    def rank_to_x(r):
        """Map rank [1,k] to x coordinate [0,10]."""
        return (r - 1) * 10 / (k - 1)

    sorted_algos = sorted(ranks.items(), key=lambda x: x[1])

    # Canvas limits
    ax.set_xlim(-3.5, 13.5)
    ax.set_ylim(-2.5, 1.6)

    # ── Panel title (upper left) ──
    ax.text(-3.5, 1.4, title, fontsize=13, fontweight='bold',
            ha='left', va='top', fontfamily='serif')

    # ── CD bar (upper right) ──
    cd_x_len = rank_to_x(1 + CD) - rank_to_x(1)
    cd_right = 10
    cd_left = cd_right - cd_x_len
    y_cd = 1.1
    ax.annotate('', xy=(cd_right, y_cd), xytext=(cd_left, y_cd),
                arrowprops=dict(arrowstyle='<->', color='black',
                                lw=1.2, shrinkA=0, shrinkB=0))
    ax.text((cd_left + cd_right) / 2, y_cd + 0.12,
            f'CD = {CD:.2f}', ha='center', va='bottom',
            fontsize=11, fontfamily='serif')

    # ── Main axis ──
    ax.plot([0, 10], [0, 0], 'k-', linewidth=1.8, zorder=5)
    for r in range(1, k + 1):
        x = rank_to_x(r)
        ax.plot([x, x], [-0.12, 0.12], 'k-', linewidth=1.8, zorder=5)
        ax.text(x, 0.22, str(r), ha='center', va='bottom',
                fontsize=11.5, fontfamily='serif')

    # ── Clique bars (below axis) ──
    for j, (r_start, r_end, color) in enumerate(cliques):
        x_start = rank_to_x(r_start)
        x_end = rank_to_x(r_end)
        y_bar = -0.30 - j * 0.18
        ax.plot([x_start, x_end], [y_bar, y_bar],
                color=color, linewidth=5, solid_capstyle='round',
                alpha=0.65, zorder=4)

    # ── Algorithm labels ──
    n_half = len(sorted_algos) // 2
    left_algos = sorted_algos[:n_half]
    right_algos = sorted_algos[n_half:]

    # y positions for labels (below clique bars)
    n_cliques = len(cliques)
    label_y_start = -0.30 - n_cliques * 0.18 - 0.20

    # Left side (better ranks)
    for i, (name, rank) in enumerate(left_algos):
        color = ALGO_COLORS[name]
        x_rank = rank_to_x(rank)
        y_label = label_y_start - i * 0.28

        # Tick drop from axis
        ax.plot([x_rank, x_rank], [0, -0.15], color=color,
                linewidth=2.0, zorder=5)
        # Vertical line down to label height
        ax.plot([x_rank, x_rank], [-0.15, y_label], color=color,
                linewidth=0.7, zorder=3)
        # Horizontal line to left margin
        ax.plot([x_rank, -0.5], [y_label, y_label], color=color,
                linewidth=0.7, zorder=3)
        # Label text
        ax.text(-0.6, y_label, f'{name} ({rank:.2f})',
                ha='right', va='center', fontsize=11, color=color,
                fontfamily='serif', fontweight='bold')

    # Right side (worse ranks)
    for i, (name, rank) in enumerate(right_algos):
        color = ALGO_COLORS[name]
        x_rank = rank_to_x(rank)
        y_label = label_y_start - i * 0.28

        # Tick drop from axis
        ax.plot([x_rank, x_rank], [0, -0.15], color=color,
                linewidth=2.0, zorder=5)
        # Vertical line down
        ax.plot([x_rank, x_rank], [-0.15, y_label], color=color,
                linewidth=0.7, zorder=3)
        # Horizontal line to right margin
        ax.plot([x_rank, 10.5], [y_label, y_label], color=color,
                linewidth=0.7, zorder=3)
        # Label text
        ax.text(10.6, y_label, f'({rank:.2f}) {name}',
                ha='left', va='center', fontsize=11, color=color,
                fontfamily='serif', fontweight='bold')

    ax.set_axis_off()

def main():
    fig_width = 6.5
    fig_height = 7.2

    fig, axes = plt.subplots(3, 1, figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.01,
                        hspace=0.50)

    for ax, panel in zip(axes, panels):
        draw_cd_panel(ax, panel)

    out_dir = Path(__file__).resolve().parent.parent / 'paper' / 'figures'
    out_path = out_dir / 'cd_diagram.pdf'
    fig.savefig(out_path, format='pdf', bbox_inches='tight',
                pad_inches=0.1, dpi=300)
    plt.close(fig)
    print(f'Saved: {out_path}')
    print(f'Size: {out_path.stat().st_size / 1024:.1f} KB')

if __name__ == '__main__':
    main()
