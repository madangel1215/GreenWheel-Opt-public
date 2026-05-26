"""Shared style module for GreenWheel-Opt paper figures.

Defines a unified color palette (colorblind-friendly, print-safe) and
matplotlib rcParams targeting Applied Energy journal requirements.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- Algorithm color palette (colorblind-friendly, print-safe) ---
# Vanilla algorithms: cool/blue tones
# SA-enhanced algorithms: warm/red tones
ALGO_COLORS = {
    "NSGA-II":     "#2166AC",  # strong blue
    "MOEA/D":      "#67A9CF",  # medium blue
    "NSGA-III":    "#D1E5F0",  # light blue
    "SA-NSGA-II":  "#B2182B",  # strong red
    "SA-MOEA/D":   "#EF8A62",  # coral
    "SA-NSGA-III": "#FDDBC7",  # light coral
}

# Ablation plot colors
ABLATION_BAR_COLOR = "#2166AC"   # blue bars for HV ratio
ABLATION_LINE_COLOR = "#B2182B"  # red line for eval savings
ABLATION_OPTIMAL_COLOR = "#FFD700"  # gold star for optimal

# --- Journal figure dimensions (Applied Energy / Elsevier) ---
SINGLE_COL_WIDTH_IN = 3.54   # 90 mm
DOUBLE_COL_WIDTH_IN = 7.48   # 190 mm
ASPECT_RATIO = 0.75          # default height/width

def setup_style():
    """Configure matplotlib rcParams for journal-quality figures."""
    mpl.rcParams.update({
        # Font
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        # Lines
        "lines.linewidth": 1.2,
        "lines.markersize": 5,
        # Axes
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Ticks
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 1000,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # PDF backend
        "pdf.fonttype": 42,  # TrueType fonts (not Type 3)
        "ps.fonttype": 42,
    })

def single_col_fig(aspect=ASPECT_RATIO):
    """Create a single-column figure (90 mm wide)."""
    w = SINGLE_COL_WIDTH_IN
    return plt.figure(figsize=(w, w * aspect))

def double_col_fig(aspect=ASPECT_RATIO):
    """Create a double-column figure (190 mm wide)."""
    w = DOUBLE_COL_WIDTH_IN
    return plt.figure(figsize=(w, w * aspect))

def save_pdf(fig, path, **kwargs):
    """Save figure as vector PDF with embedded fonts."""
    fig.savefig(path, format="pdf", backend="pdf", **kwargs)
    plt.close(fig)
    print(f"  Saved: {path}")
