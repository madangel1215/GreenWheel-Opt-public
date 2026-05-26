"""
Visualization module for experiment results.

Generates publication-quality figures for:
1. Pareto front scatter plot (NSGA-II vs baselines)
2. Hypervolume convergence curve
3. Box plots comparing methods across seeds
4. Performance metrics table
5. Convergence comparison (mean ± std band)
6. Scalability (n_var vs time, log-log)
7. Ablation (correction_fraction vs HV + savings)
8. HV boxplot (multi-seed distribution)
9. Sensitivity analysis (parameter sweep line plots)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# Publication-quality defaults
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.figsize": (8, 6),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Color scheme for methods
METHOD_COLORS = {
    "NSGA-II": "#2196F3",      # blue
    "Random": "#9E9E9E",       # gray
    "Proportional": "#FF9800", # orange
    "Greedy": "#4CAF50",       # green
    "ITRI-GA": "#F44336",      # red
}

METHOD_MARKERS = {
    "NSGA-II": "o",
    "Random": "x",
    "Proportional": "s",
    "Greedy": "^",
    "ITRI-GA": "D",
    "SA-MOEA": "s",
}

# Additional colors for Phase 6
SAMOEA_COLORS = {
    "Vanilla": "#2196F3",   # blue
    "SA-MOEA": "#E91E63",   # pink
}

# Multi-algorithm comparison colors
ALGO_COLORS = {
    "NSGA-II": "#2196F3",     # blue
    "MOEA/D": "#FF9800",      # orange
    "NSGA-III": "#4CAF50",    # green
    "SA-MOEA (GNN)": "#E91E63",  # pink
    "SA-MOEA (MLP)": "#9C27B0",  # purple
}

ALGO_MARKERS = {
    "NSGA-II": "o",
    "MOEA/D": "s",
    "NSGA-III": "^",
    "SA-MOEA (GNN)": "D",
    "SA-MOEA (MLP)": "v",
}

def plot_pareto_front(
    nsga2_front: np.ndarray,
    baseline_points: dict[str, tuple[float, float]],
    output_path: Path | str,
    title: str = "Pareto Front: NSGA-II vs Baselines",
):
    """Plot Pareto front with baseline comparison points.

    Args:
        nsga2_front: (n_solutions, 2) array of [-profit, TMI].
        baseline_points: dict of {method_name: (neg_profit, tmi)}.
        output_path: Path to save the figure.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot NSGA-II Pareto front (convert neg_profit to profit)
    profits = -nsga2_front[:, 0]
    tmis = nsga2_front[:, 1]

    # Sort by profit for connected line
    sort_idx = np.argsort(profits)
    ax.plot(
        profits[sort_idx], tmis[sort_idx],
        "-o", color=METHOD_COLORS["NSGA-II"],
        markersize=8, linewidth=1.5, alpha=0.8,
        label=f"NSGA-II ({len(profits)} solutions)",
        zorder=5,
    )

    # Plot baseline points
    for name, (neg_profit, tmi) in baseline_points.items():
        color = METHOD_COLORS.get(name, "#000000")
        marker = METHOD_MARKERS.get(name, "x")
        ax.scatter(
            -neg_profit, tmi,
            c=color, marker=marker, s=150, linewidths=2,
            edgecolors="black", label=name, zorder=10,
        )

    ax.set_xlabel("Retailer Profit (NTD)")
    ax.set_ylabel("Temporal Mismatch Index (TMI)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Annotate trade-off direction
    ax.annotate(
        "Better →", xy=(0.95, 0.02), xycoords="axes fraction",
        ha="right", fontsize=10, color="gray",
    )
    ax.annotate(
        "← Better", xy=(0.02, 0.02), xycoords="axes fraction",
        ha="left", fontsize=10, color="gray",
    )

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")

def plot_boxplots(
    metrics_by_method: dict[str, list[float]],
    metric_name: str,
    output_path: Path | str,
    title: str | None = None,
    ylabel: str | None = None,
    higher_is_better: bool = True,
):
    """Box plot comparing a metric across methods and seeds.

    Args:
        metrics_by_method: {method_name: [values_per_seed]}.
        metric_name: Name of the metric.
        output_path: Path to save figure.
        title: Plot title.
        ylabel: Y-axis label.
        higher_is_better: Arrow annotation direction.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(metrics_by_method.keys())
    data = [metrics_by_method[m] for m in methods]
    colors = [METHOD_COLORS.get(m, "#000000") for m in methods]

    bp = ax.boxplot(data, tick_labels=methods, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    if title is None:
        title = f"{metric_name} Comparison Across Methods"
    if ylabel is None:
        ylabel = metric_name

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)

    direction = "↑ Better" if higher_is_better else "↓ Better"
    ax.annotate(
        direction, xy=(0.98, 0.98), xycoords="axes fraction",
        ha="right", va="top", fontsize=10, color="gray",
    )

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")

def plot_convergence(
    hv_history: list[float],
    output_path: Path | str,
    title: str = "Hypervolume Convergence",
):
    """Plot hypervolume over generations.

    Args:
        hv_history: List of hypervolume values per generation.
        output_path: Path to save figure.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    generations = range(1, len(hv_history) + 1)
    ax.plot(generations, hv_history, "-", color=METHOD_COLORS["NSGA-II"], linewidth=2)
    ax.fill_between(generations, hv_history, alpha=0.1, color=METHOD_COLORS["NSGA-II"])

    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")

def plot_radar_chart(
    method_scores: dict[str, dict[str, float]],
    output_path: Path | str,
    title: str = "Method Comparison Radar Chart",
):
    """Radar chart comparing methods across multiple metrics.

    Args:
        method_scores: {method_name: {metric_name: normalized_score}}.
        output_path: Path to save figure.
    """
    if not method_scores:
        return

    first_method = next(iter(method_scores.values()))
    metrics = list(first_method.keys())
    n_metrics = len(metrics)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for method_name, scores in method_scores.items():
        values = [scores[m] for m in metrics]
        values += values[:1]
        color = METHOD_COLORS.get(method_name, "#000000")
        ax.plot(angles, values, "o-", linewidth=2, label=method_name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, size=10)
    ax.set_ylim(0, 1.1)
    ax.set_title(title, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")

def generate_results_table(
    results: dict[str, dict[str, float]],
    output_path: Path | str,
):
    """Generate a LaTeX-formatted results table.

    Args:
        results: {method_name: {metric_name: value}}.
        output_path: Path to save the .tex file.
    """
    if not results:
        return

    first = next(iter(results.values()))
    metrics = list(first.keys())

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Phase 1 Experiment Results}}")
    lines.append("\\begin{tabular}{l" + "r" * len(metrics) + "}")
    lines.append("\\hline")

    # Header
    header = "Method & " + " & ".join(metrics) + " \\\\"
    lines.append(header)
    lines.append("\\hline")

    # Data rows
    for method, values in results.items():
        row = method
        for m in metrics:
            v = values.get(m, 0)
            if abs(v) >= 1000:
                row += f" & {v:,.0f}"
            else:
                row += f" & {v:.4f}"
        row += " \\\\"
        lines.append(row)

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {output_path}")

# ---------------------------------------------------------------------------
# Phase 6 visualization functions
# ---------------------------------------------------------------------------

def plot_convergence_comparison(
    hv_histories: dict[str, list[list[float]]],
    output_path: Path | str,
    title: str = "HV Convergence: Vanilla vs SA-MOEA",
):
    """Plot HV vs generation with mean +/- std bands for multiple methods.

    Args:
        hv_histories: {method_name: [[hv_per_gen_seed0], [hv_per_gen_seed1], ...]}.
            Each inner list is the HV series for one seed.
        output_path: Path to save figure.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, seeds_data in hv_histories.items():
        # Align to shortest series
        min_len = min(len(s) for s in seeds_data)
        arr = np.array([s[:min_len] for s in seeds_data])  # (n_seeds, n_gens)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        gens = np.arange(1, min_len + 1)

        color = SAMOEA_COLORS.get(method, "#000000")
        ax.plot(gens, mean, "-", color=color, linewidth=2, label=method)
        ax.fill_between(gens, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")

def plot_scalability(
    scale_data: list[dict],
    output_path: Path | str,
    title: str = "Scalability: Problem Size vs Wall Time",
):
    """Log-log plot of n_var vs wall-clock time.

    Args:
        scale_data: List of dicts with keys 'n_var', 'time_mean', 'time_std'.
        output_path: Path to save figure.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    n_vars = [d["n_var"] for d in scale_data]
    times = [d["time_mean"] for d in scale_data]
    stds = [d.get("time_std", 0) for d in scale_data]

    ax.errorbar(
        n_vars, times, yerr=stds,
        fmt="o-", color="#2196F3", linewidth=2,
        markersize=10, capsize=5, capthick=2,
    )

    # Annotate scale labels
    for d in scale_data:
        ax.annotate(
            d.get("label", ""),
            (d["n_var"], d["time_mean"]),
            textcoords="offset points", xytext=(10, 5),
            fontsize=10,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Decision Variables (m × n)")
    ax.set_ylabel("Wall-Clock Time (seconds)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")

def plot_ablation(
    cf_values: list[float],
    hv_values: list[float],
    savings_values: list[float],
    output_path: Path | str,
    title: str = "Ablation: Correction Fraction vs HV & Savings",
):
    """Grouped bar chart for correction_fraction ablation study.

    Args:
        cf_values: List of correction_fraction values tested.
        hv_values: Corresponding mean HV (normalized to vanilla=1.0).
        savings_values: Corresponding mean eval savings (fraction, 0-1).
        output_path: Path to save figure.
        title: Plot title.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = np.arange(len(cf_values))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, hv_values, width,
                    label="HV Ratio (SA-MOEA / Vanilla)",
                    color="#2196F3", alpha=0.8)
    ax1.set_ylabel("HV Ratio (1.0 = Vanilla)", color="#2196F3")
    ax1.axhline(y=1.0, color="#2196F3", linestyle="--", alpha=0.5)
    ax1.tick_params(axis="y", labelcolor="#2196F3")

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, [s * 100 for s in savings_values], width,
                    label="Eval Savings (%)", color="#E91E63", alpha=0.8)
    ax2.set_ylabel("True Eval Savings (%)", color="#E91E63")
    ax2.tick_params(axis="y", labelcolor="#E91E63")

    ax1.set_xlabel("Correction Fraction (cf)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{cf:.1f}" for cf in cf_values])
    ax1.set_title(title)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")

def plot_hv_boxplot(
    hv_by_method: dict[str, list[float]],
    output_path: Path | str,
    title: str = "Hypervolume Distribution (10 Seeds)",
    scale_label: str = "",
):
    """Box plot of HV distributions across seeds for Vanilla vs SA-MOEA.

    Args:
        hv_by_method: {method_name: [hv_per_seed]}.
        output_path: Path to save figure.
        title: Plot title.
        scale_label: Scale label for subtitle.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    methods = list(hv_by_method.keys())
    data = [hv_by_method[m] for m in methods]
    colors = [SAMOEA_COLORS.get(m, "#666666") for m in methods]

    bp = ax.boxplot(data, tick_labels=methods, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay individual points
    for i, (method, vals) in enumerate(hv_by_method.items()):
        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
        ax.scatter(
            np.full(len(vals), i + 1) + jitter, vals,
            c=colors[i], alpha=0.7, s=30, zorder=5,
        )

    full_title = f"{title}\n{scale_label}" if scale_label else title
    ax.set_title(full_title)
    ax.set_ylabel("Hypervolume")
    ax.grid(True, axis="y", alpha=0.3)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")

def plot_sensitivity(
    sensitivity_results: dict[str, dict],
    output_path: Path | str,
    title: str = "Parameter Sensitivity Analysis",
):
    """Plot parameter sensitivity: 1×N subplot, one per parameter.

    Args:
        sensitivity_results: {param_name: {
            "values": [v1, v2, ...],
            "default": v_default,
            "label": "display name",
            "metrics": {
                "HV": {"mean": [...], "std": [...]},
                "Profit": {"mean": [...], "std": [...]},
                "TMI": {"mean": [...], "std": [...]},
                ...
            },
        }}.
        output_path: Path to save figure.
        title: Plot title.
    """
    params = list(sensitivity_results.keys())
    n_params = len(params)
    fig, axes = plt.subplots(1, n_params, figsize=(6 * n_params, 5))
    if n_params == 1:
        axes = [axes]

    metric_colors = {
        "HV": "#2196F3",
        "Profit": "#4CAF50",
        "TMI": "#F44336",
        "RE100 Rate": "#FF9800",
        "Surplus Ratio": "#9C27B0",
    }

    for ax, param_name in zip(axes, params):
        pdata = sensitivity_results[param_name]
        x_vals = pdata["values"]
        default_val = pdata.get("default")
        label = pdata.get("label", param_name)

        for metric_name, metric_data in pdata["metrics"].items():
            means = metric_data["mean"]
            stds = metric_data.get("std", [0] * len(means))
            color = metric_colors.get(metric_name, "#666666")

            # Normalize to default=1.0 for comparable y-axis
            if default_val is not None and default_val in x_vals:
                idx = x_vals.index(default_val)
                base = means[idx] if means[idx] != 0 else 1.0
                norm_means = [m / base for m in means]
                norm_stds = [s / abs(base) for s in stds]
            else:
                norm_means = means
                norm_stds = stds

            ax.errorbar(
                x_vals, norm_means, yerr=norm_stds,
                fmt="o-", color=color, linewidth=1.5,
                markersize=6, capsize=3, label=metric_name,
            )

        # Mark default value
        if default_val is not None:
            ax.axvline(x=default_val, color="gray", linestyle="--", alpha=0.5)
            ax.annotate(
                "default", xy=(default_val, ax.get_ylim()[1]),
                xytext=(0, 5), textcoords="offset points",
                ha="center", fontsize=9, color="gray",
            )

        ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.3)
        ax.set_xlabel(label)
        ax.set_ylabel("Normalized Metric (default = 1.0)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")

# ---------------------------------------------------------------------------
# Phase 6H/6I visualization functions
# ---------------------------------------------------------------------------

def plot_multi_pareto_fronts(
    fronts: dict[str, np.ndarray],
    output_path: Path | str,
    title: str = "Pareto Front Comparison",
    ref_point: np.ndarray | None = None,
):
    """Plot 2D scatter of multiple Pareto fronts for method comparison.

    Args:
        fronts: {method_name: (n_solutions, 2) array of [neg_profit, tmi]}.
        output_path: Path to save figure.
        title: Plot title.
        ref_point: Optional reference point to display.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    for method, F in fronts.items():
        if len(F) == 0:
            continue
        profits = -F[:, 0]
        tmis = F[:, 1]

        color = ALGO_COLORS.get(method, "#666666")
        marker = ALGO_MARKERS.get(method, "o")

        # Sort by profit for connected line
        sort_idx = np.argsort(profits)
        ax.plot(
            profits[sort_idx], tmis[sort_idx],
            f"-{marker}", color=color,
            markersize=6, linewidth=1.2, alpha=0.8,
            label=f"{method} ({len(F)} sol.)",
            zorder=5,
        )

    if ref_point is not None:
        ax.scatter(
            -ref_point[0], ref_point[1],
            c="red", marker="*", s=200, zorder=10,
            label="Ref. point",
        )

    ax.set_xlabel("Retailer Profit (NTD)")
    ax.set_ylabel("Temporal Mismatch Index (TMI)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")

def plot_multi_hv_boxplot(
    hv_by_method: dict[str, list[float]],
    output_path: Path | str,
    title: str = "Hypervolume Distribution",
    scale_label: str = "",
):
    """Box plot of HV distributions for multiple algorithms (3+).

    Args:
        hv_by_method: {method_name: [hv_per_seed]}.
        output_path: Path to save figure.
        title: Plot title.
        scale_label: Scale label for subtitle.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(hv_by_method.keys())
    data = [hv_by_method[m] for m in methods]
    colors = [ALGO_COLORS.get(m, "#666666") for m in methods]

    bp = ax.boxplot(data, tick_labels=methods, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay individual points
    for i, (method, vals) in enumerate(hv_by_method.items()):
        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
        ax.scatter(
            np.full(len(vals), i + 1) + jitter, vals,
            c=colors[i], alpha=0.7, s=30, zorder=5,
        )

    full_title = f"{title}\n{scale_label}" if scale_label else title
    ax.set_title(full_title)
    ax.set_ylabel("Hypervolume")
    ax.grid(True, axis="y", alpha=0.3)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")

def plot_walltime_comparison(
    walltime_data: list[dict],
    output_path: Path | str,
    title: str = "Wall-Time Comparison: Vanilla vs SA-MOEA",
):
    """Grouped bar chart comparing wall-time across scales and methods.

    Args:
        walltime_data: List of dicts with keys:
            'scale', 'vanilla_time_mean', 'vanilla_time_std',
            'samoea_time_mean', 'samoea_time_std',
            'vanilla_evals_mean', 'samoea_evals_mean'.
        output_path: Path to save figure.
        title: Plot title.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    scales = [d["scale"] for d in walltime_data]
    x = np.arange(len(scales))
    width = 0.35

    # Left panel: Wall time
    vanilla_times = [d["vanilla_time_mean"] for d in walltime_data]
    vanilla_time_std = [d.get("vanilla_time_std", 0) for d in walltime_data]
    samoea_times = [d["samoea_time_mean"] for d in walltime_data]
    samoea_time_std = [d.get("samoea_time_std", 0) for d in walltime_data]

    ax1.bar(x - width / 2, vanilla_times, width, yerr=vanilla_time_std,
            label="Vanilla NSGA-II", color="#2196F3", alpha=0.8, capsize=3)
    ax1.bar(x + width / 2, samoea_times, width, yerr=samoea_time_std,
            label="SA-MOEA", color="#E91E63", alpha=0.8, capsize=3)

    # Annotate speedup
    for i in range(len(scales)):
        if samoea_times[i] > 0:
            speedup = vanilla_times[i] / samoea_times[i]
            if speedup > 1.05:
                ax1.annotate(
                    f"{speedup:.1f}x",
                    xy=(x[i] + width / 2, samoea_times[i]),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=9, color="#E91E63", fontweight="bold",
                )

    ax1.set_xlabel("Problem Scale")
    ax1.set_ylabel("Wall-Clock Time (seconds)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scales)
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_title("Wall-Clock Time")

    # Right panel: Evaluations
    vanilla_evals = [d["vanilla_evals_mean"] for d in walltime_data]
    samoea_evals = [d["samoea_evals_mean"] for d in walltime_data]

    ax2.bar(x - width / 2, vanilla_evals, width,
            label="Vanilla NSGA-II", color="#2196F3", alpha=0.8)
    ax2.bar(x + width / 2, samoea_evals, width,
            label="SA-MOEA (true evals)", color="#E91E63", alpha=0.8)

    # Annotate savings
    for i in range(len(scales)):
        if vanilla_evals[i] > 0:
            savings = (1.0 - samoea_evals[i] / vanilla_evals[i]) * 100
            ax2.annotate(
                f"-{savings:.0f}%",
                xy=(x[i] + width / 2, samoea_evals[i]),
                xytext=(0, 5), textcoords="offset points",
                ha="center", fontsize=9, color="#E91E63", fontweight="bold",
            )

    ax2.set_xlabel("Problem Scale")
    ax2.set_ylabel("Number of True Evaluations")
    ax2.set_xticks(x)
    ax2.set_xticklabels(scales)
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_title("True Evaluations")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")
