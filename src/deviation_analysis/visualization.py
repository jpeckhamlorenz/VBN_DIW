"""Visualization functions for deviation analysis results.

Generates publication-quality figures:
- Spatial deviation maps projected onto CAD surface
- KDE distribution comparison plots
- Box-and-whisker deviation comparisons
- Summary metric tables
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from deviation_analysis.config import VisualizationConfig
from deviation_analysis.deviation import DeviationMetrics

# Publication-quality defaults (Nature Communications style)
_FONT = {"family": "serif", "size": 10}
_FIG_SINGLE_COL_MM = 89.0
_FIG_DOUBLE_COL_MM = 183.0
_MM_TO_INCHES = 1.0 / 25.4


def _setup_mpl() -> None:
    """Apply publication-quality matplotlib defaults."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _fig_size_from_mm(width_mm: float, aspect: float = 0.75) -> tuple[float, float]:
    """Convert figure width in mm to (width, height) in inches."""
    w = width_mm * _MM_TO_INCHES
    return (w, w * aspect)


def plot_spatial_deviation_map(
    cad_mesh_path: Path,
    scan_points_aligned: np.ndarray,
    signed_distances: np.ndarray,
    method_name: str,
    config: VisualizationConfig,
    output_path: Path | None = None,
) -> None:
    """Render signed deviations as a color map on the CAD mesh surface.

    Uses pyvista to interpolate scan point deviations onto the mesh surface
    and render with a diverging colormap (blue = deficit, white = nominal,
    red = excess).

    Parameters
    ----------
    cad_mesh_path
        Path to the STL file for rendering.
    scan_points_aligned
        (M, 3) aligned scan points.
    signed_distances
        (M,) signed deviations corresponding to scan_points_aligned.
    method_name
        Name for the figure title.
    config
        Visualization parameters (color limits, DPI, etc.).
    output_path
        If provided, save figure to this path. Otherwise display interactively.
    """
    import pyvista as pv

    mesh = pv.read(str(cad_mesh_path))
    cloud = pv.PolyData(scan_points_aligned)
    cloud["deviation"] = signed_distances

    # Interpolate scan deviations onto the mesh surface
    interpolation_radius = config.clim[1] * 5  # reasonable search radius
    mesh_with_dev = mesh.interpolate(
        cloud,
        radius=max(interpolation_radius, 0.5),
        sharpness=5,
    )

    plotter = pv.Plotter(off_screen=output_path is not None)
    plotter.add_mesh(
        mesh_with_dev,
        scalars="deviation",
        cmap="RdBu_r",
        clim=list(config.clim),
        scalar_bar_args={
            "title": "Deviation (mm)",
            "title_font_size": 12,
            "label_font_size": 10,
            "shadow": False,
            "n_labels": 5,
            "fmt": "%.2f",
        },
    )
    plotter.add_text(method_name, position="upper_left", font_size=12)
    plotter.camera_position = "xy"
    plotter.set_background("white")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(
            str(output_path),
            window_size=[1920, 1080],
        )
        plotter.close()
    else:
        plotter.show()


def plot_deviation_kde(
    method_distances: dict[str, np.ndarray],
    config: VisualizationConfig,
    output_path: Path | None = None,
    colors: dict[str, str] | None = None,
) -> None:
    """Plot overlaid KDE distributions of signed deviations for each method.

    X axis: signed deviation (mm). Y axis: probability density.
    Vertical dashed line at 0. One curve per method with legend.

    Parameters
    ----------
    method_distances
        Mapping from display name to signed distance array.
    config
        Visualization parameters.
    output_path
        If provided, save figure. Otherwise display.
    colors
        Optional mapping from method name to matplotlib color string.
    """
    _setup_mpl()
    fig, ax = plt.subplots(figsize=_fig_size_from_mm(config.figure_width_mm))

    # Default color cycle
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Determine shared x-axis range from config clim with some padding
    x_min, x_max = config.clim[0] * 1.5, config.clim[1] * 1.5
    x_eval = np.linspace(x_min, x_max, 500)

    for i, (name, distances) in enumerate(method_distances.items()):
        color = (colors or {}).get(name, default_colors[i % len(default_colors)])
        # Subsample if very large to keep KDE fast
        d = distances
        if len(d) > 100_000:
            rng = np.random.default_rng(42)
            d = rng.choice(d, size=100_000, replace=False)
        kde = gaussian_kde(d)
        ax.plot(x_eval, kde(x_eval), label=name, color=color, linewidth=1.5)

    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Signed deviation (mm)")
    ax.set_ylabel("Probability density")
    ax.legend(frameon=False)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=config.dpi)
        plt.close(fig)
    else:
        plt.show()


def plot_boxplot_comparison(
    method_distances: dict[str, np.ndarray],
    config: VisualizationConfig,
    output_path: Path | None = None,
) -> None:
    """Box-and-whisker plot comparing deviation distributions across methods.

    Parameters
    ----------
    method_distances
        Mapping from display name to signed distance array.
    config
        Visualization parameters.
    output_path
        If provided, save figure. Otherwise display.
    """
    _setup_mpl()
    fig, ax = plt.subplots(figsize=_fig_size_from_mm(config.figure_width_mm, aspect=0.5))

    names = list(method_distances.keys())
    data = [method_distances[n] for n in names]

    bp = ax.boxplot(
        data,
        labels=names,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"linewidth": 1.0},
        boxprops={"linewidth": 1.0},
    )

    # Color boxes with a light palette
    box_colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_ylabel("Signed deviation (mm)")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=config.dpi)
        plt.close(fig)
    else:
        plt.show()


def plot_metrics_summary_table(
    metrics_dict: dict[str, DeviationMetrics],
    config: VisualizationConfig,
    output_path: Path | None = None,
) -> None:
    """Render a publication-quality table of metrics for all methods.

    Columns: Method, RMSD (mm), MSD (mm), 95th% (mm), Excess Vol (mm^3),
    Deficit Vol (mm^3).

    Parameters
    ----------
    metrics_dict
        Mapping from display name to DeviationMetrics.
    config
        Visualization parameters.
    output_path
        If provided, save as PNG. Otherwise display.
    """
    _setup_mpl()

    col_labels = [
        "Method",
        "RMSD\n(mm)",
        "MSD\n(mm)",
        "MAE\n(mm)",
        "95th %\n(mm)",
        "Excess Vol\n(mm\u00b3)",
        "Deficit Vol\n(mm\u00b3)",
    ]

    cell_text = []
    for name, m in metrics_dict.items():
        cell_text.append([
            name,
            f"{m.rmsd:.4f}",
            f"{m.msd:.4f}",
            f"{m.mae:.4f}",
            f"{m.percentile_abs:.4f}",
            f"{m.excess_volume_mm3:.3f}",
            f"{m.deficit_volume_mm3:.3f}",
        ])

    fig, ax = plt.subplots(
        figsize=_fig_size_from_mm(config.figure_width_mm, aspect=0.15 * len(cell_text) + 0.15)
    )
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Style header row
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#e6e6e6")
        table[0, j].set_text_props(weight="bold")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=config.dpi)
        plt.close(fig)
    else:
        plt.show()


def export_metrics_csv(
    metrics_dict: dict[str, DeviationMetrics],
    output_path: Path,
) -> None:
    """Export metrics to a CSV file for direct inclusion in papers.

    Parameters
    ----------
    metrics_dict
        Mapping from display name to DeviationMetrics.
    output_path
        Path to write the CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "method",
        "rmsd_mm",
        "msd_mm",
        "mae_mm",
        "percentile_95_mm",
        "max_abs_mm",
        "excess_volume_mm3",
        "deficit_volume_mm3",
        "n_points",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, m in metrics_dict.items():
            writer.writerow({
                "method": name,
                "rmsd_mm": f"{m.rmsd:.6f}",
                "msd_mm": f"{m.msd:.6f}",
                "mae_mm": f"{m.mae:.6f}",
                "percentile_95_mm": f"{m.percentile_abs:.6f}",
                "max_abs_mm": f"{m.max_abs_deviation:.6f}",
                "excess_volume_mm3": f"{m.excess_volume_mm3:.6f}",
                "deficit_volume_mm3": f"{m.deficit_volume_mm3:.6f}",
                "n_points": m.n_points,
            })
