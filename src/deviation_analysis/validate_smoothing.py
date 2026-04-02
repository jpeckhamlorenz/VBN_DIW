"""Validation script for anisotropic scan-direction smoothing.

Runs the deviation analysis pipeline twice for each scan — once with raw
bead points and once with smoothed — then compares signed-distance metrics
and distributions.  Output is intended for supplementary materials in the
Nature Communications paper.

Run from the ``src/`` directory::

    python -m deviation_analysis.validate_smoothing
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from deviation_analysis.batch import process_single_scan
from deviation_analysis.config import (
    MethodSpec,
    PipelineConfig,
    SmoothingConfig,
    VisualizationConfig,
)
from deviation_analysis.deviation import ks_test
from deviation_analysis.visualization import plot_deviation_kde

# ---------------------------------------------------------------------------
# Configuration — adjust paths and scan selection as needed
# ---------------------------------------------------------------------------
DATA_DIR = Path("demos/m")
STL_FILE = DATA_DIR / "m_ideal.stl"
OUTPUT_DIR = Path("deviation_analysis/output/smoothing_validation")

# Methods and cycles to validate (first cycle per method is used)
METHODS: list[MethodSpec] = [
    MethodSpec(
        name="m_VBN_05",
        display_name="VBN (v5)",
        toolpath_csv="m_VBN_05.csv",
        cycle_csvs=["m_VBN_05_cycle_001.csv"],
    ),
]


def _run_pipeline(
    method: MethodSpec,
    *,
    use_smoothed: bool,
    config: PipelineConfig,
) -> tuple[np.ndarray, "object"]:
    """Run the pipeline for one scan with a specific smoothing toggle."""
    cfg = copy.deepcopy(config)
    cfg.smoothing = replace(cfg.smoothing, use_smoothed=use_smoothed)
    scan_csv = cfg.scan_path(method.cycle_csvs[0])
    toolpath_csv = cfg.scan_path(method.toolpath_csv)
    return process_single_scan(scan_csv, toolpath_csv, STL_FILE, cfg)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        cache_dir=Path("deviation_analysis/cache"),
        stl_file=STL_FILE.name,
        smoothing=SmoothingConfig(enabled=True, use_smoothed=True),
    )

    for method in METHODS:
        print(f"\n{'=' * 60}")
        print(f"  Validating smoothing: {method.display_name}")
        print(f"{'=' * 60}")

        # --- Run raw and smoothed ---
        print("\n  Running pipeline (raw)...")
        raw_distances, raw_metrics = _run_pipeline(
            method, use_smoothed=False, config=config
        )
        print("  Running pipeline (smoothed)...")
        sm_distances, sm_metrics = _run_pipeline(
            method, use_smoothed=True, config=config
        )

        # --- Metrics comparison ---
        print(f"\n  {'Metric':<28s} {'Raw':>10s} {'Smoothed':>10s} {'Delta':>10s}")
        print(f"  {'-' * 58}")
        for label, raw_val, sm_val in [
            ("RMSD (mm)", raw_metrics.rmsd, sm_metrics.rmsd),
            ("MSD (mm)", raw_metrics.msd, sm_metrics.msd),
            ("MAE (mm)", raw_metrics.mae, sm_metrics.mae),
            ("95th %ile abs (mm)", raw_metrics.percentile_abs, sm_metrics.percentile_abs),
            ("Max abs (mm)", raw_metrics.max_abs_deviation, sm_metrics.max_abs_deviation),
        ]:
            delta = sm_val - raw_val
            print(f"  {label:<28s} {raw_val:>10.4f} {sm_val:>10.4f} {delta:>+10.4f}")

        # --- KS test ---
        ks_stat, ks_pval = ks_test(raw_distances, sm_distances)
        print(f"\n  KS test (raw vs smoothed): D={ks_stat:.4f}, p={ks_pval:.2e}")

        # --- Point-wise change ---
        n_common = min(len(raw_distances), len(sm_distances))
        pw_diff = np.abs(raw_distances[:n_common] - sm_distances[:n_common])
        print(f"  Point-wise |d_sm - d_raw|: mean={pw_diff.mean():.4f} mm, "
              f"95th={np.percentile(pw_diff, 95):.4f} mm")

        # --- Comparison figure ---
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Subplot 1: Overlaid KDE
        ax = axes[0]
        for label, dists, color in [
            ("Raw", raw_distances, "#1f77b4"),
            ("Smoothed", sm_distances, "#d62728"),
        ]:
            from scipy.stats import gaussian_kde
            subsample = dists[::max(1, len(dists) // 100_000)]
            kde = gaussian_kde(subsample)
            xs = np.linspace(dists.min(), dists.max(), 500)
            ax.plot(xs, kde(xs), label=label, color=color, linewidth=1.5)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Signed deviation (mm)")
        ax.set_ylabel("Density")
        ax.set_title("Distribution comparison")
        ax.legend(frameon=False)

        # Subplot 2: Summary table
        ax2 = axes[1]
        ax2.axis("off")
        table_data = [
            ["RMSD (mm)", f"{raw_metrics.rmsd:.4f}", f"{sm_metrics.rmsd:.4f}"],
            ["MSD (mm)", f"{raw_metrics.msd:.4f}", f"{sm_metrics.msd:.4f}"],
            ["MAE (mm)", f"{raw_metrics.mae:.4f}", f"{sm_metrics.mae:.4f}"],
            ["95th %ile (mm)", f"{raw_metrics.percentile_abs:.4f}", f"{sm_metrics.percentile_abs:.4f}"],
            ["KS stat", "", f"{ks_stat:.4f}"],
            ["KS p-value", "", f"{ks_pval:.2e}"],
            ["|d_sm-d_raw| mean", "", f"{pw_diff.mean():.4f}"],
        ]
        table = ax2.table(
            cellText=table_data,
            colLabels=["Metric", "Raw", "Smoothed"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)
        # Style header row
        for j in range(3):
            table[0, j].set_facecolor("#dddddd")
            table[0, j].set_text_props(weight="bold")
        ax2.set_title("Metrics summary")

        fig.suptitle(f"Smoothing validation — {method.display_name}", fontsize=12)
        fig.tight_layout()

        out_path = OUTPUT_DIR / f"{method.name}_smoothing_validation.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"\n  Saved: {out_path}")
        plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
