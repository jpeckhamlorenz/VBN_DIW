"""
Compare deviation metrics across printing methods.

This script processes multiple scan files against the same CAD reference,
computes deviation metrics for each, and prints a comparative summary.

All intermediate results are left as module-level variables so you can
explore them interactively in an IPython console:

    %run compare_methods.py
    results["VBN (v5)"].metrics.rmsd
    results["VBN (v5)"].signed_distances
    results["Static naive"].aligned_bead

Usage:
    python compare_methods.py          # run with defaults
    %run compare_methods.py            # in IPython / PyCharm console
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from deviation_analysis.config import (
    DeviationConfig,
    RegistrationConfig,
    ScanConfig,
    SmoothingConfig,
    VisualizationConfig,
)
from deviation_analysis.deviation import (
    DeviationMetrics,
    compute_metrics,
    compute_pairwise_ks,
    compute_signed_distances,
)
from deviation_analysis.loader import (
    compute_y_spacing,
    height_correct,
    load_cad_mesh,
    load_scan_csv,
    load_toolpath_csv,
    raster_to_point_cloud,
    save_points_as_stl,
)
from deviation_analysis.registration import (
    flatten_point_cloud,
    register_scan_to_cad,
    segment_bed_from_bead,
)
from deviation_analysis.smoothing import add_sidewalls, remove_islands, smooth_anisotropic_grid
from deviation_analysis.visualization import (
    export_metrics_csv,
    plot_boxplot_comparison,
    plot_deviation_kde,
    plot_metrics_summary_table,
    plot_spatial_deviation_map,
)

# ============================================================================
#  CONFIGURATION — edit these to change which scans are compared
# ============================================================================

DATA_DIR = Path('demos/m')
STL_FILE = DATA_DIR / 'm_ideal.stl'
OUTPUT_DIR = Path('deviation_analysis/output/comparison')
STL_OUTPUT_DIR = OUTPUT_DIR / 'scan_stls'

# Each entry: (display_name, scan_cycle_csv, toolpath_csv)
# Change these to compare different scans or methods.
# VBN (v5) is the primary method; the rest are baselines.
METHODS = [
    ('VBN (v9)', 'm_VBN_09_cycle_003.csv', 'm_VBN_05.csv'),
    ('VBN (v5)', 'm_VBN_05_cycle_001.csv', 'm_VBN_05.csv'),
    ('VBN (v2)', 'm_VBN_2_cycle_002.csv', 'm_VBN_2.csv'),
    ('Static ideal', 'm_static_ideal_cycle_001.csv', 'm_static_ideal.csv'),
    ('Static naive', 'm_static_naive_cycle_001.csv', 'm_static_naive.csv'),
]

# Pipeline parameters — sensible defaults, tweak as needed
SCAN_CONFIG = ScanConfig(
    resolution=0.02,  # mm per column
    scan_speed=5.0,  # mm/s
    scan_rate=1000.0,  # Hz
    invalid_threshold=10.0,
    scale_factor=1e-3,
)

REG_CONFIG = RegistrationConfig()  # all defaults
SMOOTH_CONFIG = SmoothingConfig()  # uses defaults from config.py
DEV_CONFIG = DeviationConfig()  # percentile=95
VIS_CONFIG = VisualizationConfig()  # clim=(-0.3, 0.3), dpi=300


# ============================================================================
#  DATA CONTAINER — one per method, holds everything for interactive use
# ============================================================================


@dataclass
class MethodResult:
    """All pipeline outputs for one scan, kept around for exploration."""

    name: str
    scan_csv: str
    toolpath_csv: str

    # Raw loaded data
    data_mm: np.ndarray  # (rows, cols) raster in mm
    toolpath_xyz: np.ndarray  # (N, 3) scan trajectory

    # Processed point cloud
    points: np.ndarray  # (P, 3) full point cloud
    valid_mask: np.ndarray  # (P,) bool

    # Flattened
    flat_points: np.ndarray  # (P, 3) after RANSAC plane removal
    rotation: np.ndarray  # (3, 3) flattening rotation
    intercept: float  # Z offset removed

    # Segmented
    bead_points: np.ndarray  # (M, 3) bead only
    bed_points: np.ndarray  # (K, 3) bed only

    # Registered
    transform: np.ndarray  # (4, 4) scan -> CAD
    fitness: float
    inlier_rmse: float
    aligned_bead: np.ndarray  # (M, 3) bead in CAD frame

    # Deviation
    signed_distances: np.ndarray  # (M,) signed distance per bead point
    metrics: DeviationMetrics


# ============================================================================
#  PROCESSING FUNCTIONS
# ============================================================================


def process_one_method(
    name: str,
    scan_csv: str,
    toolpath_csv: str,
    cad_mesh_o3d,
    cad_mesh_trimesh,
) -> MethodResult:
    """Run the full pipeline for a single method and return all intermediates."""
    scan_path = DATA_DIR / scan_csv
    tool_path = DATA_DIR / toolpath_csv
    print(f'\n--- Processing: {name} ---')
    print(f'    Scan: {scan_csv}')
    print(f'    Tool: {toolpath_csv}')

    # 1. Load
    data_mm = load_scan_csv(scan_path, SCAN_CONFIG)
    _time, toolpath_xyz = load_toolpath_csv(tool_path)
    print(f'    Loaded raster: {data_mm.shape[0]} rows x {data_mm.shape[1]} cols')

    # Physical Y row spacing from toolpath (handles mixed scan speeds correctly)
    y_spacing = compute_y_spacing(toolpath_xyz, data_mm.shape[0])
    print(f'    Y row spacing: {y_spacing:.5f} mm (from toolpath)')

    # 2. Height correct + point cloud
    data_corrected = height_correct(data_mm, toolpath_xyz)
    points, valid_mask = raster_to_point_cloud(data_corrected, toolpath_xyz, SCAN_CONFIG)
    n_valid = int(valid_mask.sum())
    print(f'    Valid points: {n_valid:,} / {len(valid_mask):,}')

    # 3. Flatten
    flat_points, rotation, intercept = flatten_point_cloud(points, valid_mask)
    print(f'    Flattened (intercept={intercept:.3f} mm)')

    # 4. Segment
    bead_points, bed_points, bead_mask = segment_bed_from_bead(flat_points, valid_mask, REG_CONFIG.bed_z_threshold)
    print(f'    Segmented: {len(bead_points):,} bead / {len(bed_points):,} bed')

    # 4.5 Smoothing + sidewalls
    n_rows, n_cols = data_mm.shape
    if SMOOTH_CONFIG.enabled:
        bead_points = smooth_anisotropic_grid(
            flat_points=flat_points,
            valid_mask=valid_mask,
            bead_mask=bead_mask,
            n_rows=n_rows,
            n_cols=n_cols,
            sigma_scan=SMOOTH_CONFIG.sigma_scan,
            sigma_perp=SMOOTH_CONFIG.sigma_perp,
            scan_direction=np.array(SMOOTH_CONFIG.scan_direction),
            x_spacing=SCAN_CONFIG.resolution,
            y_spacing=y_spacing,
            n_iterations=SMOOTH_CONFIG.n_iterations,
        )
        print(f'    Smoothed (σ_scan={SMOOTH_CONFIG.sigma_scan}, σ_perp={SMOOTH_CONFIG.sigma_perp})')
        if SMOOTH_CONFIG.remove_islands:
            bead_points, bead_mask = remove_islands(
                bead_points,
                bead_mask,
                n_rows,
                n_cols,
                closing_radius=SMOOTH_CONFIG.island_closing_radius,
                min_distance=SMOOTH_CONFIG.island_min_distance,
                x_spacing=SCAN_CONFIG.resolution,
                y_spacing=y_spacing,
            )
        if SMOOTH_CONFIG.add_sidewalls:
            bead_points = add_sidewalls(
                bead_points,
                bead_mask=bead_mask,
                n_rows=n_rows,
                n_cols=n_cols,
                flat_points=flat_points,
                z_step=SMOOTH_CONFIG.sidewall_z_step,
                floor_z=0.0,
            )
            print(f'    Added sidewalls ({len(bead_points):,} total points)')

    # 5. Register
    print('    Registering to CAD (FPFH + ICP)...')
    transform, fitness, inlier_rmse = register_scan_to_cad(bead_points, cad_mesh_o3d, REG_CONFIG)
    print(f'    Registration: fitness={fitness:.4f}, RMSE={inlier_rmse:.4f} mm')

    # 6. Transform bead points into CAD frame
    bead_hom = np.hstack([bead_points, np.ones((len(bead_points), 1))])
    aligned_bead = (transform @ bead_hom.T).T[:, :3]

    # 7. Signed distances
    signed_distances = compute_signed_distances(aligned_bead, cad_mesh_trimesh)
    print(f'    Distances: mean={signed_distances.mean():.4f}, std={signed_distances.std():.4f} mm')

    # 8. Metrics
    point_area = SCAN_CONFIG.resolution * y_spacing
    metrics = compute_metrics(signed_distances, point_area, DEV_CONFIG)

    return MethodResult(
        name=name,
        scan_csv=scan_csv,
        toolpath_csv=toolpath_csv,
        data_mm=data_mm,
        toolpath_xyz=toolpath_xyz,
        points=points,
        valid_mask=valid_mask,
        flat_points=flat_points,
        rotation=rotation,
        intercept=intercept,
        bead_points=bead_points,
        bed_points=bed_points,
        transform=transform,
        fitness=fitness,
        inlier_rmse=inlier_rmse,
        aligned_bead=aligned_bead,
        signed_distances=signed_distances,
        metrics=metrics,
    )


def print_comparison(results: dict[str, MethodResult]) -> None:
    """Print a formatted comparison table and per-method insights."""
    names = list(results.keys())
    print('\n' + '=' * 72)
    print('  METRICS COMPARISON')
    print('=' * 72)

    # Header
    header = f'{"Metric":<24}'
    for name in names:
        header += f'{name:>16}'
    print(header)
    print('-' * (24 + 16 * len(names)))

    # Rows
    rows = [
        ('RMSD (mm)', lambda m: f'{m.rmsd:.4f}'),
        ('MSD (mm)', lambda m: f'{m.msd:.4f}'),
        ('MAE (mm)', lambda m: f'{m.mae:.4f}'),
        ('95th %ile abs (mm)', lambda m: f'{m.percentile_abs:.4f}'),
        ('Max abs (mm)', lambda m: f'{m.max_abs_deviation:.4f}'),
        ('Excess vol (mm³)', lambda m: f'{m.excess_volume_mm3:.4f}'),
        ('Deficit vol (mm³)', lambda m: f'{m.deficit_volume_mm3:.4f}'),
        ('Points analyzed', lambda m: f'{m.n_points:,}'),
    ]
    for label, fmt_fn in rows:
        line = f'{label:<24}'
        for name in names:
            line += f'{fmt_fn(results[name].metrics):>16}'
        print(line)

    # Per-method insights
    print('\n' + '-' * 72)
    print('  INSIGHTS')
    print('-' * 72)

    # Best RMSD
    best_rmsd_name = min(names, key=lambda n: results[n].metrics.rmsd)
    worst_rmsd_name = max(names, key=lambda n: results[n].metrics.rmsd)
    best = results[best_rmsd_name].metrics.rmsd
    worst = results[worst_rmsd_name].metrics.rmsd
    improvement = (1 - best / worst) * 100
    print(f'  Best RMSD:  {best_rmsd_name} ({best:.4f} mm)')
    print(f'  Worst RMSD: {worst_rmsd_name} ({worst:.4f} mm)')
    print(f'  Improvement: {improvement:.1f}%')

    # Bias direction
    for name in names:
        msd = results[name].metrics.msd
        if msd > 0.01:
            print(f'  {name}: systematic OVER-extrusion (MSD = +{msd:.4f} mm)')
        elif msd < -0.01:
            print(f'  {name}: systematic UNDER-extrusion (MSD = {msd:.4f} mm)')
        else:
            print(f'  {name}: minimal bias (MSD = {msd:.4f} mm)')

    # Volume comparison
    for name in names:
        m = results[name].metrics
        total_vol_error = m.excess_volume_mm3 + m.deficit_volume_mm3
        ratio = m.excess_volume_mm3 / max(m.deficit_volume_mm3, 1e-12)
        print(f'  {name}: total volumetric error = {total_vol_error:.4f} mm³ (excess/deficit ratio = {ratio:.2f})')

    print()


def print_ks_results(ks: dict[tuple[str, str], tuple[float, float]]) -> None:
    """Print pairwise KS test results."""
    print('=' * 72)
    print('  PAIRWISE KOLMOGOROV-SMIRNOV TESTS')
    print('=' * 72)
    print(f'  {"Method A":<20} {"Method B":<20} {"D-stat":>8} {"p-value":>12}')
    print('  ' + '-' * 64)
    for (a, b), (stat, pval) in ks.items():
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        print(f'  {a:<20} {b:<20} {stat:>8.4f} {pval:>12.2e} {sig}')
    print('  (* p<0.05, ** p<0.01, *** p<0.001)\n')


def generate_figures(results: dict[str, MethodResult]) -> None:
    """Generate and save all comparison figures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('Generating figures...')

    # Collect data for multi-method plots
    all_distances = {name: r.signed_distances for name, r in results.items()}
    all_metrics = {name: r.metrics for name, r in results.items()}

    # KDE comparison
    plot_deviation_kde(all_distances, VIS_CONFIG, output_path=OUTPUT_DIR / 'kde_comparison.png')
    print(f'  -> {OUTPUT_DIR / "kde_comparison.png"}')

    # Boxplot
    plot_boxplot_comparison(all_distances, VIS_CONFIG, output_path=OUTPUT_DIR / 'boxplot_comparison.png')
    print(f'  -> {OUTPUT_DIR / "boxplot_comparison.png"}')

    # Metrics table (PNG)
    plot_metrics_summary_table(all_metrics, VIS_CONFIG, output_path=OUTPUT_DIR / 'metrics_table.png')
    print(f'  -> {OUTPUT_DIR / "metrics_table.png"}')

    # Metrics CSV
    export_metrics_csv(all_metrics, output_path=OUTPUT_DIR / 'metrics.csv')
    print(f'  -> {OUTPUT_DIR / "metrics.csv"}')

    # Per-method spatial deviation maps
    for name, r in results.items():
        safe_name = r.scan_csv.replace('.csv', '')
        try:
            plot_spatial_deviation_map(
                cad_mesh_path=STL_FILE,
                scan_points_aligned=r.aligned_bead,
                signed_distances=r.signed_distances,
                method_name=name,
                config=VIS_CONFIG,
                output_path=OUTPUT_DIR / f'{safe_name}_deviation_map.png',
            )
            print(f'  -> {OUTPUT_DIR / safe_name}_deviation_map.png')
        except Exception as e:
            print(f'  !! Deviation map for {name} failed: {e}')

    print(f'\nAll outputs in: {OUTPUT_DIR}/')


def save_scan_stls(results: dict[str, MethodResult]) -> None:
    """
    Save bead point clouds as STL meshes for each method.

    Saves two STL files per method:
    - *_bead_scan.stl:    bead points in the flattened/segmented frame
    - *_bead_aligned.stl: bead points registered to the CAD frame
    """
    STL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('Saving scan STL meshes...')

    for name, r in results.items():
        print(f'  {name}:')
        safe_name = r.scan_csv.replace('.csv', '')

        # Bead in scan frame (post-segmentation, pre-registration)
        path_scan = save_points_as_stl(
            r.bead_points,
            STL_OUTPUT_DIR / f'{safe_name}_bead_scan.stl',
            max_edge_length=0.5,
        )
        print(f'  -> {path_scan}  ({len(r.bead_points):,} pts)')

        # Bead aligned to CAD frame (post-registration)
        path_aligned = save_points_as_stl(
            r.aligned_bead,
            STL_OUTPUT_DIR / f'{safe_name}_bead_aligned.stl',
            max_edge_length=0.5,
        )
        print(f'  -> {path_aligned}  ({len(r.aligned_bead):,} pts)')

    print(f'\nSTL files in: {STL_OUTPUT_DIR}/')


# ============================================================================
#  MAIN
# ============================================================================

if __name__ == '__main__':
    import trimesh

    # Load CAD mesh once (shared across all methods)
    print('Loading CAD mesh...')
    cad_mesh_o3d = load_cad_mesh(STL_FILE)
    cad_mesh_trimesh = trimesh.load(str(STL_FILE))
    print(
        f'  {len(np.asarray(cad_mesh_o3d.vertices)):,} vertices, {len(np.asarray(cad_mesh_o3d.triangles)):,} triangles'
    )

    # Process each method
    results: dict[str, MethodResult] = {}
    for display_name, scan_csv, toolpath_csv in METHODS:
        r = process_one_method(
            display_name,
            scan_csv,
            toolpath_csv,
            cad_mesh_o3d,
            cad_mesh_trimesh,
        )
        results[display_name] = r

    # Print comparison
    print_comparison(results)

    # KS tests
    all_distances = {name: r.signed_distances for name, r in results.items()}
    ks_results = compute_pairwise_ks(all_distances)
    print_ks_results(ks_results)

    # Generate figures
    generate_figures(results)

    # Save bead scans as STL meshes
    save_scan_stls(results)

    # -----------------------------------------------------------------
    # At this point, everything is available for interactive exploration:
    #
    #   results["VBN (v5)"]                  -> MethodResult dataclass
    #   results["VBN (v5)"].metrics          -> DeviationMetrics
    #   results["VBN (v5)"].signed_distances -> np.ndarray
    #   results["VBN (v5)"].aligned_bead     -> np.ndarray (M, 3)
    #   results["VBN (v5)"].bead_points      -> np.ndarray (M, 3) pre-registration
    #   results["VBN (v5)"].flat_points      -> np.ndarray (P, 3) full flattened cloud
    #   cad_mesh_o3d                         -> open3d TriangleMesh
    #   cad_mesh_trimesh                     -> trimesh.Trimesh
    #   ks_results                           -> dict of (stat, pvalue) per pair
    # -----------------------------------------------------------------
    print("Done. All results are in the 'results' dict for interactive use.")
