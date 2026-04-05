"""Batch processing across multiple methods and scan cycles.

Orchestrates the full pipeline: load -> register -> compute deviations ->
aggregate metrics -> generate visualizations and summary outputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from tqdm import tqdm

from deviation_analysis.cache import cache_key, load_cache, save_cache
from deviation_analysis.config import MethodSpec, PipelineConfig
from deviation_analysis.deviation import (
    DeviationMetrics,
    compute_metrics,
    compute_pairwise_ks,
    compute_signed_distances,
)
from deviation_analysis.loader import (
    build_sidewalled_mesh,
    height_correct,
    load_cad_mesh,
    load_scan_csv,
    load_toolpath_csv,
    points_to_open3d,
    raster_to_point_cloud,
    save_points_as_stl,
)
from deviation_analysis.smoothing import add_sidewalls, remove_islands, smooth_anisotropic_grid
from deviation_analysis.registration import (
    flatten_point_cloud,
    register_scan_to_cad,
    segment_bed_from_bead,
)
from deviation_analysis.visualization import (
    export_metrics_csv,
    plot_boxplot_comparison,
    plot_deviation_kde,
    plot_metrics_summary_table,
    plot_spatial_deviation_map,
)


def process_single_scan(
    scan_csv: Path,
    toolpath_csv: Path,
    stl_path: Path,
    config: PipelineConfig,
) -> tuple[np.ndarray, DeviationMetrics]:
    """Run the full pipeline for one scan CSV.

    Steps:
    1. Check cache for each stage; skip if hit
    2. Load scan CSV -> height correct -> raster to point cloud
    3. Flatten point cloud (RANSAC plane)
    4. Segment bed from bead
    5. Register bead to CAD mesh (FPFH + ICP)
    6. Compute signed distances
    7. Compute metrics
    8. Save intermediate results to cache

    Parameters
    ----------
    scan_csv
        Path to the scan cycle CSV file.
    toolpath_csv
        Path to the scan trajectory CSV file.
    stl_path
        Path to the CAD reference STL file.
    config
        Pipeline configuration.

    Returns
    -------
    signed_distances : np.ndarray
        (M,) signed distance array for the bead points.
    metrics : DeviationMetrics
        Summary metrics for this scan.
    """
    cache_dir = config.cache_dir
    reg_config = config.registration
    point_area = config.scan.resolution * config.scan.slice_thickness

    # Smoothing state affects registration and distance results, so encode
    # it in the cache suffix to avoid stale hits when toggling use_smoothed.
    _sc = config.smoothing
    _ri = (
        f"ri{_sc.island_closing_radius}d{_sc.island_min_distance}"
        if _sc.remove_islands else "nori"
    )
    _sw = "sw" if _sc.add_sidewalls else "nosw"
    _smooth_tag = (
        f"_sm{_sc.sigma_scan}_{_sc.sigma_perp}_{_sc.n_iterations}_{_ri}_{_sw}"
        if _sc.enabled and _sc.use_smoothed
        else "_raw"
    )

    # --- Check for final cached result ---
    dist_key = cache_key(scan_csv, f"distances{_smooth_tag}")
    cached = load_cache(cache_dir, dist_key, source_path=scan_csv)
    if cached is not None:
        signed_distances = cached["signed_distances"]
        metrics = compute_metrics(signed_distances, point_area, config.deviation)
        return signed_distances, metrics

    # --- Stage 1: Load and preprocess ---
    data_mm = load_scan_csv(scan_csv, config.scan)
    _time, toolpath_xyz = load_toolpath_csv(toolpath_csv)
    data_corrected = height_correct(data_mm, toolpath_xyz)
    points, valid_mask = raster_to_point_cloud(data_corrected, toolpath_xyz, config.scan)

    # --- Stage 2: Flatten (RANSAC plane) ---
    flat_key = cache_key(scan_csv, "flatten")
    cached = load_cache(cache_dir, flat_key, source_path=scan_csv)
    if cached is not None:
        flat_points = cached["flat_points"]
        valid_mask = cached["valid_mask"].astype(bool)
    else:
        flat_points, rotation, intercept = flatten_point_cloud(points, valid_mask, random_state=reg_config.random_seed)
        save_cache(
            cache_dir, flat_key,
            flat_points=flat_points,
            valid_mask=valid_mask,
            rotation=rotation,
            intercept=np.array([intercept]),
        )

    # --- Stage 3: Segment bed from bead ---
    bead_points, _bed_points, bead_mask = segment_bed_from_bead(
        flat_points, valid_mask, reg_config.bed_z_threshold
    )

    # --- Stage 3.5: Anisotropic smoothing ---
    smooth_cfg = config.smoothing
    raw_bead_points = bead_points

    if smooth_cfg.enabled:
        sc = smooth_cfg
        smooth_key = cache_key(
            scan_csv,
            f"smooth_s{sc.sigma_scan}_p{sc.sigma_perp}_n{sc.n_iterations}",
        )
        cached_smooth = load_cache(cache_dir, smooth_key, source_path=scan_csv)
        if cached_smooth is not None:
            smoothed_bead_points = cached_smooth["smoothed_bead_points"]
        else:
            n_rows, n_cols = data_corrected.shape
            smoothed_bead_points = smooth_anisotropic_grid(
                flat_points=flat_points,
                valid_mask=valid_mask,
                bead_mask=bead_mask,
                n_rows=n_rows,
                n_cols=n_cols,
                sigma_scan=sc.sigma_scan,
                sigma_perp=sc.sigma_perp,
                scan_direction=np.array(sc.scan_direction),
                x_spacing=config.scan.resolution,
                y_spacing=config.scan.slice_thickness,
                n_iterations=sc.n_iterations,
            )
            save_cache(
                cache_dir, smooth_key,
                smoothed_bead_points=smoothed_bead_points,
            )

        # Export raw + smoothed STL meshes for visual comparison (skip if exists)
        out = config.output_dir
        out.mkdir(parents=True, exist_ok=True)
        stem = scan_csv.stem
        raw_stl = out / f"{stem}_raw_bead.stl"
        sm_stl = out / f"{stem}_smoothed_bead.stl"
        if not raw_stl.exists():
            save_points_as_stl(raw_bead_points, raw_stl, max_edge_length=0.5)
        if not sm_stl.exists():
            save_points_as_stl(smoothed_bead_points, sm_stl, max_edge_length=0.5)

        # Select which points downstream stages use
        if smooth_cfg.use_smoothed:
            bead_points = smoothed_bead_points

    # --- Stage 3.6: Remove small islands ---
    if smooth_cfg.enabled and smooth_cfg.remove_islands:
        n_rows, n_cols = data_corrected.shape
        bead_points, bead_mask = remove_islands(
            bead_points, bead_mask, n_rows, n_cols,
            closing_radius=smooth_cfg.island_closing_radius,
            min_distance=smooth_cfg.island_min_distance,
            x_spacing=config.scan.resolution,
            y_spacing=config.scan.slice_thickness,
        )

    # --- Stage 3.7: Add sidewalls ---
    if smooth_cfg.enabled and smooth_cfg.add_sidewalls:
        n_rows, n_cols = data_corrected.shape
        bead_points_surface = bead_points  # M-point surface, before sidewall augmentation

        bead_points = add_sidewalls(
            bead_points,
            bead_mask=bead_mask,
            n_rows=n_rows,
            n_cols=n_cols,
            flat_points=flat_points,
            z_step=smooth_cfg.sidewall_z_step,
            floor_z=0.0,
        )

        # Export sidewalled mesh as STL (explicit top + sidewalls + floor).
        # Include _smooth_tag in filename so raw/smoothed passes don't collide.
        sw_stl = out / f"{stem}_sidewalled_bead{_smooth_tag}.stl"
        if not sw_stl.exists():
            build_sidewalled_mesh(
                bead_points=bead_points_surface,
                bead_mask=bead_mask,
                n_rows=n_rows,
                n_cols=n_cols,
                output_path=sw_stl,
                floor_z=0.0,
                max_edge_length=0.5,
            )

    # --- Stage 4: Register bead to CAD mesh ---
    reg_key = cache_key(scan_csv, f"register{_smooth_tag}")
    cached = load_cache(cache_dir, reg_key, source_path=scan_csv)
    if cached is not None:
        transform = cached["transform"]
    else:
        cad_mesh_o3d = load_cad_mesh(stl_path)
        transform, fitness, inlier_rmse = register_scan_to_cad(
            bead_points, cad_mesh_o3d, reg_config
        )
        save_cache(
            cache_dir, reg_key,
            transform=transform,
            fitness=np.array([fitness]),
            inlier_rmse=np.array([inlier_rmse]),
        )

    # --- Stage 5: Transform bead points and compute signed distances ---
    # Apply 4x4 transform to bead points
    bead_homogeneous = np.hstack([bead_points, np.ones((len(bead_points), 1))])
    aligned_bead = (transform @ bead_homogeneous.T).T[:, :3]

    # Load CAD mesh as trimesh for distance queries
    cad_mesh_tri = trimesh.load(str(stl_path))
    signed_distances = compute_signed_distances(aligned_bead, cad_mesh_tri)

    save_cache(
        cache_dir, dist_key,
        signed_distances=signed_distances,
        aligned_bead_points=aligned_bead,
    )

    # --- Stage 6: Compute metrics ---
    metrics = compute_metrics(signed_distances, point_area, config.deviation)

    return signed_distances, metrics


def process_method(
    method: MethodSpec,
    stl_path: Path,
    config: PipelineConfig,
) -> dict[str, tuple[np.ndarray, DeviationMetrics]]:
    """Process all selected cycles for one experimental method.

    Parameters
    ----------
    method
        Method specification with cycle CSV filenames.
    stl_path
        Path to the CAD reference STL.
    config
        Pipeline configuration.

    Returns
    -------
    dict
        Mapping from cycle CSV filename to (signed_distances, metrics).
    """
    results: dict[str, tuple[np.ndarray, DeviationMetrics]] = {}
    toolpath_path = config.scan_path(method.toolpath_csv)

    for cycle_csv in method.cycle_csvs:
        scan_path = config.scan_path(cycle_csv)
        distances, metrics = process_single_scan(
            scan_csv=scan_path,
            toolpath_csv=toolpath_path,
            stl_path=stl_path,
            config=config,
        )
        results[cycle_csv] = (distances, metrics)

    return results


def run_batch(config: PipelineConfig) -> dict[str, dict]:
    """Run the full batch analysis across all configured methods.

    1. Process each method's selected cycles
    2. Select first cycle per method as exemplar for visualizations
    3. Compute pairwise KS tests between methods
    4. Generate all visualizations (deviation maps, KDE, boxplots, tables)
    5. Export summary CSV

    Parameters
    ----------
    config
        Pipeline configuration with methods list populated.

    Returns
    -------
    dict
        Nested results: {method_name: {cycle_name: (distances, metrics), ...}, ...}
    """
    if not config.methods:
        raise ValueError(
            "No methods configured. Populate config.methods before calling run_batch()."
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    stl_path = config.stl_path
    all_results: dict[str, dict[str, tuple[np.ndarray, DeviationMetrics]]] = {}

    # --- Process each method ---
    for method in tqdm(config.methods, desc="Processing methods"):
        all_results[method.name] = process_method(method, stl_path, config)

    # --- Collect exemplar data per method (first cycle) for comparison ---
    exemplar_distances: dict[str, np.ndarray] = {}
    exemplar_metrics: dict[str, DeviationMetrics] = {}
    exemplar_aligned_points: dict[str, np.ndarray] = {}

    for method in config.methods:
        cycle_results = all_results[method.name]
        # Use first cycle as exemplar
        first_cycle = method.cycle_csvs[0]
        distances, metrics = cycle_results[first_cycle]
        exemplar_distances[method.display_name] = distances
        exemplar_metrics[method.display_name] = metrics

        # Load aligned bead points from cache for deviation map
        _sc = config.smoothing
        _smooth_tag = (
            f"_sm{_sc.sigma_scan}_{_sc.sigma_perp}_{_sc.n_iterations}"
            if _sc.enabled and _sc.use_smoothed
            else "_raw"
        )
        dist_key = cache_key(config.scan_path(first_cycle), f"distances{_smooth_tag}")
        cached = load_cache(config.cache_dir, dist_key)
        if cached is not None and "aligned_bead_points" in cached:
            exemplar_aligned_points[method.display_name] = cached["aligned_bead_points"]

    # --- Pairwise KS tests ---
    ks_results = compute_pairwise_ks(exemplar_distances)
    print("\nPairwise KS tests:")
    for (a, b), (stat, pval) in ks_results.items():
        print(f"  {a} vs {b}: D={stat:.4f}, p={pval:.2e}")

    # --- Generate visualizations ---
    # Spatial deviation maps
    for method in config.methods:
        display = method.display_name
        if display in exemplar_aligned_points:
            plot_spatial_deviation_map(
                cad_mesh_path=stl_path,
                scan_points_aligned=exemplar_aligned_points[display],
                signed_distances=exemplar_distances[display],
                method_name=display,
                config=config.visualization,
                output_path=config.output_dir / f"{method.name}_deviation_map.png",
            )

    # KDE comparison
    plot_deviation_kde(
        exemplar_distances,
        config=config.visualization,
        output_path=config.output_dir / "kde_comparison.png",
    )

    # Boxplot comparison
    plot_boxplot_comparison(
        exemplar_distances,
        config=config.visualization,
        output_path=config.output_dir / "boxplot_comparison.png",
    )

    # Metrics summary table
    plot_metrics_summary_table(
        exemplar_metrics,
        config=config.visualization,
        output_path=config.output_dir / "metrics_table.png",
    )

    # CSV export
    export_metrics_csv(
        exemplar_metrics,
        output_path=config.output_dir / "metrics_summary.csv",
    )

    # --- Print summary ---
    print("\nMetrics Summary:")
    print(f"{'Method':<20} {'RMSD':>8} {'MSD':>8} {'MAE':>8} {'95th%':>8}")
    print("-" * 56)
    for name, m in exemplar_metrics.items():
        print(f"{name:<20} {m.rmsd:>8.4f} {m.msd:>8.4f} {m.mae:>8.4f} {m.percentile_abs:>8.4f}")

    print(f"\nOutputs saved to: {config.output_dir}")

    return all_results
