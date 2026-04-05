"""Step-by-step test script for the deviation analysis pipeline.

Run from the src/ directory:
    python test_pipeline.py

Each stage prints diagnostics and optionally shows visualizations so you can
verify correctness before moving on.  Stages are numbered and can be toggled
on/off with the flags at the top of the script.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Configuration — toggle individual stages on/off
# ---------------------------------------------------------------------------
RUN_STAGE_1_LOAD = True  # Load scan CSV + toolpath
RUN_STAGE_2_HEIGHT_CORRECT = True  # Height correction
RUN_STAGE_3_POINT_CLOUD = True  # Raster -> point cloud conversion
RUN_STAGE_4_FLATTEN = True  # RANSAC plane flatten
RUN_STAGE_5_SEGMENT = True  # Bed / bead segmentation
RUN_STAGE_5B_SMOOTH = True  # Smoothing + island removal + sidewalls
RUN_STAGE_6_CAD_MESH = True  # Load CAD STL mesh
RUN_STAGE_7_REGISTER = True  # FPFH + ICP registration
RUN_STAGE_8_DISTANCES = True  # Signed distance computation
RUN_STAGE_9_METRICS = True  # Metric extraction
RUN_STAGE_10_VISUALIZE = True  # All visualization outputs

SHOW_PLOTS = True  # Set False to only save PNGs and skip interactive display

# ---------------------------------------------------------------------------
# Paths — adjust if your working directory is different
# ---------------------------------------------------------------------------

# m_static_naive use cycle 1
# m_static_ideal use cycle 1
# m_VBN_2 use cycle 2
# m_VBN_05 use cycle 1


DATA_DIR = Path('demos/m')
SCAN_CSV = DATA_DIR / 'm_VBN_2_cycle_002.csv'
TOOLPATH_CSV = DATA_DIR / 'm_VBN_2.csv'
STL_FILE = DATA_DIR / 'm_ideal.stl'
OUTPUT_DIR = Path('deviation_analysis/output/test')
CACHE_DIR = Path('deviation_analysis/cache/test')

# Map toolpath stem to display name (must match compare_methods.py METHODS list)
METHOD_NAMES = {
    'm_VBN_05': 'VBN (v5)',
    'm_VBN_2': 'VBN (v2)',
    'm_static_ideal': 'Static ideal',
    'm_static_naive': 'Static naive',
}
METHOD_NAME = METHOD_NAMES.get(TOOLPATH_CSV.stem, TOOLPATH_CSV.stem)


def _separator(stage: int | str, title: str) -> None:
    print(f'\n{"=" * 60}')
    print(f'  STAGE {stage}: {title}')
    print(f'{"=" * 60}')


def _show_or_save(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f'{name}.png', dpi=150, bbox_inches='tight')
    print(f'  -> Saved {OUTPUT_DIR / name}.png')
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


# ===== STAGE 1: Load raw data ==============================================
if RUN_STAGE_1_LOAD:
    _separator(1, 'Load scan CSV + toolpath')

    from deviation_analysis.config import ScanConfig
    from deviation_analysis.loader import load_scan_csv, load_toolpath_csv

    scan_config = ScanConfig()  # defaults: resolution=0.02, scan_speed=5.0, etc.

    print(f'  Scan CSV:     {SCAN_CSV}')
    print(f'  Toolpath CSV: {TOOLPATH_CSV}')

    data_mm = load_scan_csv(SCAN_CSV, scan_config)
    time_arr, toolpath_xyz = load_toolpath_csv(TOOLPATH_CSV)

    print(f'  Scan data shape:     {data_mm.shape}  (rows x cols)')
    print(f'  Toolpath shape:      {toolpath_xyz.shape}')
    print(f'  Toolpath time range: {time_arr[0]:.2f} – {time_arr[-1]:.2f} s')
    print('  Toolpath XYZ ranges:')
    for ax, label in enumerate(['X', 'Y', 'Z']):
        print(f'    {label}: [{toolpath_xyz[:, ax].min():.3f}, {toolpath_xyz[:, ax].max():.3f}] mm')

    valid_count = np.count_nonzero(~np.isnan(data_mm))
    total = data_mm.size
    print(f'  Valid scan points: {valid_count:,} / {total:,} ({100 * valid_count / total:.1f}%)')
    print(f'  Height range (valid): [{np.nanmin(data_mm):.3f}, {np.nanmax(data_mm):.3f}] mm')

    # Quick heatmap of raw scan
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(data_mm, aspect='auto', cmap='viridis', origin='lower')
    ax.set_title('Stage 1: Raw scan raster (mm)')
    ax.set_xlabel('Column index')
    ax.set_ylabel('Row index')
    plt.colorbar(im, ax=ax, label='Height (mm)')
    _show_or_save(fig, 'stage1_raw_raster')
else:
    print('\n  [Stage 1 skipped]')


# ===== STAGE 2: Height correction ==========================================
if RUN_STAGE_2_HEIGHT_CORRECT:
    _separator(2, 'Height correction')

    from deviation_analysis.loader import height_correct

    data_corrected = height_correct(data_mm, toolpath_xyz)
    print(f'  Corrected height range: [{np.nanmin(data_corrected):.3f}, {np.nanmax(data_corrected):.3f}] mm')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(data_mm, aspect='auto', cmap='viridis', origin='lower')
    axes[0].set_title('Before height correction')
    im = axes[1].imshow(data_corrected, aspect='auto', cmap='viridis', origin='lower')
    axes[1].set_title('After height correction')
    for a in axes:
        a.set_xlabel('Column index')
        a.set_ylabel('Row index')
    plt.colorbar(im, ax=axes, label='Height (mm)')
    _show_or_save(fig, 'stage2_height_correction')
else:
    print('\n  [Stage 2 skipped]')


# ===== STAGE 3: Raster -> point cloud ======================================
if RUN_STAGE_3_POINT_CLOUD:
    _separator(3, 'Raster -> point cloud')

    from deviation_analysis.loader import raster_to_point_cloud

    points, valid_mask = raster_to_point_cloud(data_corrected, toolpath_xyz, scan_config)
    valid_points = points[valid_mask]

    print(f'  Total points:  {len(points):,}')
    print(f'  Valid points:  {len(valid_points):,}')
    print('  Point cloud XYZ ranges:')
    for ax, label in enumerate(['X', 'Y', 'Z']):
        print(f'    {label}: [{valid_points[:, ax].min():.3f}, {valid_points[:, ax].max():.3f}] mm')
    print(
        f'  Point area (res x slice_thickness): '
        f'{scan_config.resolution} x {scan_config.slice_thickness:.4f} '
        f'= {scan_config.resolution * scan_config.slice_thickness:.6f} mm²'
    )

    # 3D scatter of subsampled points
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    # Subsample for plotting performance
    step = max(1, len(valid_points) // 50_000)
    sub = valid_points[::step]
    sc = ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], c=sub[:, 2], cmap='viridis', s=0.1)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'Stage 3: Point cloud ({len(sub):,} of {len(valid_points):,} shown)')
    plt.colorbar(sc, ax=ax, label='Z (mm)', shrink=0.6)
    _show_or_save(fig, 'stage3_point_cloud')
else:
    print('\n  [Stage 3 skipped]')


# ===== STAGE 4: RANSAC plane flatten =======================================
if RUN_STAGE_4_FLATTEN:
    _separator(4, 'RANSAC plane flatten')

    from deviation_analysis.registration import flatten_point_cloud

    flat_points, rotation, intercept = flatten_point_cloud(points, valid_mask)
    flat_valid = flat_points[valid_mask]

    print(f'  Rotation matrix:\n{rotation}')
    print(f'  Plane intercept: {intercept:.4f} mm')
    print(f'  Flattened Z range (valid): [{flat_valid[:, 2].min():.4f}, {flat_valid[:, 2].max():.4f}] mm')

    # Before/after Z histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(valid_points[:, 2], bins=200, color='steelblue', edgecolor='none')
    axes[0].set_title('Before flatten: Z distribution')
    axes[0].set_xlabel('Z (mm)')
    axes[1].hist(flat_valid[:, 2], bins=200, color='coral', edgecolor='none')
    axes[1].set_title('After flatten: Z distribution')
    axes[1].set_xlabel('Z (mm)')
    for a in axes:
        a.set_ylabel('Count')
    _show_or_save(fig, 'stage4_flatten_z_hist')

    # Top-down view colored by Z
    fig, ax = plt.subplots(figsize=(10, 6))
    step = max(1, len(flat_valid) // 50_000)
    sub = flat_valid[::step]
    sc = ax.scatter(sub[:, 0], sub[:, 1], c=sub[:, 2], cmap='viridis', s=0.1)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Stage 4: Flattened point cloud (top-down, colored by Z)')
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, label='Z (mm)')
    _show_or_save(fig, 'stage4_flatten_topdown')
else:
    print('\n  [Stage 4 skipped]')


# ===== STAGE 5: Bed / bead segmentation ====================================
if RUN_STAGE_5_SEGMENT:
    _separator(5, 'Bed / bead segmentation')

    from deviation_analysis.config import RegistrationConfig
    from deviation_analysis.registration import segment_bed_from_bead

    reg_config = RegistrationConfig()
    bead_points, bed_points, bead_mask = segment_bed_from_bead(flat_points, valid_mask, reg_config.bed_z_threshold)

    print(f'  Z threshold:  {reg_config.bed_z_threshold} mm')
    print(f'  Bead points:  {len(bead_points):,}')
    print(f'  Bed points:   {len(bed_points):,}')
    print(f'  Bead Z range: [{bead_points[:, 2].min():.4f}, {bead_points[:, 2].max():.4f}] mm')
    print(f'  Bed Z range:  [{bed_points[:, 2].min():.4f}, {bed_points[:, 2].max():.4f}] mm')

    # Top-down view: bed vs bead
    fig, ax = plt.subplots(figsize=(10, 6))
    step_bed = max(1, len(bed_points) // 30_000)
    step_bead = max(1, len(bead_points) // 30_000)
    ax.scatter(bed_points[::step_bed, 0], bed_points[::step_bed, 1], c='lightgray', s=0.1, label='Bed', rasterized=True)
    ax.scatter(bead_points[::step_bead, 0], bead_points[::step_bead, 1], c='red', s=0.3, label='Bead', rasterized=True)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title(f'Stage 5: Bed/bead segmentation (threshold={reg_config.bed_z_threshold} mm)')
    ax.set_aspect('equal')
    ax.legend(markerscale=20)
    _show_or_save(fig, 'stage5_segmentation_topdown')

    # Side view (X-Z)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.scatter(bed_points[::step_bed, 0], bed_points[::step_bed, 2], c='lightgray', s=0.1, label='Bed', rasterized=True)
    ax.scatter(bead_points[::step_bead, 0], bead_points[::step_bead, 2], c='red', s=0.3, label='Bead', rasterized=True)
    ax.axhline(
        reg_config.bed_z_threshold,
        color='blue',
        linestyle='--',
        linewidth=1,
        label=f'Threshold = {reg_config.bed_z_threshold} mm',
    )
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_title('Stage 5: Side view (X-Z)')
    ax.legend(markerscale=20)
    _show_or_save(fig, 'stage5_segmentation_sideview')
    # Save bead points as STL
    print('\n  Saving bead points as STL...')
    import open3d as o3d

    from deviation_analysis.loader import save_points_as_stl

    stl_out = save_points_as_stl(
        bead_points,
        OUTPUT_DIR / 'stage5_bead_scan.stl',
        max_edge_length=0.5,
    )
    print(f'  -> Saved bead STL: {stl_out}')
    print(f'     Vertices: {len(bead_points):,}')

    # Verify it loads back
    reload_mesh = o3d.io.read_triangle_mesh(str(stl_out))
    n_verts = len(np.asarray(reload_mesh.vertices))
    n_tris = len(np.asarray(reload_mesh.triangles))
    print(f'     Reload check: {n_verts:,} vertices, {n_tris:,} triangles')

else:
    print('\n  [Stage 5 skipped]')


# ===== STAGE 5B: Smoothing + island removal + sidewalls ====================
if RUN_STAGE_5B_SMOOTH:
    _separator('5B', 'Smoothing + island removal + sidewalls')

    from deviation_analysis.config import SmoothingConfig
    from deviation_analysis.smoothing import (
        add_sidewalls,
        remove_islands,
        smooth_anisotropic_grid,
    )

    smooth_config = SmoothingConfig()  # uses defaults from config.py

    n_rows, n_cols = data_mm.shape
    bead_points_raw = bead_points.copy()  # keep raw for comparison plots

    if smooth_config.enabled:
        # --- Smoothing ---
        bead_points_smoothed = smooth_anisotropic_grid(
            flat_points=flat_points,
            valid_mask=valid_mask,
            bead_mask=bead_mask,
            n_rows=n_rows,
            n_cols=n_cols,
            sigma_scan=smooth_config.sigma_scan,
            sigma_perp=smooth_config.sigma_perp,
            scan_direction=np.array(smooth_config.scan_direction),
            x_spacing=scan_config.resolution,
            y_spacing=scan_config.slice_thickness,
            n_iterations=smooth_config.n_iterations,
        )
        print(f'  Smoothed: σ_scan={smooth_config.sigma_scan}, σ_perp={smooth_config.sigma_perp}')
        z_diff = np.abs(bead_points_smoothed[:, 2] - bead_points_raw[:, 2])
        print(f'  Z change: mean={z_diff.mean():.4f}, max={z_diff.max():.4f} mm')

        if smooth_config.use_smoothed:
            bead_points = bead_points_smoothed
            print('  Using smoothed points for downstream stages')
        else:
            print('  use_smoothed=False — keeping raw points for downstream stages')

        # --- Island removal ---
        bead_mask_pre_island = bead_mask.copy()  # save for STL re-run below
        if smooth_config.remove_islands:
            n_before = len(bead_points)
            bead_points, bead_mask = remove_islands(
                bead_points,
                bead_mask,
                n_rows,
                n_cols,
                closing_radius=smooth_config.island_closing_radius,
                min_distance=smooth_config.island_min_distance,
                x_spacing=scan_config.resolution,
                y_spacing=scan_config.slice_thickness,
            )
            n_removed = n_before - len(bead_points)
            print(f'  Island removal: {n_removed:,} points removed, {len(bead_points):,} remaining')

        # --- Sidewalls ---
        if smooth_config.add_sidewalls:
            n_surface = len(bead_points)
            bead_points = add_sidewalls(
                bead_points,
                bead_mask=bead_mask,
                n_rows=n_rows,
                n_cols=n_cols,
                flat_points=flat_points,
                z_step=smooth_config.sidewall_z_step,
                floor_z=0.0,
            )
            n_wall = len(bead_points) - n_surface
            print(f'  Sidewalls: +{n_wall:,} points ({len(bead_points):,} total)')

        # Save smoothed+walled STL
        from deviation_analysis.loader import build_sidewalled_mesh, save_points_as_stl

        stl_sm = save_points_as_stl(
            bead_points_smoothed if smooth_config.use_smoothed else bead_points_raw,
            OUTPUT_DIR / 'stage5b_bead_smoothed.stl',
            max_edge_length=0.5,
        )
        print(f'  -> Saved smoothed STL: {stl_sm}')

        if smooth_config.add_sidewalls:
            stl_sw = OUTPUT_DIR / 'stage5b_bead_sidewalled.stl'
            # Use surface points (pre-sidewall augmentation) for mesh construction
            surface_pts = bead_points_smoothed if smooth_config.use_smoothed else bead_points_raw
            # Re-apply island removal to surface points if needed
            if smooth_config.remove_islands:
                surface_pts_clean, bead_mask_for_mesh = remove_islands(
                    surface_pts,
                    bead_mask_pre_island,
                    n_rows,
                    n_cols,
                    closing_radius=smooth_config.island_closing_radius,
                    min_distance=smooth_config.island_min_distance,
                    x_spacing=scan_config.resolution,
                    y_spacing=scan_config.slice_thickness,
                )
            else:
                surface_pts_clean = surface_pts
                bead_mask_for_mesh = bead_mask
            build_sidewalled_mesh(
                bead_points=surface_pts_clean,
                bead_mask=bead_mask_for_mesh,
                n_rows=n_rows,
                n_cols=n_cols,
                output_path=stl_sw,
                floor_z=0.0,
                max_edge_length=0.5,
            )
            print(f'  -> Saved sidewalled STL: {stl_sw}')

        # Before/after Z comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(bead_points_raw[:, 2], bins=200, color='steelblue', edgecolor='none', density=True)
        axes[0].set_title('Before smoothing: bead Z')
        axes[0].set_xlabel('Z (mm)')

        z_plot = bead_points_smoothed[:, 2] if smooth_config.use_smoothed else bead_points_raw[:, 2]
        axes[1].hist(z_plot, bins=200, color='coral', edgecolor='none', density=True)
        axes[1].set_title('After smoothing: bead Z')
        axes[1].set_xlabel('Z (mm)')

        for a in axes:
            a.set_ylabel('Density')
        fig.suptitle(f'Stage 5B: Smoothing (σ_scan={smooth_config.sigma_scan}, σ_perp={smooth_config.sigma_perp})')
        _show_or_save(fig, 'stage5b_smoothing_z_hist')
    else:
        print('  Smoothing disabled (smooth_config.enabled=False)')
else:
    print('\n  [Stage 5B skipped]')


# ===== STAGE 6: Load CAD mesh ==============================================
if RUN_STAGE_6_CAD_MESH:
    _separator(6, 'Load CAD mesh (STL)')

    import open3d as o3d

    from deviation_analysis.loader import load_cad_mesh

    cad_mesh = load_cad_mesh(STL_FILE)
    vertices = np.asarray(cad_mesh.vertices)
    triangles = np.asarray(cad_mesh.triangles)

    print(f'  STL file: {STL_FILE}')
    print(f'  Vertices:  {len(vertices):,}')
    print(f'  Triangles: {len(triangles):,}')
    print('  Vertex XYZ ranges:')
    for ax_i, label in enumerate(['X', 'Y', 'Z']):
        print(f'    {label}: [{vertices[:, ax_i].min():.3f}, {vertices[:, ax_i].max():.3f}] mm')

    # Sample points from mesh for visualization
    cad_pcd = cad_mesh.sample_points_uniformly(number_of_points=20_000)
    cad_pts = np.asarray(cad_pcd.points)

    # Top-down overlay: CAD vs bead points
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(cad_pts[:, 0], cad_pts[:, 1], c='blue', s=0.5, label='CAD mesh samples', alpha=0.5)
    step_bead = max(1, len(bead_points) // 20_000)
    ax.scatter(
        bead_points[::step_bead, 0],
        bead_points[::step_bead, 1],
        c='red',
        s=0.5,
        label='Bead points (pre-registration)',
        alpha=0.5,
    )
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Stage 6: CAD mesh vs bead points (BEFORE registration)')
    ax.set_aspect('equal')
    ax.legend(markerscale=10)
    _show_or_save(fig, 'stage6_cad_vs_bead_before_reg')
else:
    print('\n  [Stage 6 skipped]')


# ===== STAGE 7: Registration (FPFH + ICP) ==================================
if RUN_STAGE_7_REGISTER:
    _separator(7, 'Registration (FPFH global + ICP refine)')

    from deviation_analysis.registration import register_scan_to_cad

    print('  Running FPFH + RANSAC global registration...')
    print('  (This may take a minute for large point clouds)')

    transform, fitness, inlier_rmse = register_scan_to_cad(bead_points, cad_mesh, reg_config)

    print(f'  ICP fitness:     {fitness:.4f}')
    print(f'  ICP inlier RMSE: {inlier_rmse:.4f} mm')
    print(f'  Transform:\n{transform}')

    # Apply transform to bead points
    bead_hom = np.hstack([bead_points, np.ones((len(bead_points), 1))])
    aligned_bead = (transform @ bead_hom.T).T[:, :3]

    print('  Aligned bead XYZ ranges:')
    for ax_i, label in enumerate(['X', 'Y', 'Z']):
        print(f'    {label}: [{aligned_bead[:, ax_i].min():.3f}, {aligned_bead[:, ax_i].max():.3f}] mm')

    # Top-down overlay: CAD vs aligned bead
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    step_bead = max(1, len(bead_points) // 20_000)
    step_aligned = max(1, len(aligned_bead) // 20_000)

    # Before
    axes[0].scatter(cad_pts[:, 0], cad_pts[:, 1], c='blue', s=0.5, alpha=0.5, label='CAD')
    axes[0].scatter(
        bead_points[::step_bead, 0], bead_points[::step_bead, 1], c='red', s=0.5, alpha=0.5, label='Bead (before)'
    )
    axes[0].set_title('Before registration')
    axes[0].set_aspect('equal')
    axes[0].legend(markerscale=10)

    # After
    axes[1].scatter(cad_pts[:, 0], cad_pts[:, 1], c='blue', s=0.5, alpha=0.5, label='CAD')
    axes[1].scatter(
        aligned_bead[::step_aligned, 0],
        aligned_bead[::step_aligned, 1],
        c='red',
        s=0.5,
        alpha=0.5,
        label='Bead (after)',
    )
    axes[1].set_title('After registration')
    axes[1].set_aspect('equal')
    axes[1].legend(markerscale=10)

    for a in axes:
        a.set_xlabel('X (mm)')
        a.set_ylabel('Y (mm)')
    fig.suptitle(f'Stage 7: Registration (fitness={fitness:.4f}, RMSE={inlier_rmse:.4f} mm)')
    _show_or_save(fig, 'stage7_registration')
else:
    print('\n  [Stage 7 skipped]')


# ===== STAGE 8: Signed distances ===========================================
if RUN_STAGE_8_DISTANCES:
    _separator(8, 'Signed distance computation')

    import trimesh as tm

    from deviation_analysis.deviation import compute_signed_distances

    cad_trimesh = tm.load(str(STL_FILE))
    print(f'  Trimesh loaded: {len(cad_trimesh.vertices)} verts, {len(cad_trimesh.faces)} faces')
    print(f'  Computing signed distances for {len(aligned_bead):,} points...')

    signed_distances = compute_signed_distances(aligned_bead, cad_trimesh)

    print('  Signed distance stats:')
    print(f'    Min:    {signed_distances.min():.4f} mm')
    print(f'    Max:    {signed_distances.max():.4f} mm')
    print(f'    Mean:   {signed_distances.mean():.4f} mm')
    print(f'    Std:    {signed_distances.std():.4f} mm')
    print(f'    Median: {np.median(signed_distances):.4f} mm')
    n_pos = np.sum(signed_distances > 0)
    n_neg = np.sum(signed_distances < 0)
    print(f'    Positive (over-extrusion):  {n_pos:,} ({100 * n_pos / len(signed_distances):.1f}%)')
    print(f'    Negative (under-extrusion): {n_neg:,} ({100 * n_neg / len(signed_distances):.1f}%)')

    # Histogram of signed distances
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(signed_distances, bins=300, color='steelblue', edgecolor='none', density=True)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Signed deviation (mm)')
    ax.set_ylabel('Density')
    ax.set_title('Stage 8: Signed distance distribution')
    _show_or_save(fig, 'stage8_distance_histogram')

    # Top-down colored by deviation
    fig, ax = plt.subplots(figsize=(10, 6))
    step = max(1, len(aligned_bead) // 50_000)
    clim = max(abs(np.percentile(signed_distances, 5)), abs(np.percentile(signed_distances, 95)))
    sc = ax.scatter(
        aligned_bead[::step, 0],
        aligned_bead[::step, 1],
        c=signed_distances[::step],
        cmap='RdBu_r',
        vmin=-clim,
        vmax=clim,
        s=0.3,
        rasterized=True,
    )
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Stage 8: Spatial deviation (top-down)')
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, label='Deviation (mm)')
    _show_or_save(fig, 'stage8_spatial_deviation')
else:
    print('\n  [Stage 8 skipped]')


# ===== STAGE 9: Metrics ====================================================
if RUN_STAGE_9_METRICS:
    _separator(9, 'Metric extraction')

    from deviation_analysis.config import DeviationConfig
    from deviation_analysis.deviation import compute_metrics

    dev_config = DeviationConfig()
    point_area = scan_config.resolution * scan_config.slice_thickness

    metrics = compute_metrics(signed_distances, point_area, dev_config)

    print(f'  RMSD:             {metrics.rmsd:.6f} mm')
    print(f'  MSD:              {metrics.msd:.6f} mm')
    print(f'  MAE:              {metrics.mae:.6f} mm')
    print(f'  95th %ile (abs):  {metrics.percentile_abs:.6f} mm')
    print(f'  Max abs dev:      {metrics.max_abs_deviation:.6f} mm')
    print(f'  Excess volume:    {metrics.excess_volume_mm3:.6f} mm³')
    print(f'  Deficit volume:   {metrics.deficit_volume_mm3:.6f} mm³')
    print(f'  Points analyzed:  {metrics.n_points:,}')
    print(f'  Point area:       {point_area:.6f} mm²')
else:
    print('\n  [Stage 9 skipped]')


# ===== STAGE 10: Visualization outputs =====================================
if RUN_STAGE_10_VISUALIZE:
    _separator(10, 'Visualization outputs')

    from deviation_analysis.config import VisualizationConfig
    from deviation_analysis.visualization import (
        export_metrics_csv,
        plot_boxplot_comparison,
        plot_deviation_kde,
        plot_metrics_summary_table,
        plot_spatial_deviation_map,
    )

    vis_config = VisualizationConfig()

    method_name = METHOD_NAME
    method_distances = {method_name: signed_distances}
    method_metrics = {method_name: metrics}

    # 10a: Spatial deviation map (pyvista)
    print('  10a: Spatial deviation map...')
    try:
        plot_spatial_deviation_map(
            cad_mesh_path=STL_FILE,
            scan_points_aligned=aligned_bead,
            signed_distances=signed_distances,
            method_name=method_name,
            config=vis_config,
            output_path=OUTPUT_DIR / 'stage10a_deviation_map.png',
        )
        print(f'  -> Saved {OUTPUT_DIR / "stage10a_deviation_map.png"}')
    except Exception as e:
        print(f'  !! Deviation map failed: {e}')
        print("     (pyvista may need a display or 'xvfb-run' for off-screen rendering)")

    # 10b: KDE plot
    print('  10b: KDE distribution plot...')
    plot_deviation_kde(
        method_distances,
        config=vis_config,
        output_path=OUTPUT_DIR / 'stage10b_kde.png',
    )
    print(f'  -> Saved {OUTPUT_DIR / "stage10b_kde.png"}')

    # 10c: Boxplot
    print('  10c: Boxplot comparison...')
    plot_boxplot_comparison(
        method_distances,
        config=vis_config,
        output_path=OUTPUT_DIR / 'stage10c_boxplot.png',
    )
    print(f'  -> Saved {OUTPUT_DIR / "stage10c_boxplot.png"}')

    # 10d: Metrics table
    print('  10d: Metrics summary table...')
    plot_metrics_summary_table(
        method_metrics,
        config=vis_config,
        output_path=OUTPUT_DIR / 'stage10d_metrics_table.png',
    )
    print(f'  -> Saved {OUTPUT_DIR / "stage10d_metrics_table.png"}')

    # 10e: CSV export
    print('  10e: Metrics CSV export...')
    export_metrics_csv(
        method_metrics,
        output_path=OUTPUT_DIR / 'stage10e_metrics.csv',
    )
    print(f'  -> Saved {OUTPUT_DIR / "stage10e_metrics.csv"}')

else:
    print('\n  [Stage 10 skipped]')


# ===== CACHE TEST ===========================================================
_separator(11, 'Cache round-trip test')

from deviation_analysis.cache import cache_key, load_cache, save_cache

test_key = cache_key(SCAN_CSV, 'test')
test_array = np.array([1.0, 2.0, 3.0])
save_path = save_cache(CACHE_DIR, test_key, test_data=test_array)
loaded = load_cache(CACHE_DIR, test_key)

assert loaded is not None, 'Cache load returned None'
assert np.allclose(loaded['test_data'], test_array), 'Cache data mismatch'
print(f'  Cache key:    {test_key}')
print(f'  Saved to:     {save_path}')
print('  Round-trip:   PASS')

# Clean up test cache
save_path.unlink()
print('  Cleaned up test cache file')


# ===== DONE =================================================================
print(f'\n{"=" * 60}')
print('  ALL STAGES COMPLETE')
print(f'  Outputs saved to: {OUTPUT_DIR}')
print(f'{"=" * 60}\n')
