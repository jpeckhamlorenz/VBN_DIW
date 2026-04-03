"""Anisotropic scan-direction smoothing for bead point clouds.

Removes periodic ridges caused by oscillatory robot arm speed during
Keyence raster scanning.  For the raster-grid data produced by the
Keyence profiler the smoothing reduces to a **NaN-aware 2-D anisotropic
Gaussian convolution** on the height map — orders of magnitude faster than
per-point KD-tree queries while producing identical results for 2.5-D
single-layer geometry (surface normal ≈ +Z everywhere).

A general KD-tree + PCA-normal implementation is retained as a fallback
for arbitrary (non-grid) point clouds.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion
from scipy.spatial import cKDTree


# ── Fast grid-based path (used by batch.py) ──────────────────────────


def smooth_anisotropic_grid(
    flat_points: np.ndarray,
    valid_mask: np.ndarray,
    bead_mask: np.ndarray,
    n_rows: int,
    n_cols: int,
    sigma_scan: float,
    sigma_perp: float,
    scan_direction: np.ndarray,
    x_spacing: float,
    y_spacing: float,
    n_iterations: int = 1,
) -> np.ndarray:
    """Smooth bead Z values using 2-D anisotropic Gaussian convolution.

    Operates on the raster-grid structure of the scan data, which makes
    the convolution separable and O(N).

    Parameters
    ----------
    flat_points
        (N, 3) full point cloud (rows*cols points, including bed/invalid).
    valid_mask
        (N,) bool — True for points with valid height data.
    bead_mask
        (N,) bool — True for bead points (above bed threshold).
    n_rows, n_cols
        Raster grid dimensions (flat_points has ``n_rows * n_cols`` rows).
    sigma_scan
        Gaussian sigma along the scan direction (mm).
    sigma_perp
        Gaussian sigma perpendicular to the scan direction (mm).
    scan_direction
        (3,) unit vector of the scan direction.  Used to determine which
        grid axis corresponds to the scan direction.
    x_spacing
        Grid spacing along X (mm) — ``ScanConfig.resolution``.
    y_spacing
        Grid spacing along Y (mm) — ``ScanConfig.slice_thickness``.
    n_iterations
        Number of smoothing passes.

    Returns
    -------
    smoothed_bead_points : (M, 3)
        Smoothed bead-only point cloud (same order as
        ``flat_points[bead_mask]``).  X and Y are unchanged; only Z is
        modified.
    """
    scan_dir = np.asarray(scan_direction, dtype=np.float64)
    scan_dir = scan_dir / np.linalg.norm(scan_dir)

    # Determine which grid axis is the scan direction.
    # Grid rows = Y slices (axis 0), cols = X pixels (axis 1).
    # If scan_direction is predominantly Y → axis 0, else axis 1.
    if abs(scan_dir[1]) >= abs(scan_dir[0]):
        # Scan along Y → axis 0 is scan, axis 1 is perp
        sigma_axis0 = sigma_scan / y_spacing  # cells
        sigma_axis1 = sigma_perp / x_spacing
    else:
        # Scan along X → axis 1 is scan, axis 0 is perp
        sigma_axis0 = sigma_perp / y_spacing
        sigma_axis1 = sigma_scan / x_spacing

    # Build 2-D height map and mask
    z_all = flat_points[:, 2].copy()
    mask_2d = (valid_mask & bead_mask).reshape(n_rows, n_cols).astype(np.float64)
    z_2d = z_all.reshape(n_rows, n_cols).copy()
    z_2d[mask_2d == 0] = 0.0  # zero out non-bead cells for convolution

    for _ in range(n_iterations):
        # NaN-aware (normalised) convolution: convolve numerator and
        # denominator separately, then divide.
        z_smooth = gaussian_filter(z_2d * mask_2d, sigma=[sigma_axis0, sigma_axis1], mode="constant")
        w_smooth = gaussian_filter(mask_2d, sigma=[sigma_axis0, sigma_axis1], mode="constant")

        # Avoid division by zero at cells with no bead neighbours
        safe = w_smooth > 1e-12
        with np.errstate(invalid="ignore", divide="ignore"):
            z_2d_new = np.where(safe, z_smooth / w_smooth, z_2d)

        # Only update bead cells
        z_2d = np.where(mask_2d > 0.5, z_2d_new, z_2d)

    # Extract smoothed bead points
    bead_points = flat_points[bead_mask].copy()
    smoothed_z = z_2d.ravel()[bead_mask]
    bead_points[:, 2] = smoothed_z
    return bead_points


# ── Sidewall generation ──────────────────────────────────────────────


def add_sidewalls(
    bead_points: np.ndarray,
    bead_mask: np.ndarray,
    n_rows: int,
    n_cols: int,
    flat_points: np.ndarray,
    z_step: float = 0.1,
    floor_z: float = 0.0,
) -> np.ndarray:
    """Add vertical sidewall points from bead edges down to the floor.

    Finds the boundary pixels of the bead region in the raster grid,
    then for each boundary pixel creates a vertical column of points
    from that pixel's Z height down to *floor_z*.

    Parameters
    ----------
    bead_points
        (M, 3) bead surface point cloud (smoothed or raw).
    bead_mask
        (N,) bool mask over the full flattened grid (``n_rows * n_cols``).
    n_rows, n_cols
        Raster grid dimensions.
    flat_points
        (N, 3) full flattened point cloud (used to get XY positions of
        edge pixels; Z values come from *bead_points*).
    z_step
        Vertical spacing (mm) between successive sidewall points.
    floor_z
        Z level of the floor / bed plane (mm).

    Returns
    -------
    augmented : (M + S, 3)
        Original *bead_points* with sidewall points appended.  The first
        M rows are the original bead points (unchanged).
    """
    mask_2d = bead_mask.reshape(n_rows, n_cols)

    # Edge detection: bead pixels with at least one non-bead 4-neighbour
    eroded = binary_erosion(mask_2d, structure=np.array([[0, 1, 0],
                                                          [1, 1, 1],
                                                          [0, 1, 0]]))
    edge_2d = mask_2d & ~eroded  # boundary ring

    # Map from flattened grid index → bead_points row index
    # bead_points[k] corresponds to flat_points[bead_indices[k]]
    bead_indices = np.flatnonzero(bead_mask)
    # For edge pixels, find their position in bead_points
    edge_flat_indices = np.flatnonzero(edge_2d.ravel() & bead_mask)

    # Build lookup: flat_index → bead_points row
    flat_to_bead = np.full(n_rows * n_cols, -1, dtype=np.intp)
    flat_to_bead[bead_indices] = np.arange(len(bead_indices))

    sidewall_points = []
    for fi in edge_flat_indices:
        bi = flat_to_bead[fi]
        if bi < 0:
            continue
        x, y, z_top = bead_points[bi]
        if z_top <= floor_z + z_step:
            continue  # too close to floor, skip
        # Create column from just below z_top down to floor_z
        n_steps = max(1, int(np.ceil((z_top - floor_z) / z_step)))
        z_vals = np.linspace(z_top - z_step, floor_z, n_steps)
        col = np.empty((len(z_vals), 3), dtype=np.float64)
        col[:, 0] = x
        col[:, 1] = y
        col[:, 2] = z_vals
        sidewall_points.append(col)

    if sidewall_points:
        sidewall = np.vstack(sidewall_points)
        return np.vstack([bead_points, sidewall])
    return bead_points


# ── General KD-tree path (fallback for non-grid point clouds) ────────


def _estimate_normals(
    points: np.ndarray,
    tree: cKDTree,
    radius: float,
) -> np.ndarray:
    """Estimate surface normals via PCA of local neighbourhoods.

    Parameters
    ----------
    points
        (N, 3) point cloud.
    tree
        Pre-built KD-tree for *points*.
    radius
        Ball-query radius for neighbourhood selection.

    Returns
    -------
    normals : (N, 3)
        Unit normals oriented toward +Z (bead is above bed).
    """
    n = len(points)
    normals = np.empty((n, 3), dtype=np.float64)

    neighbour_lists = tree.query_ball_point(points, radius, workers=-1)

    for i, nbrs in enumerate(neighbour_lists):
        if len(nbrs) < 3:
            normals[i] = [0.0, 0.0, 1.0]
            continue

        local = points[nbrs]
        cov = np.cov(local, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]

        if normal[2] < 0:
            normal = -normal
        normals[i] = normal

    return normals


def smooth_anisotropic(
    points: np.ndarray,
    scan_direction: np.ndarray,
    sigma_scan: float,
    sigma_perp: float,
    n_iterations: int = 1,
) -> np.ndarray:
    """Apply anisotropic Gaussian smoothing along the scan direction.

    General-purpose implementation using KD-tree ball queries and PCA
    normal estimation.  Suitable for arbitrary (non-grid) point clouds
    but **slow** for large datasets (>100 K points).  Prefer
    :func:`smooth_anisotropic_grid` for raster scan data.

    Parameters
    ----------
    points
        (N, 3) bead point cloud (after bed/bead segmentation).
    scan_direction
        (3,) unit vector along the scan raster direction.
    sigma_scan
        Gaussian sigma along *scan_direction* (mm).
    sigma_perp
        Gaussian sigma perpendicular to *scan_direction* (mm).
    n_iterations
        Number of Jacobi-style smoothing passes.

    Returns
    -------
    smoothed : (N, 3)
        Smoothed point cloud (same length and ordering as *points*).
    """
    scan_dir = np.asarray(scan_direction, dtype=np.float64)
    scan_dir = scan_dir / np.linalg.norm(scan_dir)

    r_max = max(sigma_scan, sigma_perp) * 3.0
    inv_2ss = 1.0 / (2.0 * sigma_scan ** 2)
    inv_2sp = 1.0 / (2.0 * sigma_perp ** 2)

    current = points.astype(np.float64, copy=True)

    for _iteration in range(n_iterations):
        tree = cKDTree(current)
        normals = _estimate_normals(current, tree, radius=r_max)
        neighbour_lists = tree.query_ball_point(current, r_max, workers=-1)

        smoothed = current.copy()

        for i, nbrs in enumerate(neighbour_lists):
            if len(nbrs) < 2:
                continue

            neighbours = current[nbrs]
            displacements = neighbours - current[i]

            d_scan = displacements @ scan_dir
            d_perp_vec = displacements - np.outer(d_scan, scan_dir)
            d_perp_sq = np.sum(d_perp_vec ** 2, axis=1)

            weights = np.exp(-d_scan ** 2 * inv_2ss - d_perp_sq * inv_2sp)
            w_sum = weights.sum()
            if w_sum < 1e-12:
                continue

            v_mean = (weights[:, np.newaxis] * neighbours).sum(axis=0) / w_sum
            delta = v_mean - current[i]

            n_i = normals[i]
            correction = np.dot(delta, n_i) * n_i
            smoothed[i] = current[i] + correction

        current = smoothed

    return current
