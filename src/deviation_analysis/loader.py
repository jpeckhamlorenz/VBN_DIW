"""Load and preprocess Keyence raster scan CSVs into 3D point clouds.

Adapted from beadscan_processor.py data loading patterns but as standalone
functions suitable for the deviation analysis pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.interpolate import interp1d

from deviation_analysis.config import ScanConfig


def load_scan_csv(filepath: Path, config: ScanConfig) -> np.ndarray:
    """Load a raw Keyence raster scan CSV and convert to mm.

    The CSV is headerless with ~6000 rows x 800 columns of raw integer values.
    Values below ``config.invalid_threshold`` are replaced with NaN.
    Remaining values are scaled by ``config.scale_factor`` to convert to mm.

    Parameters
    ----------
    filepath
        Path to the scan cycle CSV file.
    config
        Scan configuration with thresholds and scale factor.

    Returns
    -------
    np.ndarray
        2D array (rows, cols) of height values in mm, with NaN for invalid points.
    """
    data = np.loadtxt(filepath, delimiter=",")
    data = np.where(data < config.invalid_threshold, np.nan, data)
    data_mm = data * config.scale_factor
    return data_mm


def load_toolpath_csv(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a scan trajectory CSV.

    The CSV has a header row with columns: t,x,y,z,w,w_pred,v_noz,q,r.
    Only t, x, y, z columns are used.

    Parameters
    ----------
    filepath
        Path to the toolpath CSV file.

    Returns
    -------
    time : np.ndarray
        (N,) array of timestamps.
    xyz : np.ndarray
        (N, 3) array of XYZ coordinates in mm.
    """
    time = np.loadtxt(filepath, delimiter=",", skiprows=1, usecols=(0,))
    xyz = np.loadtxt(filepath, delimiter=",", skiprows=1, usecols=(1, 2, 3))
    return time, xyz


def _upsample_toolpath(
    toolpath_xyz: np.ndarray,
    num_rows: int,
) -> np.ndarray:
    """Upsample toolpath XYZ to match the number of scan rows.

    Uses linear interpolation if toolpath has fewer points than scan rows.

    Parameters
    ----------
    toolpath_xyz
        (N, 3) original toolpath coordinates.
    num_rows
        Target number of rows (from scan data).

    Returns
    -------
    np.ndarray
        (num_rows, 3) upsampled toolpath coordinates.
    """
    if len(toolpath_xyz) >= num_rows:
        return toolpath_xyz[:num_rows]
    interp_func = interp1d(
        np.linspace(0, 1, len(toolpath_xyz)),
        toolpath_xyz,
        axis=0,
        kind="linear",
        fill_value="extrapolate",
    )
    return interp_func(np.linspace(0, 1, num_rows))


def height_correct(
    data_mm: np.ndarray,
    toolpath_xyz: np.ndarray,
) -> np.ndarray:
    """Correct for height variations in the scanning toolpath.

    Adjusts each row of the raster data by the difference between that row's
    scanning Z height and the mean Z height across the toolpath.

    Parameters
    ----------
    data_mm
        (rows, cols) raster data in mm.
    toolpath_xyz
        (N, 3) toolpath coordinates. N is upsampled to match rows if needed.

    Returns
    -------
    np.ndarray
        Height-corrected raster data, same shape as ``data_mm``.
    """
    num_rows = data_mm.shape[0]
    toolpath_up = _upsample_toolpath(toolpath_xyz, num_rows)
    heights = toolpath_up[:, 2]
    mean_height = np.mean(heights)
    adjustments = heights - mean_height
    # Broadcast row-wise adjustment across all columns
    return data_mm + adjustments[:, np.newaxis]


def raster_to_point_cloud(
    data_mm: np.ndarray,
    toolpath_xyz: np.ndarray,
    config: ScanConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a raster height grid to a 3D point cloud.

    X coordinates are computed from column indices times ``config.resolution``.
    Y coordinates come from the toolpath Y values (upsampled to match row count).
    Z coordinates are the raster height values.

    Parameters
    ----------
    data_mm
        (rows, cols) height-corrected raster data in mm.
    toolpath_xyz
        (N, 3) toolpath coordinates for Y-axis mapping.
    config
        Scan configuration with resolution parameter.

    Returns
    -------
    points : np.ndarray
        (rows*cols, 3) array of XYZ coordinates.
    valid_mask : np.ndarray
        (rows*cols,) boolean mask, True where height data is valid (not NaN).
    """
    rows, cols = data_mm.shape
    toolpath_up = _upsample_toolpath(toolpath_xyz, rows)

    x = np.arange(cols) * config.resolution
    y = toolpath_up[:, 1]
    X, Y = np.meshgrid(x, y)
    Z = data_mm

    points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    valid_mask = ~np.isnan(Z.ravel())
    return points, valid_mask


def points_to_open3d(
    points: np.ndarray,
    valid_mask: np.ndarray,
) -> o3d.geometry.PointCloud:
    """Convert valid points to an Open3D PointCloud.

    Parameters
    ----------
    points
        (N, 3) array of all points (including invalid).
    valid_mask
        (N,) boolean mask selecting valid points.

    Returns
    -------
    o3d.geometry.PointCloud
        Point cloud containing only valid points.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[valid_mask])
    return pcd


def load_cad_mesh(filepath: Path) -> o3d.geometry.TriangleMesh:
    """Load a CAD reference STL as an Open3D TriangleMesh.

    Parameters
    ----------
    filepath
        Path to the binary STL file.

    Returns
    -------
    o3d.geometry.TriangleMesh
        Triangle mesh with computed vertex normals.
    """
    mesh = o3d.io.read_triangle_mesh(str(filepath))
    mesh.compute_vertex_normals()
    return mesh
