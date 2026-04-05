"""Load and preprocess Keyence raster scan CSVs into 3D point clouds.

Adapted from beadscan_processor.py data loading patterns but as standalone
functions suitable for the deviation analysis pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.interpolate import interp1d
from scipy.ndimage import binary_erosion, binary_fill_holes
from scipy.spatial import Delaunay

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


def _filter_long_edges(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_edge_length: float,
) -> np.ndarray:
    """Return boolean mask of faces whose longest edge ≤ *max_edge_length*."""
    v = vertices[faces]  # (F, 3, 3)
    e0 = np.linalg.norm(v[:, 1] - v[:, 0], axis=1)
    e1 = np.linalg.norm(v[:, 2] - v[:, 1], axis=1)
    e2 = np.linalg.norm(v[:, 0] - v[:, 2], axis=1)
    return np.maximum(np.maximum(e0, e1), e2) <= max_edge_length


def _fix_winding(
    vertices: np.ndarray,
    faces: np.ndarray,
    desired_z_sign: float,
) -> np.ndarray:
    """Flip triangle winding so face normals have the desired Z sign.

    Parameters
    ----------
    vertices
        (V, 3) vertex array.
    faces
        (F, 3) int face array (modified in-place and returned).
    desired_z_sign
        +1.0 for upward-facing normals, −1.0 for downward-facing.
    """
    if len(faces) == 0:
        return faces
    faces = faces.copy()
    v = vertices[faces]
    nz = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])[:, 2]
    flip = (nz * desired_z_sign) < 0
    faces[flip] = faces[flip][:, ::-1]
    return faces


def save_points_as_stl(
    points: np.ndarray,
    output_path: Path,
    max_edge_length: float | None = None,
) -> Path:
    """Save a point cloud as an STL mesh using 2D Delaunay triangulation.

    The scan data is essentially a 2.5D height map (one Z per XY location),
    so we triangulate on the XY plane and use the Z values as vertex heights.
    This produces a clean surface mesh without the artifacts of Poisson or
    ball-pivot reconstruction.

    Optionally filters out triangles with edges longer than
    ``max_edge_length`` to avoid large triangles spanning gaps in the data
    (e.g., between separate strokes of the letter M).

    Parameters
    ----------
    points
        (N, 3) array of XYZ coordinates. Should contain only valid points
        (no NaNs).
    output_path
        Where to write the binary STL file.
    max_edge_length
        If provided, triangles with any edge longer than this value (in mm)
        are removed.  A good default for typical scan data is ~0.5 mm
        (about 5-10× the scan resolution).

    Returns
    -------
    Path
        The path the STL was written to.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 2D Delaunay on XY projection
    tri = Delaunay(points[:, :2])
    faces = tri.simplices  # (F, 3) triangle vertex indices

    # Optionally filter long-edge triangles (removes spanning artifacts)
    if max_edge_length is not None:
        keep = np.ones(len(faces), dtype=bool)
        for i, face in enumerate(faces):
            v0, v1, v2 = points[face]
            e0 = np.linalg.norm(v1 - v0)
            e1 = np.linalg.norm(v2 - v1)
            e2 = np.linalg.norm(v0 - v2)
            if max(e0, e1, e2) > max_edge_length:
                keep[i] = False
        faces = faces[keep]

    # Build Open3D mesh and save
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(str(output_path), mesh)
    return output_path


def build_sidewalled_mesh(
    bead_points: np.ndarray,
    bead_mask: np.ndarray,
    n_rows: int,
    n_cols: int,
    output_path: Path,
    floor_z: float = 0.0,
    max_edge_length: float = 0.5,
) -> Path:
    """Build an STL mesh with bead top surface, vertical sidewalls, and floor.

    Unlike :func:`save_points_as_stl` (2-D Delaunay, can't represent vertical
    geometry), this function constructs the mesh explicitly:

    * **Top surface** — 2-D Delaunay on bead point XY projection.
    * **Sidewalls** — quad strips between adjacent edge-pixel columns,
      connecting bead surface to floor.
    * **Floor** — 2-D Delaunay on edge-pixel XY footprint at *floor_z*.

    Parameters
    ----------
    bead_points
        (M, 3) smoothed bead surface points **before** sidewall point
        augmentation (i.e. one Z per XY location).
    bead_mask
        (N,) bool mask over the full flattened raster grid.
    n_rows, n_cols
        Raster grid dimensions (``N = n_rows * n_cols``).
    output_path
        Where to write the binary STL.
    floor_z
        Z level of the synthetic floor (mm).
    max_edge_length
        Triangles with any edge longer than this (mm) are removed from the
        top and floor surfaces to avoid spanning gaps between bead strokes.

    Returns
    -------
    Path
        The path the STL was written to.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    M = len(bead_points)

    # ── Edge detection (outer perimeter only, ignores internal holes) ─
    from deviation_analysis.smoothing import outer_edge_mask

    mask_2d = bead_mask.reshape(n_rows, n_cols)
    edge_2d = outer_edge_mask(mask_2d)

    bead_indices = np.flatnonzero(bead_mask)
    flat_to_bead = np.full(n_rows * n_cols, -1, dtype=np.intp)
    flat_to_bead[bead_indices] = np.arange(len(bead_indices))

    edge_flat_indices = np.flatnonzero(edge_2d.ravel() & bead_mask)
    E = len(edge_flat_indices)

    # ── Floor vertices ────────────────────────────────────────────────
    edge_bead_idx = flat_to_bead[edge_flat_indices]
    floor_verts = bead_points[edge_bead_idx].copy()
    floor_verts[:, 2] = floor_z

    vertices = np.vstack([bead_points, floor_verts])  # (M+E, 3)

    # Lookup: flat grid index → floor vertex index (M-based)
    flat_to_floor = np.full(n_rows * n_cols, -1, dtype=np.intp)
    for k, fi in enumerate(edge_flat_indices):
        flat_to_floor[fi] = M + k

    # ── Top surface faces (2-D Delaunay) ──────────────────────────────
    tri_top = Delaunay(bead_points[:, :2])
    top_faces = tri_top.simplices
    if max_edge_length is not None:
        top_faces = top_faces[_filter_long_edges(vertices, top_faces, max_edge_length)]

    # ── Sidewall quad faces ───────────────────────────────────────────
    is_edge = np.zeros(n_rows * n_cols, dtype=bool)
    is_edge[edge_flat_indices] = True

    sw_faces: list[list[int]] = []
    for fi in edge_flat_indices:
        row, col = divmod(int(fi), n_cols)
        top_a = flat_to_bead[fi]
        floor_a = flat_to_floor[fi]
        if top_a < 0 or floor_a < 0:
            continue

        # Right neighbour
        if col + 1 < n_cols:
            fi_r = fi + 1
            if is_edge[fi_r]:
                top_b = flat_to_bead[fi_r]
                floor_b = flat_to_floor[fi_r]
                if top_b >= 0 and floor_b >= 0:
                    sw_faces.append([top_a, top_b, floor_b])
                    sw_faces.append([top_a, floor_b, floor_a])

        # Bottom neighbour
        if row + 1 < n_rows:
            fi_d = fi + n_cols
            if is_edge[fi_d]:
                top_b = flat_to_bead[fi_d]
                floor_b = flat_to_floor[fi_d]
                if top_b >= 0 and floor_b >= 0:
                    sw_faces.append([top_a, top_b, floor_b])
                    sw_faces.append([top_a, floor_b, floor_a])

    sidewall_faces = np.array(sw_faces, dtype=np.int32) if sw_faces else np.empty((0, 3), dtype=np.int32)

    # ── Floor surface faces (2-D Delaunay on edge footprint) ──────────
    if E >= 3:
        tri_floor = Delaunay(floor_verts[:, :2])
        floor_faces = tri_floor.simplices + M  # offset into vertex array
        if max_edge_length is not None:
            floor_faces = floor_faces[_filter_long_edges(vertices, floor_faces, max_edge_length)]

        # Remove floor triangles whose centroid falls outside the bead
        # footprint — Delaunay fills the convex hull, which spans across
        # concavities of shapes like "M".
        if len(floor_faces) > 0:
            filled = binary_fill_holes(mask_2d)
            local_faces = floor_faces - M  # back to 0-based edge indices
            edge_rows = edge_flat_indices // n_cols
            edge_cols = edge_flat_indices % n_cols
            crows = np.round(edge_rows[local_faces].mean(axis=1)).astype(int)
            ccols = np.round(edge_cols[local_faces].mean(axis=1)).astype(int)
            crows = np.clip(crows, 0, n_rows - 1)
            ccols = np.clip(ccols, 0, n_cols - 1)
            floor_faces = floor_faces[filled[crows, ccols]]
    else:
        floor_faces = np.empty((0, 3), dtype=np.int32)

    # ── Fix winding order ────────────────────────────────────────────
    # Ensure top faces point +Z (outward), floor faces point −Z.
    top_faces = _fix_winding(vertices, top_faces, desired_z_sign=+1.0)
    if len(floor_faces) > 0:
        floor_faces = _fix_winding(vertices, floor_faces, desired_z_sign=-1.0)

    # ── Combine and export ────────────────────────────────────────────
    all_faces = np.vstack([top_faces, sidewall_faces, floor_faces])

    # Use Open3D for export — it computes vertex normals without requiring
    # networkx (which trimesh.fix_normals needs but isn't installed).
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(all_faces)
    o3d_mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(output_path), o3d_mesh)
    return output_path
