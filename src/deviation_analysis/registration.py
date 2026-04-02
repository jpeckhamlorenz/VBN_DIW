"""Point cloud registration to CAD mesh.

Pipeline stages:
1. RANSAC plane fitting to flatten bed tilt
2. Bed/bead segmentation
3. FPFH-based RANSAC global registration for coarse alignment
4. Point-to-plane ICP for fine alignment
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from deviation_analysis.config import RegistrationConfig


@dataclass
class RegistrationResult:
    """Result of the full registration pipeline."""

    transform: np.ndarray
    """4x4 homogeneous transformation matrix (scan -> CAD frame)."""

    fitness: float
    """ICP fitness score (fraction of inlier correspondences)."""

    inlier_rmse: float
    """ICP inlier RMSE in mm."""

    plane_normal: np.ndarray
    """Fitted bed plane normal vector (before flattening)."""

    plane_rotation: np.ndarray
    """3x3 rotation matrix used to flatten the bed."""

    plane_intercept: float
    """Z-intercept subtracted after rotation."""


def _rotation_matrix_from_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """Compute rotation matrix that aligns vec1 to vec2.

    Uses Rodrigues' rotation formula.

    Parameters
    ----------
    vec1
        Source unit vector (will be normalized).
    vec2
        Target unit vector (will be normalized).

    Returns
    -------
    np.ndarray
        (3, 3) rotation matrix.
    """
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s == 0:
        return np.eye(3)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2 + 1e-10))


def fit_ransac_plane(
    points: np.ndarray,
    random_state: int | None = 42,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit a plane to a point cloud using RANSAC.

    Uses sklearn RANSACRegressor with PolynomialFeatures(degree=1) to fit
    z = a*x + b*y + c, then computes the rotation matrix that aligns the
    fitted plane normal with [0, 0, 1].

    Parameters
    ----------
    points
        (N, 3) array of XYZ coordinates (valid points only).

    Returns
    -------
    plane_normal : np.ndarray
        (3,) unit normal vector of the fitted plane.
    intercept : float
        Z-intercept of the fitted plane.
    rotation_matrix : np.ndarray
        (3, 3) rotation that aligns plane_normal to [0, 0, 1].
    """
    xy = points[:, :2]
    z = points[:, 2]

    ransac = make_pipeline(PolynomialFeatures(degree=1), RANSACRegressor(random_state=random_state))
    ransac.fit(xy, z)

    coef = ransac.named_steps["ransacregressor"].estimator_.coef_
    intercept = ransac.named_steps["ransacregressor"].estimator_.intercept_
    # PolynomialFeatures(degree=1) on 2D input produces [1, x, y]
    # so coef = [bias_term, a, b] where z = a*x + b*y + c
    a, b = coef[1], coef[2]

    plane_normal = np.array([-a, -b, 1.0])
    plane_normal /= np.linalg.norm(plane_normal)

    rotation_matrix = _rotation_matrix_from_vectors(plane_normal, np.array([0.0, 0.0, 1.0]))

    return plane_normal, intercept, rotation_matrix


def flatten_point_cloud(
    points: np.ndarray,
    valid_mask: np.ndarray,
    random_state: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply RANSAC plane fitting and rotation to flatten a scan point cloud.

    Fits a plane to the valid points, rotates all points so the plane normal
    aligns with Z, then shifts Z so the bed plane sits at Z=0.

    Parameters
    ----------
    points
        (N, 3) array of all points.
    valid_mask
        (N,) boolean mask of valid points.

    Returns
    -------
    flat_points : np.ndarray
        (N, 3) rotated and Z-shifted points (bed plane at Z=0).
    rotation_matrix : np.ndarray
        (3, 3) rotation applied.
    intercept : float
        Z-intercept subtracted after rotation.
    """
    valid_pts = points[valid_mask]
    plane_normal, intercept, rotation_matrix = fit_ransac_plane(valid_pts, random_state=random_state)

    # Rotate all points (including invalid ones — their coords are arbitrary but
    # we keep them for index consistency with valid_mask)
    flat_points = points @ rotation_matrix.T
    flat_points[:, 2] -= intercept

    return flat_points, rotation_matrix, intercept


def segment_bed_from_bead(
    flat_points: np.ndarray,
    valid_mask: np.ndarray,
    z_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Separate flattened points into bed and bead regions.

    Points with Z > z_threshold are classified as bead (printed material).
    Points at or below are classified as bed (print surface).

    Parameters
    ----------
    flat_points
        (N, 3) flattened point cloud.
    valid_mask
        (N,) boolean mask of valid points.
    z_threshold
        Height threshold in mm above flattened bed plane.

    Returns
    -------
    bead_points : np.ndarray
        (M, 3) points classified as printed material.
    bed_points : np.ndarray
        (K, 3) points classified as print bed.
    bead_mask : np.ndarray
        (N,) boolean mask where True = valid bead point.
    """
    bead_mask = valid_mask & (flat_points[:, 2] > z_threshold)
    bed_mask = valid_mask & (flat_points[:, 2] <= z_threshold)

    bead_points = flat_points[bead_mask]
    bed_points = flat_points[bed_mask]

    return bead_points, bed_points, bead_mask


def augment_bead_with_floor(
    bead_points: np.ndarray,
    floor_z: float = 0.0,
) -> np.ndarray:
    """Add synthetic floor points underneath bead points to close the shape.

    The laser scan only captures the top surface of the printed bead, but the
    CAD mesh is a closed solid with top, bottom, and side faces.  FPFH feature
    descriptors struggle with thin, open surfaces because every point looks
    like "flat surface, normal up" — there are no edges or curvature changes
    to create distinctive signatures.

    By projecting the bead footprint down to the bed plane (Z = 0), we create
    a synthetic bottom face.  After voxel downsampling, the boundary between
    top and bottom surfaces produces sharp edges with highly distinctive FPFH
    features, dramatically improving feature matching.

    These floor points are used **only for registration** (computing the
    alignment transform).  They are never included in distance computation
    or deviation metrics.

    Parameters
    ----------
    bead_points
        (M, 3) array of bead surface points (post-segmentation, bed at Z=0).
    floor_z
        Z coordinate for the synthetic floor.  Default 0.0 (bed level after
        flattening).

    Returns
    -------
    np.ndarray
        (M + F, 3) combined array: original bead points on top, floor points
        appended below.  F ≤ M (may be fewer if voxel deduplication is used).
    """
    # Create floor points: same XY as bead, Z = floor_z
    floor_points = bead_points.copy()
    floor_points[:, 2] = floor_z

    # Combine: original bead on top, floor appended
    augmented = np.vstack([bead_points, floor_points])
    return augmented


def compute_fpfh_features(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    radius_normal: float,
    radius_feature: float,
) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    """Downsample a point cloud and compute FPFH features.

    Steps:
    1. Voxel downsample
    2. Estimate normals with hybrid KDTree search
    3. Compute FPFH features

    Parameters
    ----------
    pcd
        Input point cloud.
    voxel_size
        Voxel size for downsampling in mm.
    radius_normal
        Search radius for normal estimation in mm.
    radius_feature
        Search radius for FPFH computation in mm.

    Returns
    -------
    pcd_down : o3d.geometry.PointCloud
        Downsampled point cloud with normals.
    fpfh : o3d.pipelines.registration.Feature
        FPFH feature descriptors.
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, fpfh


def ransac_global_registration(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    source_fpfh: o3d.pipelines.registration.Feature,
    target_fpfh: o3d.pipelines.registration.Feature,
    config: RegistrationConfig,
) -> np.ndarray:
    """RANSAC-based global registration using FPFH feature matching.

    Parameters
    ----------
    source_pcd
        Source (scan) point cloud, downsampled.
    target_pcd
        Target (CAD mesh samples) point cloud, downsampled.
    source_fpfh
        FPFH features for source.
    target_fpfh
        FPFH features for target.
    config
        Registration parameters.

    Returns
    -------
    np.ndarray
        4x4 transformation matrix (source -> target).
    """
    distance_threshold = config.fpfh_voxel_size * 1.5

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_pcd,
        target_pcd,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=config.ransac_n_points,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration=config.ransac_max_iterations,
            confidence=config.ransac_confidence,
        ),
    )
    return np.asarray(result.transformation)


def icp_refine(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    config: RegistrationConfig,
) -> tuple[np.ndarray, float, float]:
    """Refine alignment using point-to-plane ICP.

    The target point cloud must have normals estimated.

    Parameters
    ----------
    source_pcd
        Source (scan bead) point cloud.
    target_pcd
        Target (CAD mesh samples) point cloud with normals.
    init_transform
        Initial 4x4 transformation from coarse registration.
    config
        Registration parameters (threshold, max iterations).

    Returns
    -------
    transform : np.ndarray
        Refined 4x4 transformation matrix.
    fitness : float
        Fraction of source points with correspondences below threshold.
    inlier_rmse : float
        RMSE of inlier correspondences in mm.
    """
    # Ensure source also has normals for point-to-plane
    if not source_pcd.has_normals():
        source_pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=config.fpfh_radius_normal, max_nn=30
            )
        )

    result = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        config.icp_threshold,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=config.icp_max_iterations
        ),
    )
    return np.asarray(result.transformation), result.fitness, result.inlier_rmse


def register_scan_to_cad(
    bead_points: np.ndarray,
    cad_mesh: o3d.geometry.TriangleMesh,
    config: RegistrationConfig,
) -> tuple[np.ndarray, float, float]:
    """Full registration pipeline: FPFH global + ICP refinement.

    1. Sample points from CAD mesh surface (with normals from the mesh)
    2. Create Open3D point cloud from bead points
    3. Compute FPFH features on both (after voxel downsampling)
    4. RANSAC global registration for coarse alignment
    5. ICP refinement for fine alignment

    Parameters
    ----------
    bead_points
        (M, 3) segmented bead points from scan.
    cad_mesh
        CAD reference mesh loaded via Open3D (with vertex normals).
    config
        Registration configuration parameters.

    Returns
    -------
    transform : np.ndarray
        4x4 transformation matrix (scan bead frame -> CAD frame).
    fitness : float
        ICP fitness score.
    inlier_rmse : float
        ICP inlier RMSE in mm.
    """
    # Seed all PRNGs for reproducible registration
    if config.random_seed is not None:
        np.random.seed(config.random_seed)
        try:
            o3d.utility.random.seed(config.random_seed)
        except AttributeError:
            pass  # Open3D < 0.17 lacks this API

    # Sample points from CAD mesh to create target point cloud
    # Use Poisson disk sampling for even coverage; fall back to uniform
    num_target_points = max(50_000, len(bead_points) // 2)
    target_pcd = cad_mesh.sample_points_poisson_disk(
        number_of_points=num_target_points,
    )
    # Normals from mesh are inherited by sampled points via Poisson disk sampling
    if not target_pcd.has_normals():
        target_pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=config.fpfh_radius_normal, max_nn=30
            )
        )

    # Optionally augment bead with synthetic floor to improve feature matching
    # on thin geometry.  The augmented cloud is used for FPFH + RANSAC + ICP
    # to get a better transform.  The transform is then applied to only the
    # original bead points downstream (in deviation computation).
    if config.augment_with_floor:
        reg_points = augment_bead_with_floor(bead_points, floor_z=0.0)
    else:
        reg_points = bead_points

    # Create source point cloud for registration
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(reg_points)

    # Compute FPFH features on downsampled versions of both
    source_down, source_fpfh = compute_fpfh_features(
        source_pcd,
        config.fpfh_voxel_size,
        config.fpfh_radius_normal,
        config.fpfh_radius_feature,
    )
    target_down, target_fpfh = compute_fpfh_features(
        target_pcd,
        config.fpfh_voxel_size,
        config.fpfh_radius_normal,
        config.fpfh_radius_feature,
    )

    # Stage 1: RANSAC global registration (coarse)
    coarse_transform = ransac_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, config
    )

    # Stage 2: ICP refinement (fine) — use augmented source against target
    source_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=config.fpfh_radius_normal, max_nn=30
        )
    )
    transform, fitness, inlier_rmse = icp_refine(
        source_pcd, target_pcd, coarse_transform, config
    )

    return transform, fitness, inlier_rmse
