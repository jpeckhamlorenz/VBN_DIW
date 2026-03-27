"""Signed distance computation and deviation metric extraction.

Computes per-point signed distances from aligned scan points to the CAD mesh
surface, then extracts summary statistics for quantitative comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import trimesh
from scipy import stats

from deviation_analysis.config import DeviationConfig


@dataclass
class DeviationMetrics:
    """Summary metrics for one scan's deviation from CAD geometry."""

    rmsd: float
    """Root mean square deviation in mm."""

    msd: float
    """Mean signed deviation in mm (positive = net over-extrusion)."""

    mae: float
    """Mean absolute deviation in mm."""

    percentile_abs: float
    """Nth percentile of absolute deviations in mm (N from config, default 95)."""

    max_abs_deviation: float
    """Maximum absolute deviation in mm."""

    excess_volume_mm3: float
    """Total volume of over-extruded material in mm^3."""

    deficit_volume_mm3: float
    """Total volume of under-extruded material in mm^3."""

    n_points: int
    """Number of points used in the analysis."""

    signed_distances: np.ndarray
    """Raw signed distance array for downstream analysis (KDE, KS tests)."""


def compute_signed_distances(
    scan_points_aligned: np.ndarray,
    cad_mesh: trimesh.Trimesh,
) -> np.ndarray:
    """Compute signed distances from aligned scan points to CAD mesh surface.

    For each scan point, finds the closest point on the mesh surface.
    Sign is determined by the dot product of (scan_point - closest_point)
    with the face normal at the closest point:
    - Positive = point outside mesh (over-extrusion)
    - Negative = point inside mesh (under-extrusion)

    Parameters
    ----------
    scan_points_aligned
        (M, 3) scan points transformed into CAD coordinate frame.
    cad_mesh
        CAD reference mesh loaded via trimesh.

    Returns
    -------
    np.ndarray
        (M,) signed distances in mm.
    """
    closest_points, distances, triangle_ids = trimesh.proximity.closest_point(
        cad_mesh, scan_points_aligned
    )

    # Determine sign: vector from closest surface point to scan point,
    # dotted with the face normal at that closest point
    vectors = scan_points_aligned - closest_points
    face_normals = cad_mesh.face_normals[triangle_ids]
    dots = np.sum(vectors * face_normals, axis=1)
    signs = np.sign(dots)

    # Points exactly on the surface get sign 0; treat as positive (on-surface)
    signs[signs == 0] = 1.0

    return distances * signs


def compute_metrics(
    signed_distances: np.ndarray,
    point_area: float,
    config: DeviationConfig,
) -> DeviationMetrics:
    """Extract summary metrics from a signed distance array.

    Parameters
    ----------
    signed_distances
        (M,) signed distances in mm.
    point_area
        Approximate surface area represented by each scan point in mm^2.
        For raster data: resolution * slice_thickness.
    config
        Deviation configuration (percentile threshold, etc.).

    Returns
    -------
    DeviationMetrics
        Dataclass containing all computed metrics.
    """
    abs_distances = np.abs(signed_distances)

    rmsd = float(np.sqrt(np.mean(signed_distances**2)))
    msd = float(np.mean(signed_distances))
    mae = float(np.mean(abs_distances))
    percentile_abs = float(np.percentile(abs_distances, config.percentile))
    max_abs_deviation = float(np.max(abs_distances))

    # Volume estimation: each point represents a small area patch.
    # The signed distance at that point times the area gives volumetric contribution.
    positive_mask = signed_distances > 0
    negative_mask = signed_distances < 0
    excess_volume = float(np.sum(signed_distances[positive_mask]) * point_area)
    deficit_volume = float(np.sum(np.abs(signed_distances[negative_mask])) * point_area)

    return DeviationMetrics(
        rmsd=rmsd,
        msd=msd,
        mae=mae,
        percentile_abs=percentile_abs,
        max_abs_deviation=max_abs_deviation,
        excess_volume_mm3=excess_volume,
        deficit_volume_mm3=deficit_volume,
        n_points=len(signed_distances),
        signed_distances=signed_distances,
    )


def ks_test(
    distances_a: np.ndarray,
    distances_b: np.ndarray,
) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test on signed deviation distributions.

    Parameters
    ----------
    distances_a
        Signed distances from method A.
    distances_b
        Signed distances from method B.

    Returns
    -------
    statistic : float
        KS test statistic.
    p_value : float
        Two-sided p-value.
    """
    result = stats.ks_2samp(distances_a, distances_b)
    return float(result.statistic), float(result.pvalue)


def compute_pairwise_ks(
    method_distances: dict[str, np.ndarray],
) -> dict[tuple[str, str], tuple[float, float]]:
    """Compute KS test for all unique pairs of methods.

    Parameters
    ----------
    method_distances
        Mapping from method name to its signed distance array.

    Returns
    -------
    dict
        Mapping from (method_a, method_b) to (KS statistic, p-value).
        Only includes pairs where method_a < method_b alphabetically.
    """
    results = {}
    for name_a, name_b in combinations(sorted(method_distances.keys()), 2):
        results[(name_a, name_b)] = ks_test(
            method_distances[name_a], method_distances[name_b]
        )
    return results
