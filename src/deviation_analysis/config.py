"""Configuration dataclasses for the deviation analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ScanConfig:
    """Parameters for loading and converting Keyence raster scan data."""

    resolution: float = 0.02
    """Column spacing in mm (Keyence profiler pixel pitch)."""

    scan_speed: float = 5.0
    """Scan speed in mm/s. Used to compute Y-axis spacing."""

    scan_rate: float = 1000.0
    """Scan rate in Hz."""

    invalid_threshold: float = 10.0
    """Raw values below this are treated as invalid (NaN)."""

    scale_factor: float = 1e-3
    """Multiply raw values by this to convert to mm."""

    @property
    def slice_thickness(self) -> float:
        """Y-axis spacing between scan rows in mm."""
        return self.scan_speed / self.scan_rate


@dataclass
class RegistrationConfig:
    """Parameters for point cloud registration to CAD mesh."""

    # RANSAC plane fitting (bed flattening)
    plane_ransac_residual_threshold: float = 0.05
    """Residual threshold in mm for RANSAC plane inlier classification."""

    bed_z_threshold: float = 0.15
    """Z height in mm above flattened bed plane to separate bead from bed.
    Used as fallback if RANSAC inlier/outlier segmentation is insufficient."""

    # FPFH feature-based global registration
    fpfh_voxel_size: float = 0.5
    """Voxel size in mm for downsampling before FPFH computation."""

    fpfh_radius_normal: float = 1.0
    """Search radius in mm for normal estimation."""

    fpfh_radius_feature: float = 2.5
    """Search radius in mm for FPFH feature computation."""

    ransac_n_points: int = 3
    """Number of points sampled per RANSAC iteration in global registration."""

    ransac_max_iterations: int = 4_000_000
    """Maximum RANSAC iterations for global registration."""

    ransac_confidence: float = 0.999
    """RANSAC confidence threshold for early termination."""

    # ICP refinement
    icp_threshold: float = 0.1
    """Maximum correspondence distance in mm for ICP."""

    icp_max_iterations: int = 200
    """Maximum ICP iterations."""

    # Floor augmentation for thin geometry
    augment_with_floor: bool = True
    """Add synthetic floor points (Z=0) under the bead before registration.
    This closes the open scan surface to better match the closed CAD mesh,
    giving FPFH features more geometric contrast for thin/flat parts."""

    random_seed: int | None = 42
    """Random seed for RANSAC operations (sklearn + Open3D). None = non-deterministic."""


@dataclass
class SmoothingConfig:
    """
    Parameters for anisotropic scan-direction smoothing.

    Removes periodic ridges caused by oscillatory robot arm speed during
    raster scanning.  Smoothing is applied along the known scan direction
    to the surface-normal component of vertex displacement only, preserving
    macro-geometry (bead trajectory, corners, width).
    """

    enabled: bool = True
    """Master toggle for smoothing."""

    sigma_scan: float = 0.9
    """Gaussian sigma along scan direction (mm)."""

    sigma_perp: float = 0.04
    """Gaussian sigma perpendicular to scan direction (mm)."""

    n_iterations: int = 1
    """Number of Jacobi-style smoothing passes."""

    scan_direction: tuple[float, float, float] = (0.0, 1.0, 0.0)
    """Unit vector of scan direction. Y-axis for M demo raster scans."""

    use_smoothed: bool = True
    """If True, downstream stages (registration, deviation) use smoothed points.
    If False, smoothing still runs and STLs are exported but pipeline uses raw."""


@dataclass
class DeviationConfig:
    """Parameters for signed distance computation and metric extraction."""

    percentile: float = 95.0
    """Percentile for worst-case deviation metric."""


@dataclass
class VisualizationConfig:
    """Parameters for deviation map rendering."""

    clim: tuple[float, float] = (-0.3, 0.3)
    """Color scale limits in mm (symmetric around 0 for diverging colormap)."""

    dpi: int = 300
    """Resolution for saved figures."""

    figure_width_mm: float = 183.0
    """Figure width in mm (Nature double-column = 183mm)."""


@dataclass
class MethodSpec:
    """Specification for one experimental method and its scan data."""

    name: str
    """Internal key, e.g. 'm_VBN_05'."""

    display_name: str
    """Human-readable name for figures/tables, e.g. 'VBN (v5)'."""

    toolpath_csv: str
    """Filename of the scan trajectory CSV (for Y-axis mapping)."""

    cycle_csvs: list[str] = field(default_factory=list)
    """Filenames of selected scan cycle CSVs to process."""


@dataclass
class PipelineConfig:
    """Top-level configuration for the deviation analysis pipeline."""

    data_dir: Path = field(default_factory=lambda: Path('src/demos/m'))
    """Directory containing scan CSVs and STL file."""

    output_dir: Path = field(default_factory=lambda: Path('src/deviation_analysis/output'))
    """Directory for final figures, tables, and reports."""

    cache_dir: Path = field(default_factory=lambda: Path('src/deviation_analysis/cache'))
    """Directory for intermediate NPZ cache files."""

    stl_file: str = 'm_ideal.stl'
    """Filename of the CAD reference STL (relative to data_dir)."""

    scan: ScanConfig = field(default_factory=ScanConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    deviation: DeviationConfig = field(default_factory=DeviationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    methods: list[MethodSpec] = field(default_factory=list)
    """List of methods to process. If empty, must be populated before batch run."""

    @property
    def stl_path(self) -> Path:
        return self.data_dir / self.stl_file

    def scan_path(self, filename: str) -> Path:
        return self.data_dir / filename
