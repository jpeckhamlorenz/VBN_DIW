# VBN-DIW: 3D Print Deviation Analysis

## Project Overview

This codebase supports a Nature Communications paper on Direct Ink Writing (DIW) with a variable-bead nozzle (VBN) system. It includes hardware control code (ROS nodes for Kuka robot, Xeryon stages, Keyence laser profiler), bead scan processing (`beadscan_processor.py`), and a **deviation analysis pipeline** (`src/deviation_analysis/`) that quantitatively compares 3D-printed parts against their ideal CAD geometry.

## Architecture

### Deviation Analysis Pipeline (`src/deviation_analysis/`)

```
scan_cycle.csv + toolpath.csv
        |
   [loader.py]  load CSV -> clean -> height correct -> raster to point cloud
        |
   raw_points (N,3) + valid_mask
        |
   [registration.py]  RANSAC plane fit -> flatten bed tilt -> segment bed/bead
        |
   bead_points (M,3)              m_ideal.stl
        |                              |
   [registration.py]  FPFH + RANSAC global registration + point-to-plane ICP
        |
   aligned_bead_points (M,3)      trimesh.Trimesh
        |                              |
   [deviation.py]  signed distances -> metrics
        |
   DeviationMetrics               signed_distances (M,)
        |                              |
   [visualization.py]  spatial maps, KDE plots, summary tables
```

**Modules:**
- `config.py` — Dataclass configs for pipeline parameters, file paths, method specifications
- `loader.py` — Load Keyence raster CSVs and toolpath CSVs, convert to 3D point clouds
- `registration.py` — RANSAC plane flatten, bed/bead segmentation, FPFH global + ICP fine registration to CAD mesh
- `deviation.py` — Signed distance computation (trimesh), metric extraction (RMSD, MSD, 95th%, volume), KS tests
- `visualization.py` — Spatial deviation color maps (pyvista), KDE distribution plots, summary tables (matplotlib)
- `batch.py` — Batch processing across methods/cycles, aggregation, output generation
- `cache.py` — NPZ-based caching of intermediate results with mtime invalidation
- `cli.py` — argparse CLI entry point for batch and single-scan processing

### Legacy Code (do not modify)
- `beadscan_processor.py` — Per-slice bead profile analysis (areas, flowrates). Reference for data loading patterns.
- `VBN.py`, `Keyence_scan.py`, `Xeryon.py` — ROS hardware control nodes

## Key Design Decisions

- **Registration:** Two-stage — RANSAC global (FPFH features) for coarse, point-to-plane ICP for fine. Fallback: toolpath-based intermediary registration if FPFH struggles with thin geometry.
- **Signed distance convention:** Positive = material outside CAD boundary (over-extrusion). Negative = inside (under-extrusion/gap). Computed via trimesh closest-point + face normal dot product.
- **Volume computation:** Approximate — each raster point represents area = resolution × slice_thickness. Volume contribution = signed_distance × point_area. Avoids Poisson reconstruction sensitivity.
- **Bed removal:** RANSAC plane fit identifies the flat print bed; inlier/outlier classification separates bed from bead, with Z threshold as fallback.
- **Caching:** NPZ files stored in `src/deviation_analysis/cache/`. Cached after flatten, registration, and distance stages. Invalidated when source file is newer than cache.
- **Scan selection:** Config explicitly specifies which cycle CSVs to use per method. User manually selects best representative scans.

## Dependencies

Defined in `pyproject.toml`. Key packages:
- `open3d >=0.19.0` — Point cloud processing, FPFH features, ICP registration
- `trimesh >=4.0.0` — Signed distance queries, mesh operations
- `pyvista >=0.43.0` — 3D deviation map rendering for publication figures
- `scikit-learn >=1.7.2` — RANSAC plane fitting
- `scipy >=1.15.3` — KS tests, KDE, interpolation
- `numpy >=2.2.6`, `matplotlib >=3.10.8`, `tqdm >=4.67.3`

## Usage

### Single scan (Python API)
```python
from deviation_analysis.config import PipelineConfig, MethodSpec
from deviation_analysis.batch import process_single_scan

config = PipelineConfig(data_dir=Path("src/demos/m"))
# ... configure and call process_single_scan()
```

### Batch processing (CLI)
```bash
python -m deviation_analysis.cli batch --data-dir src/demos/m --output-dir src/deviation_analysis/output
```

### Individual stages
Each module's functions are independently callable for step-by-step inspection (load → flatten → segment → register → compute distances → visualize).

## File Conventions

### Input
- **Scan CSVs:** Headerless, ~6000 rows × 800 cols of raw integer values from Keyence profiler. Units: divide by 1000 → mm. Naming: `{method}_cycle_{NNN}.csv`
- **Toolpath CSVs:** Header row `t,x,y,z,w,w_pred,v_noz,q,r`. Naming: `{method}.csv` (no `cycle` in name). These are scan trajectories, not print toolpaths.
- **CAD STL:** Binary STL in mm. One per demo geometry (e.g., `m_ideal.stl`).

### Output
- `src/deviation_analysis/output/` — Final figures (PNG), metric tables (CSV), summary reports
- `src/deviation_analysis/cache/` — Intermediate NPZ files (flattened points, transforms, distances)

### Naming
- Method keys: `m_VBN_05`, `m_static_ideal`, `m_static_naive`, etc.
- Cache files: `{method}_{cycle}_{stage}.npz`
- Figures: `{method}_{cycle}_deviation_map.png`, `kde_comparison.png`, `metrics_table.png`

## Current Status / TODOs

- [x] Project architecture defined
- [ ] Module skeletons with type-hinted signatures
- [ ] `config.py` — dataclass configs
- [ ] `loader.py` — scan CSV loading and point cloud conversion
- [ ] `registration.py` — flatten, segment, FPFH + ICP registration
- [ ] `deviation.py` — signed distances and metrics
- [ ] `visualization.py` — deviation maps and plots
- [ ] `batch.py` + `cli.py` — batch processing pipeline

## Known Issues / Open Questions

1. **FPFH on thin geometry:** The M letter print is ~0.5mm tall (single layer). FPFH features may lack Z-distinctiveness. May need toolpath-based registration fallback.
2. **Bed segmentation sensitivity:** Hard Z threshold may clip bead edges near the bed interface. RANSAC inlier/outlier mask is more robust but needs validation.
3. **Memory with full-res scans:** ~4.8M points per scan at 0.02mm resolution. Voxel downsampling required for registration; full resolution used for distance computation.
4. **Scan trajectory utility:** The toolpath CSVs (e.g., `m_VBN_05.csv`) describe the scanning trajectory. They provide Y-axis coordinates for raster → point cloud conversion but may not be useful for CAD registration directly.
5. **Volume approximation accuracy:** The point-area × signed-distance approach is an approximation. Accuracy depends on point density and scan coverage relative to the CAD geometry.
