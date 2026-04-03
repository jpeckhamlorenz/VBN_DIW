# Deviation Analysis Pipeline — Methodology Reference

This document explains every algorithm, statistical method, and computational technique used in the deviation analysis pipeline. It is organized by pipeline stage, with each section covering the *what*, *why*, and *how* of each method, along with references for further reading.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Data Loading & Point Cloud Construction](#2-data-loading--point-cloud-construction)
3. [RANSAC Plane Fitting & Bed Flattening](#3-ransac-plane-fitting--bed-flattening)
4. [Bed/Bead Segmentation](#4-beadbead-segmentation)
5. [Anisotropic Smoothing, Island Removal & Sidewalls](#5-anisotropic-smoothing-island-removal--sidewalls)
6. [Floor Augmentation for Thin Geometry](#6-floor-augmentation-for-thin-geometry)
7. [FPFH Feature Descriptors](#7-fpfh-feature-descriptors)
8. [RANSAC Global Registration](#8-ransac-global-registration)
9. [Point-to-Plane ICP Refinement](#9-point-to-plane-icp-refinement)
10. [Signed Distance Computation](#10-signed-distance-computation)
11. [Deviation Metrics](#11-deviation-metrics)
12. [Kolmogorov-Smirnov Test](#12-kolmogorov-smirnov-test)
13. [Kernel Density Estimation (KDE)](#13-kernel-density-estimation-kde)
14. [Visualization & Output Methods](#14-visualization--output-methods)
15. [Caching & Batch Processing](#15-caching--batch-processing)
16. [How Everything Connects](#16-how-everything-connects)
17. [Key Parameters & Sensitivities](#17-key-parameters--sensitivities)
18. [Further Reading](#18-further-reading)

---

## 1. Pipeline Overview

The pipeline answers one question: **"How closely does a 3D-printed part match its intended CAD geometry?"**

It takes two inputs — a laser scan of the printed part and the reference CAD mesh — and produces quantitative metrics and spatial visualizations of geometric deviation. The pipeline runs in sequential stages, each feeding results to the next:

```
Raw laser scan (CSV)          CAD model (STL)
       |                            |
  [Load & rasterize]                |
       |                            |
  [RANSAC plane fit → flatten]      |
       |                            |
  [Segment bed from bead]           |
       |                            |
  [Anisotropic smooth → island removal → add sidewalls]
       |                            |
  [Augment bead with synthetic floor]  ← registration only
       |                            |
  [FPFH features → RANSAC global registration]
       |                            |
  [Point-to-plane ICP refinement]   |
       |                            |
  [Apply transform to real bead points only]
       |                            |
  [Signed distance computation] ----+
       |
  [Metrics: RMSD, MSD, volume, KS test]
       |
  [Visualization: deviation maps, KDE, boxplots, STL export]
```

Each stage is independently cacheable and inspectable.

---

## 2. Data Loading & Point Cloud Construction

### What the raw data looks like

The Keyence laser profiler scans across the part in raster lines. Each scan produces a 2D array (~6000 rows × 800 columns) of raw integer height values. Each row is one raster line (one lateral sweep across the part), and each column is one measurement point along that sweep.

### Raster-to-point-cloud conversion

To turn this 2D height map into a 3D point cloud, we need three coordinates for each measurement:

- **X (cross-scan):** Column index × sensor resolution (0.02 mm per column). This is the lateral position along each raster sweep.
- **Y (scan direction):** Comes from the scanning toolpath CSV, which records the robot's position as it moves the scanner. Since the toolpath has fewer points than the scan has rows, we linearly interpolate (upsample) the toolpath Y-values to match the row count.
- **Z (height):** The raw integer values, scaled by 1/1000 to convert to millimeters.

### Height correction

The scanner head moves along a trajectory that isn't perfectly flat — the robot arm has small Z variations as it traverses. Height correction compensates for this: for each row, we adjust Z by the difference between that row's toolpath Z and the mean toolpath Z. This removes scanner trajectory artifacts from the height data.

### Invalid point handling

The Keyence sensor returns very low values (< 10 raw units) when it can't get a valid reading (e.g., off the edge of the part, specular reflections). These are flagged as NaN and tracked via a boolean `valid_mask` that follows the data through every subsequent stage.

**Result:** ~4.8 million 3D points per scan, with a valid mask indicating which points have reliable height data.

---

## 3. RANSAC Plane Fitting & Bed Flattening

### The problem

The printed part sits on a flat build plate (bed), but the bed may be slightly tilted relative to the scanner's coordinate frame. If we don't correct for this tilt, the bed won't sit at Z = 0, and our height-based segmentation (separating the printed bead from the bed) will be unreliable — one side of the bed might be classified as "bead" while the other side is correct.

### RANSAC (Random Sample Consensus)

**Core idea:** RANSAC is a robust estimation method that fits a model to data containing outliers. Unlike ordinary least squares (which minimizes error across *all* points and gets distorted by outliers), RANSAC:

1. Randomly selects a minimal subset of points (for a plane: 3 points)
2. Fits the model (plane equation) to just those points
3. Counts how many other points agree with this model (are within a distance threshold — these are "inliers")
4. Repeats many times, keeping the model with the most inliers
5. Finally refits the model using *only* the inlier points

**Why RANSAC here:** The point cloud contains both the flat bed (the model we want to fit) and the printed bead on top (outliers relative to the bed plane). Ordinary least squares would try to compromise between the bed and bead, giving a tilted, biased plane. RANSAC automatically ignores the bead points and fits purely to the bed surface.

**Implementation:** We use scikit-learn's `RANSACRegressor` with a linear model (z = ax + by + c). The residual threshold is 0.05 mm — points within 0.05 mm of the fitted plane are considered inliers (bed), points farther away are outliers (bead or noise).

### Rodrigues' rotation

Once we have the bed plane's normal vector, we need to rotate the entire point cloud so the bed becomes horizontal (normal aligned with [0, 0, 1]). The **Rodrigues rotation formula** computes the rotation matrix that maps one unit vector to another:

Given vectors **a** (current plane normal) and **b** (target = [0, 0, 1]):
1. Cross product **v** = **a** × **b** gives the rotation axis
2. Dot product c = **a** · **b** gives the cosine of the rotation angle
3. The rotation matrix is: R = I + [v]× + [v]×² · (1 − c) / |v|²

where [v]× is the skew-symmetric matrix of **v**.

**Result:** All points are rotated so the bed is flat at Z ≈ 0, and then shifted down by the plane's Z-intercept. The bed surface now lies precisely at Z = 0.

**Further reading:** Fischler & Bolles, "Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography," *Communications of the ACM*, 1981.

---

## 4. Bed/Bead Segmentation

### The problem

After flattening, the point cloud contains both the build plate and the printed material. We need to isolate just the bead (printed part) for comparison against the CAD model.

### Method

With the bed flattened to Z = 0, segmentation is straightforward: any point with Z above a threshold (0.15 mm) is classified as "bead" (printed material); anything at or below the threshold is "bed."

The threshold of 0.15 mm is chosen to be:
- Well above the noise floor of the flattened bed surface (~0.02-0.05 mm)
- Well below the minimum bead height (~0.5 mm for a single-layer print)

This provides a clean separation in the typical case. The RANSAC plane fitting in the previous step is what makes this simple threshold work reliably — without flattening first, a tilted bed would cause misclassification along its edges.

**Result:** Two separate point clouds — `bead_points` (the printed material, typically ~1M points) and `bed_points` (the build plate, typically ~3.5M points). Only `bead_points` proceeds to registration and analysis.

---

## 5. Anisotropic Smoothing, Island Removal & Sidewalls

This stage, implemented in `smoothing.py`, post-processes the segmented bead point cloud before registration. All three sub-stages are controlled by `SmoothingConfig` and can be independently toggled.

### 5a. Anisotropic Gaussian Smoothing

#### The problem

During raster scanning, periodic oscillations in the robot arm's traverse speed imprint ridges on the measured surface with a wavelength of ~4 mm along the scan direction (Y-axis). These ridges are scanning artifacts — not real features of the printed bead — and inflate deviation metrics if left uncorrected.

#### Method: NaN-aware normalized convolution on the raster grid

Because the scan data originates as a regular 2D raster (rows × columns), we can exploit the grid structure for fast smoothing rather than using expensive KD-tree ball queries on the 3D point cloud.

The key insight is that smoothing should be **anisotropic** — aggressive along the scan direction (Y, to remove ~4 mm ridges) but minimal perpendicular to it (X, to preserve the bead's cross-sectional profile). The two sigma parameters control this:

- **σ_scan = 0.9 mm** along the scan direction (Y). This smooths out the ~4 mm ridges while preserving larger-scale geometry (corners, width changes). The effective smoothing window is roughly ±3σ ≈ ±2.7 mm.
- **σ_perp = 0.04 mm** perpendicular to scan (X). This is only 2× the column spacing (0.02 mm), so it barely smooths cross-scan — just enough to reduce single-pixel noise without blurring the bead's sidewall edges.

The sigmas are converted from millimeters to grid cells using the known point spacings:

```
sigma_axis0 (rows, Y-direction) = sigma_scan / y_spacing
sigma_axis1 (cols, X-direction) = sigma_perp / x_spacing
```

#### Handling NaN values (normalized convolution)

The raster grid contains NaN entries (invalid sensor readings, points outside the bead mask). Standard Gaussian filtering propagates NaN to all neighbors. The **normalized convolution trick** avoids this:

1. Replace NaN with 0 in the height map, creating `z_filled`
2. Create a binary weight map `w` (1.0 where valid, 0.0 where NaN)
3. Apply the same Gaussian filter to both:
   - `z_smooth = gaussian_filter(z_filled × w, sigma)`
   - `w_smooth = gaussian_filter(w, sigma)`
4. Divide: `z_result = z_smooth / w_smooth`

This effectively computes a weighted average where NaN points contribute zero weight. Points with insufficient valid neighbors (w_smooth ≈ 0) retain their original values.

#### What smoothing does NOT change

- **XY coordinates** are untouched — only Z heights are modified
- **Bead outline/mask** is unchanged — smoothing operates within the existing valid region
- **Macro geometry** (corners, width) is preserved because σ_perp is very small

**Parameters are identical across all methods** (VBN, static ideal, static naive) to ensure fair comparison. No per-scan tuning.

### 5b. Island Removal

#### The problem

After bed/bead segmentation, the bead mask may contain small disconnected clusters of points — stray droplets, sensor noise spikes, or bed artifacts that happened to exceed the Z threshold. These "islands" add noise to registration and metrics.

#### Method: connected-component labeling on the 2D grid

Using `scipy.ndimage.label` with 4-connected structure on the 2D bead mask:

1. Label each connected component (group of adjacent True pixels in the mask)
2. Count the size of each component
3. Keep only the largest component (the main bead body)
4. Remove all other components from both the mask and the point array

This is efficient because it operates on the 2D raster grid rather than doing 3D clustering on the point cloud. Typical results: ~90–100 small islands removed, totaling ~1,000–2,000 points out of ~975,000.

### 5c. Vertical Sidewall Generation

#### The problem

The laser scan only captures the top surface of the bead. When comparing to the CAD mesh (which is a closed solid with vertical side faces), scan points near the bead edges have large signed distances to the nearest CAD sidewall face. This inflates deviation metrics at the edges even when the top surface matches well.

#### Method: edge detection + vertical point columns

1. **Edge detection:** Apply binary erosion (4-connected structuring element) to the 2D bead mask. Edge pixels are those in the original mask but not in the eroded mask — i.e., pixels that border a non-bead region.

2. **Vertical columns:** For each edge pixel, create a column of points from the bead surface height down to the floor plane (Z = 0) at a spacing of `sidewall_z_step` (default 0.1 mm). Each point shares the edge pixel's XY coordinates.

3. **Augmentation:** The sidewall points are appended to the bead point array. All subsequent stages (registration, distance computation) operate on the combined surface + sidewall cloud.

#### STL export with sidewalls

Point clouds with sidewalls cannot be faithfully represented by 2D Delaunay triangulation (which collapses vertical geometry). The `build_sidewalled_mesh()` function in `loader.py` constructs the mesh explicitly from three components:

- **Top surface:** 2D Delaunay triangulation of bead surface points (XY projection), with long-edge filtering
- **Sidewall quads:** For each pair of adjacent edge pixels (right or bottom neighbor), two triangles connecting their vertical columns form a quad strip
- **Floor:** 2D Delaunay triangulation of edge pixel XY positions at Z = 0, with long-edge filtering

The final mesh is exported via Open3D (`compute_vertex_normals` + `write_triangle_mesh`).

### Configuration

All parameters live in `SmoothingConfig` (in `config.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `True` | Master toggle for the entire smoothing stage |
| `sigma_scan` | 0.9 mm | Gaussian sigma along scan direction |
| `sigma_perp` | 0.04 mm | Gaussian sigma perpendicular to scan |
| `n_iterations` | 1 | Number of smoothing passes |
| `scan_direction` | (0, 1, 0) | Unit vector of scan direction |
| `use_smoothed` | `True` | If False, smoothing runs and exports STLs but pipeline uses raw points |
| `remove_islands` | `True` | Enable connected-component island removal |
| `add_sidewalls` | `True` | Enable vertical sidewall point generation |
| `sidewall_z_step` | 0.1 mm | Vertical spacing between sidewall points |

### Validation

`validate_smoothing.py` runs the pipeline twice (raw vs. smoothed) on the same scan and compares:
- RMSD, MSD, MAE, 95th percentile metrics
- KDE distribution overlays
- KS test for statistical difference
- Point-wise |d_smoothed − d_raw| statistics

---

## 6. Floor Augmentation for Thin Geometry

### The problem

The laser scan only captures the *top* surface of the printed bead. The CAD mesh, however, is a closed solid with a top face, a bottom face, and side edges. This mismatch causes a fundamental difficulty for FPFH feature matching: when computing descriptors on the scan, almost every point has the same neighborhood geometry — flat surface, normal pointing straight up. There is very little to distinguish one part of the scan from another, resulting in very few valid feature correspondences (Open3D warns "Too few correspondences after mutual filter").

### Solution: synthetic floor points

After segmentation, we project the bead's XY footprint down to Z = 0 (the bed level) to create a synthetic bottom face:

```
bead_points:   original top-surface scan points   Z ≈ 0.2–0.6 mm
floor_points:  same XY, forced to Z = 0           Z = 0.0 mm
augmented:     vstack([bead_points, floor_points]) ~2M points
```

The augmented cloud is then used for FPFH feature computation, RANSAC global registration, and ICP refinement. After voxel downsampling, the boundary between the top surface (normals up) and the floor (normals down) creates sharp edges with highly distinctive FPFH signatures — exactly the geometric contrast needed for reliable feature matching. Corners of the "M" become especially distinctive since two perpendicular edges meet there.

### Why the floor points are valid for registration but not for metrics

The floor points represent a reasonable physical assumption: the bottom of the bead really does contact the bed at Z = 0. They are geometrically faithful enough to improve the alignment transform.

However, they are intentionally excluded from distance computation and deviation metrics. Including them would artificially deflate RMSD, MAE, and other metrics, because each floor point would have ~0 signed distance (it sits exactly where the CAD bottom face is). With ~1M synthetic points alongside ~1M real measurements, every metric would roughly halve — not because the print improved, but because we injected "perfect" data. The bottom surface is also physically unobservable by the laser (it's pressed into the bed), so it may have real defects (elephant foot, adhesion artifacts) that we cannot measure.

**The separation is clean:** `register_scan_to_cad()` accepts `bead_points`, uses the augmented cloud internally for registration, and returns only the 4×4 transform. The caller applies that transform to the original `bead_points` for all downstream work.

**Config toggle:** `RegistrationConfig.augment_with_floor = True` (default). Set to `False` to compare registration quality with and without augmentation.

---

## 7. FPFH Feature Descriptors

### The problem

We have a scan point cloud (bead) and a CAD mesh, and they're in completely different coordinate frames. We need to align them. But to align them, we first need to know which points in the scan correspond to which points on the CAD — a chicken-and-egg problem. Feature descriptors solve this by giving each point a "signature" that describes its local geometry, so we can find likely correspondences without knowing the alignment.

### What FPFH is

**FPFH (Fast Point Feature Histograms)** is a 33-dimensional descriptor that encodes the local surface geometry around each point. It works as follows:

1. **Normal estimation first:** For each point, find its k nearest neighbors (within a radius) and fit a local plane to them. The plane's normal vector describes the surface orientation at that point. We use a search radius of 1.0 mm and up to 30 neighbors.

2. **Simplified Point Feature Histogram (SPFH):** For each point p and each of its neighbors, compute three angular features (α, φ, θ) that describe the relative orientation of their normal vectors. Bin these angles into histograms (11 bins each, 33 total). This captures *how the surface curves* around point p.

3. **Fast extension (the "F" in FPFH):** Re-weight each point's SPFH by incorporating the SPFHs of its neighbors, weighted by distance. This extends the descriptive reach without the O(n·k²) cost of full PFH.

**Why FPFH works for us:** Even though the scan and CAD are in different coordinate frames, the *local surface geometry* (curvature, flatness, edges) is the same. A point at a corner of the "M" will have a similar FPFH descriptor in both the scan and the CAD, regardless of their relative position/orientation. This lets us find candidate correspondences.

### Voxel downsampling

FPFH is computed on a downsampled version of the point cloud (voxel size = 0.5 mm) rather than the full ~1M-point cloud. Voxel downsampling divides space into a regular 3D grid and replaces all points within each voxel with their centroid. This reduces the point count by roughly 100× while preserving the overall shape, making feature computation tractable.

**Result:** Each downsampled point gets a 33-dimensional FPFH vector. These are used in the next stage to find correspondences.

**Further reading:** Rusu, Blodow, & Beetz, "Fast Point Feature Histograms (FPFH) for 3D Registration," *IEEE ICRA*, 2009.

---

## 8. RANSAC Global Registration

### The problem

We have FPFH features for both the scan and the CAD, and we can find candidate point correspondences by matching similar features. But many of these correspondences will be wrong (especially for symmetric or repeated geometry). We need to find the rigid transformation (rotation + translation) that is consistent with the *correct* correspondences while ignoring the wrong ones.

### Why RANSAC again

This is the same RANSAC philosophy as the plane fitting, but applied to transformation estimation instead:

1. Randomly select 3 point correspondences (the minimum needed to determine a rigid transformation in 3D)
2. Compute the rigid transformation from those 3 pairs
3. Apply that transformation and count how many *other* correspondences agree (their distance after transformation is below a threshold)
4. Repeat millions of times, keeping the best transformation

**Key parameters:**
- **Distance threshold:** 0.75 mm (= voxel_size × 1.5). A correspondence is an "inlier" if the transformed source point is within 0.75 mm of its matched target point.
- **Max iterations:** 4,000,000 — high because the inlier ratio may be low (many false feature matches)
- **Confidence:** 0.999 — allows early termination when a good solution is found with high probability

### Correspondence filtering

Two additional checks reject geometrically inconsistent correspondences before RANSAC even evaluates them:

1. **Edge length consistency (0.9):** If points A and B are matched to A' and B', the distance |AB| should be similar to |A'B'| (since rigid transforms preserve distances). The ratio must be > 0.9.
2. **Distance check:** After transformation, matched points must be within the distance threshold.

**Output:** A 4×4 homogeneous transformation matrix that coarsely aligns the scan to the CAD. This gets the scan into roughly the right position/orientation but isn't precise enough for distance computation.

**Note on "too few correspondences" warning:** The Open3D warning about falling back from mutual filter means that very few FPFH matches survived the strict mutual-nearest-neighbor filter. This was a significant problem before floor augmentation was added — with only the thin top surface, almost every point looked like "flat surface, normal up." The synthetic floor (§5) resolves this by adding geometric contrast at the bead perimeter edges, producing many more valid correspondences.

**Further reading:** Same as RANSAC above (Fischler & Bolles, 1981), applied in the context of Rusu et al.'s feature-based registration framework.

---

## 9. Point-to-Plane ICP Refinement

### The problem

RANSAC global registration gets us close (within a few mm), but its accuracy is limited by the downsampled point clouds and the discrete nature of feature matching. We need sub-millimeter precision for meaningful deviation measurements.

### ICP (Iterative Closest Point)

ICP is the workhorse algorithm for fine alignment of point clouds. The basic idea:

1. For each point in the source cloud, find its closest point in the target cloud
2. Compute the transformation that minimizes the distances between these pairs
3. Apply that transformation to the source
4. Repeat until convergence

**The "local minimum" problem:** ICP only converges to the correct alignment if the initial guess is already close. If you start too far away, ICP will find a wrong local minimum. This is exactly why we need RANSAC global registration first — it provides a good enough initial alignment for ICP to converge to the right answer.

### Point-to-point vs. point-to-plane

There are two main ICP variants:

- **Point-to-point:** Minimizes the sum of squared distances between corresponding points. Converges slowly, especially for flat surfaces where many correspondences are ambiguous.

- **Point-to-plane:** Minimizes the sum of squared distances between each source point and the *tangent plane* at its corresponding target point. The error for each pair is: (p_source − p_target) · **n_target**, where **n** is the surface normal at the target point.

**Why point-to-plane is better here:** Our geometry (a flat-ish letter printed on a flat bed) has large planar regions. Point-to-point ICP can "slide" along these planes without converging, because moving a point along a plane doesn't change its distance to the nearest point. Point-to-plane ICP uses the normal vector to penalize any deviation from the surface, even tangential sliding, so it converges much faster and more accurately.

**Parameters:**
- **Max correspondence distance:** 0.1 mm — pairs farther apart than this are rejected. This is tight because we expect the RANSAC result to already be close.
- **Max iterations:** 200 — typically converges in 30-50 iterations.

**Output:** A refined 4×4 transformation matrix, plus two quality metrics:
- **Fitness:** Fraction of source points that found a correspondence within the threshold (0.0–1.0). Higher is better. Values of 0.25–0.35 are typical for our data because the scan covers more area than the CAD (the scan includes material beyond the CAD boundary, like excess extrusion at corners).
- **Inlier RMSE:** Root-mean-square error of the inlier correspondences. This tells you how well the *matched* points agree. Values of ~0.05 mm indicate excellent alignment.

**Further reading:** Chen & Medioni, "Object Modelling by Registration of Multiple Range Images," *Image and Vision Computing*, 1992. Rusinkiewicz & Levoy, "Efficient Variants of the ICP Algorithm," *3DIM*, 2001.

---

## 10. Signed Distance Computation

### The problem

Now that the scan is aligned to the CAD, we need to quantify the deviation at every point. For each scan point, we ask: "How far is this point from the nearest CAD surface, and is it *inside* or *outside* the intended geometry?"

### Closest-point query

For each of the ~1M aligned bead points, we find the closest point on the CAD mesh surface. This is implemented by trimesh using a BVH (Bounding Volume Hierarchy) tree for O(log n) queries per point. The result is:
- The closest point on the mesh surface
- The Euclidean distance to that point
- The ID of the mesh triangle containing the closest point

### Sign determination via normal dot product

The unsigned distance tells us *how far* a point is from the surface but not *which side* it's on. The sign encodes the physical meaning:

- **Positive (outside):** Material exists where the CAD says there should be none → over-extrusion
- **Negative (inside):** No material where the CAD says there should be → under-extrusion / gap

To determine the sign:
1. Compute the vector **v** from the closest surface point to the scan point
2. Get the outward-facing normal **n** of the triangle at the closest point
3. Compute the dot product: d = **v** · **n**
4. If d > 0, the scan point is on the outside (same direction as the outward normal) → positive
5. If d < 0, the scan point is on the inside → negative

This assumes the CAD mesh has consistently outward-facing normals, which is standard for well-formed STL files.

**Result:** An array of signed distances (one per bead point), in millimeters. This is the core data from which all metrics and visualizations are derived.

---

## 11. Deviation Metrics

Each metric captures a different aspect of print quality:

### RMSD (Root Mean Square Deviation)

```
RMSD = √( (1/N) Σ dᵢ² )
```

**What it measures:** Overall magnitude of deviations, regardless of direction. Sensitive to large outliers (because of the squaring). This is the primary "how good is the print" number.

**Interpretation:** An RMSD of 0.25 mm means that, on average (in a root-mean-square sense), each point deviates from the CAD surface by about 0.25 mm. Since typical bead widths are ~1 mm, this represents about 25% of a bead width.

### MSD (Mean Signed Deviation)

```
MSD = (1/N) Σ dᵢ
```

**What it measures:** Systematic bias. Positive MSD indicates net over-extrusion (the part is bigger than intended); negative indicates net under-extrusion. If MSD ≈ 0 but RMSD is large, the errors are symmetric — the part has both over- and under-extrusion that cancel out on average.

**Interpretation:** An MSD of +0.30 mm (like Static ideal) means the part is systematically 0.30 mm too thick. This makes physical sense: a static nozzle tuned for steady-state flow will over-extrude wherever the nozzle decelerates (corners, U-turns) because the flow doesn't ramp down to match.

### MAE (Mean Absolute Error)

```
MAE = (1/N) Σ |dᵢ|
```

**What it measures:** Average magnitude of deviation, without the squaring of RMSD. Less sensitive to extreme outliers than RMSD. If MAE ≈ RMSD, the deviations are fairly uniform; if RMSD >> MAE, there are a few large outliers dominating.

### 95th Percentile Absolute Deviation

```
P95 = the value below which 95% of |dᵢ| fall
```

**What it measures:** A robust worst-case metric. "95% of all surface points are within P95 mm of the CAD surface." Less sensitive to the single worst outlier (which might be a scan artifact) than the maximum.

**Interpretation:** A P95 of 0.49 mm (VBN) vs. 1.43 mm (Static ideal) means VBN keeps 95% of its surface within half a millimeter, while Static ideal has almost 3× worse worst-case performance.

### Maximum Absolute Deviation

```
max|d| = max(|dᵢ|)
```

**What it measures:** The single worst deviation anywhere on the part. Useful for identifying problem spots but sensitive to noise/outliers.

### Excess and Deficit Volume

```
V_excess = Σ dᵢ · A    for all dᵢ > 0
V_deficit = Σ |dᵢ| · A  for all dᵢ < 0
```

where A = point_area = resolution × slice_thickness = 0.02 × 0.005 = 0.0001 mm².

**What it measures:** The total volume of material that is outside (excess) or missing from (deficit) the intended geometry. Physically meaningful — you can say "there is 8.87 mm³ of excess material."

**The approximation:** Each scan point represents a small patch of surface with area A. The volume contribution of each point is its signed distance times that area (like a thin column of material above or below the CAD surface). This is an approximation because:
- The actual area per point varies with surface orientation
- Points aren't uniformly spaced after registration
- It doesn't account for overhanging geometry

But for a flat, single-layer print scanned from above, this approximation is reasonable.

---

## 12. Kolmogorov-Smirnov Test

### What it is

The two-sample KS test is a nonparametric statistical test that compares two distributions. It asks: **"Could these two samples have been drawn from the same underlying probability distribution?"**

### How it works

1. Compute the empirical cumulative distribution function (ECDF) of each sample
2. Find the maximum vertical distance between the two ECDFs:
   ```
   D = max |F₁(x) − F₂(x)|
   ```
3. The D-statistic quantifies how different the distributions are (0 = identical, 1 = completely non-overlapping)
4. The p-value gives the probability of observing a D-statistic this large (or larger) if the two samples truly came from the same distribution

### Why we use it

We want to claim that different printing methods produce *statistically significantly different* print quality. Showing that VBN has a lower RMSD than Static isn't enough for a rigorous paper — a reviewer could argue the difference is within normal variability. The KS test provides formal statistical evidence that the deviation distributions are genuinely different, not just noisy draws from the same distribution.

### Interpreting the results

- **D-statistic = 0.28 (VBN vs. Static ideal):** The ECDFs differ by up to 28% at their most separated point. This is a substantial difference.
- **p ≈ 0:** The probability of seeing this much difference by chance is essentially zero → the distributions are significantly different.

### Caveat: statistical power with large N

With ~1M points per scan, the KS test has enormous statistical power. It will detect even tiny, practically irrelevant differences and report them as "significant." This is why the **D-statistic magnitude** matters more than the p-value for your purposes. A D of 0.28 is meaningful; a D of 0.02 with p < 0.001 would be statistically significant but practically irrelevant.

**Further reading:** Massey, "The Kolmogorov-Smirnov Test for Goodness of Fit," *Journal of the American Statistical Association*, 1951.

---

## 13. Kernel Density Estimation (KDE)

### What it is

KDE is a method for estimating the probability density function (PDF) of a random variable from a finite sample. It's the smooth-curve version of a histogram.

### How it works

Place a small Gaussian "bump" (kernel) at each data point, then sum all the bumps:

```
f̂(x) = (1 / Nh) Σ K((x − xᵢ) / h)
```

where K is the Gaussian kernel and h is the **bandwidth** (controls smoothness).

- **Small bandwidth:** Follows every data point closely → noisy, spiky estimate
- **Large bandwidth:** Over-smooths → loses detail, biases toward a simple shape
- **Bandwidth selection:** scipy uses Scott's rule (h = N^(-1/5) · σ) by default, which balances bias and variance for approximately Gaussian data.

### Why KDE instead of histograms

- KDE produces smooth curves that overlay cleanly when comparing multiple distributions on one plot
- No bin-edge artifacts (histogram shape depends on where bins start)
- More visually interpretable for publication figures

### Subsampling for performance

With ~1M points, KDE computation becomes slow. The pipeline subsamples to 100,000 points (randomly selected with a fixed seed for reproducibility) before fitting the KDE. With 100k points, the density estimate is already highly accurate — you don't gain meaningful precision from the remaining 900k points.

### What the KDE plot shows

The x-axis is signed deviation (mm), the y-axis is probability density. Each curve represents one method's distribution of deviations. A curve centered on zero with a tall, narrow peak indicates a method with low bias and high precision (like VBN). A curve shifted to the right indicates systematic over-extrusion (like Static ideal).

---

## 14. Visualization & Output Methods

### Spatial Deviation Maps

These are the 3D color-mapped renderings showing *where* on the part deviations occur (not just their distribution).

**Method:**
1. Load the CAD mesh as a PyVista surface
2. Create a point cloud from the aligned scan points with signed deviation as a scalar field
3. Interpolate deviation values from the scan points onto the CAD mesh surface using Shepard's method (inverse-distance-weighted interpolation)
4. Render the mesh with a diverging colormap (RdBu_r: red = positive/excess, blue = negative/deficit, white = nominal)

**Shepard's interpolation:** For each mesh point, the interpolated value is a weighted average of nearby scan point values, where weights decay with distance (1/d^p, with p controlling the decay rate). The "sharpness" parameter controls how quickly the influence drops off.

**Colormap choice:** RdBu_r (reversed Red-Blue) is a perceptually uniform diverging colormap, appropriate for signed data centered on zero. The symmetric color limits (e.g., ±0.3 mm) ensure visual fairness when comparing methods.

### Box-and-Whisker Plots

Standard statistical summary:
- **Box:** 25th to 75th percentile (interquartile range, IQR)
- **Bold line:** Median (50th percentile)
- **Whiskers:** Extend to the most extreme data point within 1.5 × IQR of the box
- **Outliers:** Hidden (with ~1M points, there are many outliers that would clutter the plot)

The dashed line at zero shows the "perfect" baseline. A box centered on zero = unbiased; a tall box = high variability.

### Metrics Summary Table

A matplotlib-rendered table formatted for direct inclusion in a paper or supplementary materials. Shows all numerical metrics in a compact tabular format.

### STL Export of Scan Point Clouds

The pipeline can export the segmented bead point cloud as STL mesh files at multiple stages:

- **Raw bead (`*_raw_bead.stl`):** Bead points after segmentation, before smoothing. Useful for inspecting raw scan quality.
- **Smoothed bead (`*_smoothed_bead.stl`):** Bead points after anisotropic smoothing (top surface only, no sidewalls).
- **Sidewalled bead (`*_sidewalled_bead*.stl`):** Full mesh with top surface, vertical sidewalls, and floor — built with explicit mesh construction via `build_sidewalled_mesh()`.
- **Aligned bead (`*_bead_aligned.stl`):** Bead points after the FPFH + ICP transform has been applied, in the CAD coordinate frame. Can be directly overlaid with the CAD STL in any 3D viewer (MeshLab, Blender, etc.) to visually verify registration quality.

**Top-surface meshing — 2D Delaunay triangulation:** Since the scan data is a 2.5D height map (one Z value per XY location), we triangulate on the XY plane and use Z as vertex height. This is more appropriate than Poisson reconstruction or ball-pivot algorithms, which are designed for true 3D point clouds with arbitrary surface orientations.

**Delaunay triangulation** finds the triangulation of a 2D point set that maximizes the minimum angle of all triangles (avoiding very thin, elongated triangles). Formally, it guarantees that no point lies inside the circumcircle of any triangle. On a regular or near-regular grid like our raster scan, this produces a clean mesh of approximately equilateral triangles.

**Long-edge filtering (`max_edge_length`):** The Delaunay triangulation will connect all points in the convex hull, including long triangles that span the gaps between strokes of the "M." The `max_edge_length` parameter (default 0.5 mm, about 25× the scan resolution) removes any triangle with an edge longer than this threshold, preventing spurious spanning faces across physically separate bead strokes.

**Sidewalled mesh construction (`build_sidewalled_mesh`):** 2D Delaunay cannot represent vertical geometry (same XY, different Z). The sidewalled STL is built explicitly from three components: Delaunay top surface, quad-strip sidewalls between adjacent edge pixels, and a Delaunay floor at Z = 0. The final mesh is exported via Open3D.

---

## 15. Caching & Batch Processing

### NPZ Caching

The pipeline caches intermediate results as compressed NumPy archives (.npz) at three stages:

1. **After flattening:** Saves the flattened point cloud, valid mask, rotation matrix, and Z intercept
2. **After registration:** Saves the 4×4 transformation matrix, fitness, and RMSE
3. **After distance computation:** Saves signed distances and aligned bead points

**Cache invalidation:** If the source file (scan CSV) is newer than the cache file (based on filesystem modification times), the cache is considered stale and recomputed. This prevents using outdated results when input data is updated.

**Smoothing-aware cache keys:** Cache filenames encode the smoothing state so that raw and smoothed runs don't collide. The tag format is `_sm{sigma_scan}_{sigma_perp}_{n_iter}_{ri/nori}_{sw/nosw}` for smoothed runs, or `_raw` for unsmoothed runs. Changing any smoothing parameter automatically invalidates the relevant caches.

**Why cache?** Registration (FPFH + RANSAC + ICP) takes 30-60 seconds per scan. Signed distance computation on 1M points takes another 10-20 seconds. Caching lets you re-run visualization and metric extraction in seconds without redoing expensive computation.

### Batch Processing

The batch system iterates over multiple methods and scans, running the full pipeline for each, then:
- Aggregates metrics across all methods into a summary table
- Runs pairwise KS tests between all methods
- Generates comparison figures (KDE overlay, boxplots, deviation maps)

---

## 16. How Everything Connects

Here's the complete data flow with method connections:

```
                    INPUTS
                    ======
  scan_cycle.csv (6000×800 raw heights)
  toolpath.csv (scanning trajectory XYZ)
  m_ideal.stl (CAD reference mesh)

                      |
           ┌──────────┴──────────┐
           │    DATA LOADING      │
           │  loader.py           │
           │                      │
           │  • Scale raw→mm      │
           │  • Flag invalid pts  │
           │  • Interpolate Y     │
           │    from toolpath     │
           │  • Height-correct Z  │
           │  • Build XYZ grid    │
           └──────────┬──────────┘
                      │
              points (4.8M, 3)
              valid_mask (4.8M,)
                      │
           ┌──────────┴──────────┐
           │   BED FLATTENING     │
           │  registration.py     │
           │                      │
           │  • RANSAC plane fit  │
           │    (sklearn)         │
           │  • Rodrigues rotate  │
           │  • Shift Z to zero   │
           └──────────┬──────────┘
                      │
              flat_points (4.8M, 3)
              bed at Z=0
                      │
           ┌──────────┴──────────┐
           │   SEGMENTATION       │
           │  registration.py     │
           │                      │
           │  • Z > 0.15mm → bead │
           │  • Z ≤ 0.15mm → bed  │
           └──────────┬──────────┘
                      │
              bead_points (~1M, 3)
                      │
           ┌──────────┴──────────┐
           │  SMOOTHING & CLEANUP │
           │  smoothing.py        │
           │                      │
           │  • Anisotropic 2D    │
           │    Gaussian on grid  │
           │    (σ_scan=0.9mm,    │
           │     σ_perp=0.04mm)   │
           │  • NaN-aware norm.   │
           │    convolution       │
           │  • Island removal    │
           │    (connected comp.) │
           │  • Sidewall gen.     │
           │    (edge→Z=0 cols)   │
           └──────────┬──────────┘
                      │
              bead_points (~1M+W, 3)  ← kept for metrics
                      │
           ┌──────────┴──────────┐
           │  FLOOR AUGMENTATION  │
           │  registration.py     │
           │                      │
           │  • Copy bead XY      │
           │  • Set Z = 0 (bed)   │
           │  • Stack top+bottom  │
           │  • Registration only │
           └──────────┬──────────┘
                      │
              augmented (~2M, 3)    ← registration only
                      │
           ┌──────────┴──────────────────────────────┐
           │   COARSE REGISTRATION                    │
           │  registration.py                         │
           │                                          │
           │  Scan side:              CAD side:       │
           │  • Voxel downsample      • Poisson-disk  │
           │    (0.5 mm)                sample mesh   │
           │  • Estimate normals      • Get normals   │
           │    (KDTree, r=1mm)         from mesh     │
           │  • Compute FPFH          • Compute FPFH  │
           │    (r=2.5mm, 33-dim)       (same params) │
           │         │                      │         │
           │         └───── Match features ─┘         │
           │                     │                    │
           │         RANSAC on correspondences        │
           │         (3 pts, 4M iters, 0.999 conf)    │
           │                     │                    │
           │         Coarse 4×4 transform             │
           └──────────┬──────────────────────────────-┘
                      │
           ┌──────────┴──────────┐
           │   FINE REGISTRATION  │
           │  registration.py     │
           │                      │
           │  • Point-to-plane    │
           │    ICP (Open3D)      │
           │  • Uses normals for  │
           │    tangent-plane     │
           │    error metric      │
           │  • 0.1mm threshold   │
           │  • ≤200 iterations   │
           └──────────┬──────────┘
                      │
              Refined 4×4 transform
              fitness, inlier RMSE
                      │
           ┌──────────┴──────────┐
           │  APPLY TRANSFORM     │
           │                      │
           │  aligned_bead =      │
           │    T × bead_points   │
           │  (real points only,  │
           │   floor discarded)   │
           └──────────┬──────────┘
                      │
              aligned_bead (~1M, 3)
              now in CAD coordinate frame
                      │
           ┌──────────┴──────────────────────────────┐
           │   SIGNED DISTANCE                        │
           │  deviation.py                            │
           │                                          │
           │  For each aligned point:                 │
           │  • BVH closest-point query (trimesh)     │
           │  • Get distance d and triangle ID        │
           │  • Get face normal n at closest triangle │
           │  • sign = dot(point−closest, n) > 0      │
           │    → +1 (outside/excess)                 │
           │    → −1 (inside/deficit)                 │
           │  • signed_dist = d × sign                │
           └──────────┬──────────────────────────────-┘
                      │
              signed_distances (~1M,)
                      │
        ┌─────────────┼──────────────────────┐
        │             │                      │
  ┌─────┴─────┐ ┌────┴────┐  ┌─────────────┴──────────────┐
  │  METRICS   │ │  KS     │  │ VISUALIZATION & EXPORT      │
  │            │ │  TEST   │  │                             │
  │  RMSD      │ │         │  │ • Spatial deviation maps    │
  │  MSD       │ │ Compare │  │   (pyvista, RdBu_r)         │
  │  MAE       │ │ pairs   │  │ • KDE overlay (scipy+mpl)   │
  │  95th %    │ │ of      │  │ • Boxplots (matplotlib)     │
  │  Max       │ │ methods │  │ • Metrics table PNG + CSV   │
  │  Volumes   │ │ (scipy) │  │ • STL mesh export           │
  └────────────┘ └─────────┘  │   (Delaunay, loader.py)    │
                               └────────────────────────────┘
```

---

## 17. Key Parameters & Sensitivities

| Parameter | Value | What it controls | Sensitivity |
|-----------|-------|-----------------|-------------|
| `resolution` | 0.02 mm | X-axis point spacing | Fixed by sensor hardware |
| `slice_thickness` | 0.005 mm | Y-axis point spacing (scan_speed / scan_rate) | Fixed by scan settings |
| `invalid_threshold` | 10 | Raw values below this → NaN | Low — only affects sensor noise floor |
| `plane_ransac_residual` | 0.05 mm | Inlier distance for bed plane fit | Moderate — too large captures bead base; too small misses tilted bed regions |
| `bed_z_threshold` | 0.15 mm | Bead/bed segmentation cutoff | **High** — directly controls what counts as "printed material." Too low → includes bed noise. Too high → clips bead edges near the bed. |
| `fpfh_voxel_size` | 0.5 mm | Downsampling for feature computation | Moderate — smaller = more features but slower. Larger = fewer features, coarser alignment |
| `fpfh_radius_feature` | 2.5 mm | Neighborhood for feature computation | Moderate — should be ~5× voxel size. Larger captures more context but loses local detail |
| `ransac_max_iterations` | 4M | Iterations for global registration | Low — more iterations = more likely to find correct alignment but slower |
| `icp_threshold` | 0.1 mm | Max correspondence distance for ICP | **High** — too small rejects valid correspondences. Too large allows wrong matches to influence alignment |
| `augment_with_floor` | True | Add synthetic floor for registration | Low on metrics (floor points never used for distances). Significant effect on registration quality for thin geometry — disabling it risks poor FPFH matches |
| `sigma_scan` | 0.9 mm | Gaussian sigma along scan direction | **High** — must be large enough to cover ~4 mm ridge wavelength (effective window ±3σ ≈ ±2.7 mm) but not so large as to blur corners. |
| `sigma_perp` | 0.04 mm | Gaussian sigma perpendicular to scan | **High** — must stay small relative to bead width (~1 mm). Values > ~0.1 mm will visibly blur the bead cross-section and destroy sidewall detail. |
| `remove_islands` | True | Drop disconnected small components | Low — typically removes ~1K points out of ~975K. Improves registration robustness. |
| `add_sidewalls` | True | Add vertical sidewall points | Moderate — improves edge deviation metrics by giving the scan geometry vertical faces to match against the CAD sidewalls. |
| `sidewall_z_step` | 0.1 mm | Vertical spacing of sidewall points | Low — smaller values add more points (denser walls) but with diminishing returns. 0.1 mm is ~2× the bead Z noise floor. |
| `max_edge_length` (STL) | 0.5 mm | Long-edge filter for STL meshing | Moderate — controls which spanning triangles are removed. Too small removes valid triangles at sharp corners; too large keeps artifacts across bead gaps |
| `clim` | ±0.3 mm | Color scale for deviation maps | Visual only — affects interpretation. Should be consistent across all methods for fair comparison |
| `percentile` | 95% | Which percentile for worst-case metric | Low — 95% is standard; 99% would be more conservative |

---

## 18. Further Reading

### Point Cloud Registration
- **ICP:** Besl & McKay, "A Method for Registration of 3-D Shapes," *IEEE PAMI*, 1992.
- **Point-to-Plane ICP:** Chen & Medioni, "Object Modelling by Registration of Multiple Range Images," *Image and Vision Computing*, 1992.
- **ICP variants survey:** Rusinkiewicz & Levoy, "Efficient Variants of the ICP Algorithm," *3DIM*, 2001.
- **FPFH:** Rusu, Blodow, & Beetz, "Fast Point Feature Histograms (FPFH) for 3D Registration," *IEEE ICRA*, 2009.

### Robust Estimation
- **RANSAC:** Fischler & Bolles, "Random Sample Consensus," *Communications of the ACM*, 1981.

### Statistical Methods
- **KS Test:** Massey, "The Kolmogorov-Smirnov Test for Goodness of Fit," *JASA*, 1951.
- **KDE:** Silverman, *Density Estimation for Statistics and Data Analysis*, Chapman & Hall, 1986.

### Deviation Analysis in Additive Manufacturing
- **GD&T for AM:** Colosimo et al., "From Profile to Surface Monitoring," *Journal of Quality Technology*, 2018.
- **Scan-to-CAD comparison:** Senin et al., "Characterisation of the topography of metal additively manufactured parts," *Surface Topography*, 2017.

### Software Documentation
- **Open3D:** http://www.open3d.org/docs/ — Registration pipeline tutorials
- **trimesh:** https://trimesh.org/ — Proximity queries and mesh operations
- **PyVista:** https://docs.pyvista.org/ — 3D visualization and interpolation
- **scikit-learn RANSAC:** https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html
