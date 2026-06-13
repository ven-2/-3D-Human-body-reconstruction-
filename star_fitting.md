# Body Model Fitting (STAR)

`backend/fit_star.py` is included as a documented stub — the implementation
is withheld, but this page describes the approach.

## Inputs

- **Averaged 3D joints** (`pose.json`) — per-joint mean world position,
  accumulated on-device from Vision body pose detection back-projected
  through the LiDAR depth map (see `docs/pose_estimation.md`).
- **Cleaned point cloud** (`clean_pcd.ply`) — sampled from the cropped,
  outlier-filtered TSDF mesh (see `mesh_cleaner.py`).

## Output

- A fitted mesh in both the captured pose and a canonical T-pose
- The underlying shape/pose/translation parameters (useful for downstream
  re-rendering or simulation)
- Derived body measurements: height, chest/waist/hip circumference, inseam,
  and shoulder width, computed from the T-pose mesh so they're independent
  of the captured pose. Circumferences are estimated by taking a horizontal
  cross-section of the mesh at the relevant joint height and measuring the
  perimeter of its convex hull — more robust to body shape variation than a
  simple ellipse approximation from two diameters.

## Why this approach

A single LiDAR scan from one viewpoint gives partial, noisy surface coverage
and no ground-truth body shape. Anchoring the fit to skeletal joints first
gives a stable starting point that's robust to missing surface data (e.g.
occluded back of the body), and the surface refinement stage then lets the
model's shape parameters absorb whatever real geometry was captured —
without the optimizer over-fitting pose to compensate for clothing or
missing data.
