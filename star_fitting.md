# Body Model Fitting (STAR)

`backend/fit_star.py` is included as a documented stub — the implementation
is withheld, but this page describes the approach.

## Inputs

- **Averaged 3D joints** (`pose.json`) — per-joint mean world position,
  accumulated on-device from Vision body pose detection back-projected
  through the LiDAR depth map (see `docs/pose_estimation.md`).
- **Cleaned point cloud** (`clean_pcd.ply`) — sampled from the cropped,
  outlier-filtered TSDF mesh (see `mesh_cleaner.py`).

## Two-stage optimization

**Stage 1 — skeleton alignment.**
STAR's pose, shape (betas), global orientation and translation parameters
are initialized near the observed pelvis position and optimized so that the
model's regressed joint positions match the observed skeleton. Joints are
weighted unevenly — core joints (pelvis, hips, shoulders) are trusted more
than extremities, which are noisier from a single-viewpoint phone scan.

**Stage 2 — shape refinement against the scan surface.**
Once the skeleton roughly matches, pose is frozen and only shape (betas) and
translation continue to optimize, against the cleaned point cloud using a
nearest-neighbour (Chamfer-style) surface loss. Freezing pose here is
deliberate: letting pose continue to vary against a clothed/noisy scan
surface tends to cause the model to "cheat" by bending limbs to chase
clothing bulges rather than adjusting actual body shape.

A light regularization term keeps shape parameters and pose deviations from
drifting to extreme, unrealistic values.

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
