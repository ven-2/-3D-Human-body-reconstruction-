"""
fit_star.py

Fits the STAR parametric body model to:
  1. Averaged 3D body joints (from on-device Vision pose estimation,
     back-projected using ARKit depth + intrinsics)
  2. A cleaned point cloud sampled from the TSDF mesh

Two-stage optimization:
  - Stage 1: align STAR joints to the observed skeleton (Adam, ~600 iters)
  - Stage 2: freeze pose, refine shape (betas) + translation against the
    scan surface using a KD-tree Chamfer loss (~400-600 iters)

Outputs:
  - star_fit.ply / star_tpose.ply  — fitted mesh (posed / T-pose)
  - star_params.json               — shape, pose, translation params
  - star_measurements.json         — height, chest/waist/hip circumference
                                      (via convex-hull cross-sections),
                                      inseam, shoulder width

Implementation withheld — see docs/star_fitting.md for the approach.
"""

def fit_star_to_session(*args, **kwargs):
    raise NotImplementedError("Implementation not included in this repo.")
