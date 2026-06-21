# AR RGB-D Reconstruction & Body Model Fitting

A real-time RGB-D perception pipeline that captures depth, color, segmentation
and body-joint data from an iPhone (ARKit + LiDAR), streams it to a Python
backend, and reconstructs a clean 3D human body mesh 

## Pipeline overview

```
iPhone (ARKit + LiDAR)
  ├─ Depth + confidence map (LiDAR)
  ├─ Person segmentation mask
  ├─ Camera intrinsics + pose (ARKit world transform)
  └─ On-device body joint detection (Vision), back-projected to 3D
        │
        ▼  (streamed over HTTP, multipart per-frame)
Python backend (FastAPI)
  ├─ TSDF volumetric fusion (Open3D) of depth frames -> dense mesh
  ├─ Marching cubes surface extraction
  ├─ Mesh cleaning (largest component, floor removal, skeleton-guided crop,
  │   statistical outlier removal, light smoothing)
  └─ STAR parametric body model fit
        ├─ Stage 1: align body joints to detected skeleton
        ├─ Stage 2: refine shape against the cleaned scan surface (Chamfer loss)
        └─ Derive body measurements (height, chest, waist, hip, inseam, etc.)
```

## Repo layout

```
ios/        SwiftUI / ARKit / SceneKit capture app
backend/    FastAPI server + reconstruction pipeline
  server.py        receives frames, runs TSDF fusion, orchestrates pipeline
  mesh_cleaner.py  isolates and cleans the body mesh from the raw TSDF output
  fit_star.py      fits the STAR body model and derives measurements
models/star/       (not included — see below)
```

## iOS app (`ios/`)

- **ARManager** — owns the ARKit session, drives per-frame processing, and
  renders a live SceneKit point cloud + joint-sphere overlay.
- **PoseTracker** — runs Vision's body pose detector on-device, back-projects
  2D joints to 3D world space using the depth map and camera intrinsics, and
  accumulates a per-joint average across the capture session.
- **PointCloud** — builds a downsampled, person-segmented colour point cloud
  in world space for live preview / export.
- **FrameUploader** — streams depth, confidence, intrinsics, and camera pose
  to the backend at a throttled frame rate.
- **PLYFile** — exports the live point cloud as a `.ply` for sharing.

Set `backendBaseURL` in `ARManager.swift` to your backend machine's address
before running.

## Backend (`backend/`)

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

### Endpoints

| Endpoint              | Purpose                                              |
|-----------------------|-------------------------------------------------------|
| `POST /frame`         | Receive one depth+pose frame, integrate into TSDF     |
| `POST /pose`          | Receive averaged 3D body joints (`pose.json`)         |
| `POST /stop`          | Extract mesh, clean it, and run STAR fitting          |
| `GET  /status`        | Poll whether the result mesh is ready                 |
| `GET  /result`        | Download the cleaned mesh                             |
| `GET  /star_mesh`     | Download the fitted STAR body mesh                    |
| `GET  /star_params`   | Raw STAR shape/pose parameters                        |
| `GET  /star_measurements` | Derived body measurements (height, chest, waist, hip, inseam, shoulder width) |

### STAR body model

This project uses [STAR](https://star.is.tue.mpg.de/), a parametric human
body model. `fit_star.py` is included as a **documented stub** — the fitting
implementation itself is withheld, but the approach (two-stage joint +
surface optimization, derived measurements, etc.) is described in
[`docs/star_fitting.md`](docs/star_fitting.md).

The model weights are **not included** either way — they're distributed
under their own license by the Max Planck Institute. Download
`STAR_NEUTRAL.npz` from the link above and place it at:

```
models/star/STAR_NEUTRAL.npz
```

## Notes

- Coordinate frame handling (ARKit world space, camera intrinsics layout,
  extrinsic conventions for Open3D's TSDF integration) is one of the trickier
  parts of this pipeline — see the comments in `server.py` around
  `make_extrinsic` / `EXTR_MODE`.
- Skeleton-guided cropping (`mesh_cleaner.py`) uses the averaged 3D joints to
  remove background geometry (walls, floor, other people) before sampling a
  clean point cloud for body-model fitting.
- `fit_star.py` runs a two-stage fit (joint alignment, then shape-only
  surface refinement) — see [`docs/star_fitting.md`](docs/star_fitting.md)
  for details on why it's structured this way.

## License

MIT — see [LICENSE](LICENSE). STAR model weights are subject to their own
license from the Max Planck Institute for Intelligent Systems.

