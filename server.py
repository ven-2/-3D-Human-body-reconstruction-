# server.py
#
# FastAPI backend that receives ARKit depth + pose frames from the iOS app
# and fuses them into a 3D mesh using an Open3D TSDF volume.
#
# Pipeline:
#   iOS -> /frame (depth + intrinsics + camera pose) -> TSDF integration
#   iOS -> /pose  (averaged 3D body joints)
#   iOS -> /stop  -> extract mesh -> clean mesh -> fit STAR body model
#
# Run:
#   pip install fastapi uvicorn numpy open3d python-multipart torch scipy
#   uvicorn server:app --host 0.0.0.0 --port 8000
#
# iPhone docs:
#   http://<PC_IP>:8000/docs

from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import json
import threading
import time
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np
import open3d as o3d

from mesh_cleaner import clean_mesh_and_sample, load_averaged_joints
from fit_star import fit_star_to_session

app = FastAPI()
BASE = Path("sessions")
BASE.mkdir(exist_ok=True)

# Path to STAR neutral model .npz
# Download from https://star.is.tue.mpg.de/
STAR_MODEL_PATH = "models/star/STAR_NEUTRAL.npz"

# -----------------------------
# Debug toggles
# -----------------------------

DEBUG = False                     # set True for verbose per-frame diagnostics
DEBUG_PRINT_EVERY_N = 10          # print heavy logs every N frames (if DEBUG)
DUMP_PCD_EVERY_N = 0              # dump a PCD every N frames for offline inspection (0 = off)
START_LIVE_VIEWER = False         # open an Open3D live point-cloud window while capturing
LIVE_VIEW_MODE = "pcd"            # "pcd" or "mesh"

# How the camera extrinsic is derived from the camera->world transform (T_wc)
# sent by ARKit. World_to_camera = inv(T_wc), optionally with an axis flip
# to match Open3D's TSDF integration convention.
FLIP_X  = np.diag([-1.0,  1.0,  1.0, 1.0])
FLIP_Y  = np.diag([ 1.0, -1.0,  1.0, 1.0])
FLIP_Z  = np.diag([ 1.0,  1.0, -1.0, 1.0])
FLIP_XY = np.diag([-1.0, -1.0,  1.0, 1.0])
FLIP_XZ = np.diag([-1.0,  1.0, -1.0, 1.0])
FLIP_YZ = np.diag([ 1.0, -1.0, -1.0, 1.0])

FLIP       = FLIP_YZ   # axis flip applied to the extrinsic
FLIP_SPACE = "cam"      # "cam" (left-multiply) or "world" (right-multiply)

# How to derive the camera extrinsic (world_to_camera) from T_wc:
#   "INV_T_WC"       -> extr = inv(T_wc)
#   "INV_T_WC_FLIPZ" -> extr = inv(T_wc) with FLIP applied (default, matches
#                       Open3D's TSDF convention for ARKit's coordinate frame)
#   "IDENTITY"       -> assumes a static camera (debugging only)
EXTR_MODE = "INV_T_WC_FLIPZ"

# -----------------------------
# Utilities: decoding stride-safe
# -----------------------------

def decode_depth_float32_stride(depth_bytes: bytes, w: int, h: int, bytes_per_row: int) -> np.ndarray:
    raw = np.frombuffer(depth_bytes, dtype=np.uint8)
    expected = bytes_per_row * h
    if raw.size != expected:
        raise ValueError(f"Depth size mismatch: got {raw.size}, expected {expected} (bpr={bytes_per_row}, h={h})")

    row_payload = w * 4  # float32
    depth = np.empty((h, w), dtype=np.float32)
    for y in range(h):
        start = y * bytes_per_row
        row = raw[start:start + row_payload]
        depth[y, :] = np.frombuffer(row, dtype=np.float32, count=w)
    return depth


def decode_conf_u8_stride(conf_bytes: bytes, w: int, h: int, bytes_per_row: int) -> np.ndarray:
    raw = np.frombuffer(conf_bytes, dtype=np.uint8)
    expected = bytes_per_row * h
    if raw.size != expected:
        raise ValueError(f"Conf size mismatch: got {raw.size}, expected {expected} (bpr={bytes_per_row}, h={h})")

    conf = np.empty((h, w), dtype=np.uint8)
    for y in range(h):
        start = y * bytes_per_row
        conf[y, :] = raw[start:start + w]
    return conf


def make_o3d_intrinsics(meta: dict) -> o3d.camera.PinholeCameraIntrinsic:
    K_raw = np.array(meta["intrinsics"], dtype=np.float64)
    K = _fix_K_layout(K_raw)

    fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])

    w = int(meta["depth_width"])
    h = int(meta["depth_height"])
    img_w = float(meta.get("image_resolution", {}).get("w", w))
    img_h = float(meta.get("image_resolution", {}).get("h", h))

    if img_w > 0 and img_h > 0 and (img_w != w or img_h != h):
        sx = w / img_w
        sy = h / img_h
        fx *= sx; cx *= sx
        fy *= sy; cy *= sy

    if DEBUG:
        print("RAW K from meta:", K_raw)
        print("K used:", K)
    return o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)



def get_T_wc(meta: dict) -> np.ndarray:
    T_raw = np.array(meta["transform"], dtype=np.float64)
    T = _fix_T_layout(T_raw)
    if T.shape != (4, 4):
        raise ValueError(f"Transform must be 4x4, got {T.shape}")
    return T


def _fix_K_layout(K: np.ndarray) -> np.ndarray:
    # If cx,cy show up in [2,0],[2,1] instead of [0,2],[1,2], transpose it.
    if abs(K[0, 2]) < 1e-6 and abs(K[1, 2]) < 1e-6 and (abs(K[2, 0]) > 1e-3 or abs(K[2, 1]) > 1e-3):
        return K.T
    return K

def _fix_T_layout(T: np.ndarray) -> np.ndarray:
    # If translation seems to be in last row instead of last column, transpose it.
    t_col = np.linalg.norm(T[:3, 3])
    t_row = np.linalg.norm(T[3, :3])
    if t_col < 1e-9 and t_row > 1e-6:
        return T.T
    return T

def make_extrinsic(meta: dict) -> np.ndarray:
    T_wc = get_T_wc(meta)
    base = np.linalg.inv(T_wc)   # world_to_camera

    if EXTR_MODE == "INV_T_WC":
        return base

    if EXTR_MODE == "INV_T_WC_FLIPZ":
        if FLIP_SPACE == "cam":
            return FLIP @ base
        else:  # "world"
            return base @ FLIP

    if EXTR_MODE == "IDENTITY":
        return np.eye(4)

    raise ValueError(f"Unknown EXTR_MODE: {EXTR_MODE}")



def create_tsdf_volume(voxel_length=0.01, sdf_trunc=0.04, with_color=False):
    color_type = (
        o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        if with_color else
        o3d.pipelines.integration.TSDFVolumeColorType.NoColor
    )
    return o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=color_type,
    )


def depth_stats(depth: np.ndarray) -> Tuple[float, float, float]:
    valid = np.isfinite(depth) & (depth > 0.05) & (depth < 5.0)
    if valid.any():
        mn = float(np.nanmin(depth[valid]))
        mx = float(np.nanmax(depth[valid]))
        pct = float(valid.mean() * 100.0)
        return mn, mx, pct
    return float("nan"), float("nan"), 0.0


def sanity_check_pose(T_wc: np.ndarray) -> dict:
    R = T_wc[:3, :3]
    t = T_wc[:3, 3]
    det = float(np.linalg.det(R))
    ortho_err = float(np.linalg.norm(R.T @ R - np.eye(3)))
    return {
        "det(R)": det,
        "ortho_err": ortho_err,
        "t": [float(t[0]), float(t[1]), float(t[2])],
    }


def dump_frame_pcd(out_dir: Path, frame_index: int, depth: np.ndarray,
                   intr: o3d.camera.PinholeCameraIntrinsic,
                   extr: np.ndarray) -> None:
    """
    Create a point cloud from JUST this depth frame + pose and save it.
    This is your #1 tool to confirm whether pose/intrinsics are correct.
    """
    depth_o3d = o3d.geometry.Image(depth.astype(np.float32))

    dummy_color = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    color_o3d = o3d.geometry.Image(dummy_color)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=1.0,
        depth_trunc=2.0,
        convert_rgb_to_intensity=False,
    )

    # NOTE: create_from_rgbd_image expects extrinsic as camera pose or world? depends on API;
    # Open3D uses extrinsic as world_to_camera in many pipelines.
    # We'll instead create pcd in camera coords and transform ourselves for clarity.
    pcd_cam = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)  # camera coords
    # convert camera->world using inv(extr) (since extr = world_to_camera)
    T_wc = np.linalg.inv(extr)
    pcd_world = pcd_cam.transform(T_wc)

    pcd_path = out_dir / f"frame_pcd_{frame_index:06d}.ply"
    o3d.io.write_point_cloud(str(pcd_path), pcd_world)


# -----------------------------
# Session state
# -----------------------------

@dataclass
class SessionState:
    tsdf: o3d.pipelines.integration.ScalableTSDFVolume
    lock: threading.Lock
    ready: bool = False
    result_path: Optional[str] = None
    last_frame_index: int = -1
    integrate_count: int = 0
    last_T_wc: Optional[np.ndarray] = None


SESSIONS: Dict[str, SessionState] = {}


def get_session(session_id: str) -> SessionState:
    st = SESSIONS.get(session_id)
    if st is None:
        st = SessionState(
            tsdf=create_tsdf_volume(voxel_length=0.01, sdf_trunc=0.04, with_color=False),
            lock=threading.Lock(),
        )
        SESSIONS[session_id] = st
    return st


# -----------------------------
# Live viewer (optional)
# -----------------------------

@dataclass
class ViewerState:
    thread: threading.Thread
    stop_flag: threading.Event

VIEWERS: Dict[str, ViewerState] = {}


def start_live_viewer_if_needed(session_id: str, st: SessionState, mode: str = "pcd") -> None:
    if not START_LIVE_VIEWER:
        return
    if session_id in VIEWERS:
        return

    stop_flag = threading.Event()

    def loop():
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"Live [{session_id[:8]}] {mode}", width=960, height=720)
        geom_added = False
        last_geom = None

        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.mesh_show_back_face = True

        while not stop_flag.is_set():
            with st.lock:
                if mode == "mesh":
                    geom = st.tsdf.extract_triangle_mesh()
                    geom.compute_vertex_normals()
                else:
                    geom = st.tsdf.extract_point_cloud()

            empty = (len(geom.vertices) == 0) if mode == "mesh" else (len(geom.points) == 0)
            if empty:
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.15)
                continue

            if not geom_added:
                vis.add_geometry(geom)
                geom_added = True
                last_geom = geom
            else:
                if mode == "mesh":
                    last_geom.vertices = geom.vertices
                    last_geom.triangles = geom.triangles
                    last_geom.vertex_normals = geom.vertex_normals
                else:
                    last_geom.points = geom.points
                    last_geom.colors = geom.colors
                vis.update_geometry(last_geom)

            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.25)

        vis.destroy_window()

    t = threading.Thread(target=loop, daemon=True)
    VIEWERS[session_id] = ViewerState(thread=t, stop_flag=stop_flag)
    t.start()

def keep_largest_connected_component(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """
    Removes small floating blobs by keeping only the largest triangle cluster.
    """
    if len(mesh.triangles) == 0:
        return mesh

    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    if cluster_n_triangles.size == 0:
        return mesh

    largest = int(np.argmax(cluster_n_triangles))
    triangles_to_remove = triangle_clusters != largest
    mesh.remove_triangles_by_mask(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    return mesh
# -----------------------------
# TSDF integration
# -----------------------------

def integrate_into_tsdf(st: SessionState, meta: dict, frame_index: int,
                        depth_bytes: bytes, conf_bytes: Optional[bytes]) -> None:
    w = int(meta["depth_width"])
    h = int(meta["depth_height"])
    bpr = int(meta["depth_bytes_per_row"])

    depth = decode_depth_float32_stride(depth_bytes, w, h, bpr)

    # Optional confidence: ARKit typically gives 0/1/2
    if conf_bytes is not None and bool(meta.get("conf_present", False)):
        cw = int(meta.get("conf_width", w))
        ch = int(meta.get("conf_height", h))
        cbpr = int(meta.get("conf_bytes_per_row", cw))
        conf = decode_conf_u8_stride(conf_bytes, cw, ch, cbpr)

        if conf.shape != depth.shape:
            yy = (np.linspace(0, conf.shape[0] - 1, depth.shape[0])).astype(np.int32)
            xx = (np.linspace(0, conf.shape[1] - 1, depth.shape[1])).astype(np.int32)
            conf = conf[yy][:, xx]

        # ARKit confidence typically: 0=low, 1=medium, 2=high
        high = int(conf.max())  # expect 2
        depth = np.where(conf == high, depth, 0.0).astype(np.float32)

    mn, mx, pct = depth_stats(depth)

    intr = make_o3d_intrinsics(meta)
    extr = make_extrinsic(meta)  # world_to_camera
    T_wc = get_T_wc(meta)

    # Debug prints
    if DEBUG and (st.integrate_count < 5 or (st.integrate_count % DEBUG_PRINT_EVERY_N == 0)):
        pose_info = sanity_check_pose(T_wc)

        # motion delta check
        delta = None
        if st.last_T_wc is not None:
            dt = float(np.linalg.norm(T_wc[:3, 3] - st.last_T_wc[:3, 3]))
            delta = dt

        pp = intr.get_principal_point()
        print("\n=== INTEGRATE ===")
        print("frame:", frame_index, "count:", st.integrate_count, "EXTR_MODE:", EXTR_MODE)
        print("depth: min", mn, "max", mx, "valid%", pct)
        print("intr: w,h =", intr.width, intr.height, "fx,fy,cx,cy =",
              intr.get_focal_length()[0], intr.get_focal_length()[1],
              pp[0], pp[1])
        print("pose:", pose_info, "delta_t(m):", delta)

        # quick check: extr should be inverse of T_wc (for INV_T_WC* modes)
        if EXTR_MODE.startswith("INV_T_WC"):
            check = np.linalg.norm(extr @ T_wc - np.eye(4))
            print("|| extr @ T_wc - I || =", float(check))

    # Optionally dump a standalone point cloud for this frame — useful for
    # offline verification of pose/intrinsics alignment in MeshLab/CloudCompare.
    if DUMP_PCD_EVERY_N > 0 and (frame_index % DUMP_PCD_EVERY_N == 0):
        out_dir = BASE / meta.get("session_id_for_dump", "tmp")
        out_dir.mkdir(parents=True, exist_ok=True)
        dump_frame_pcd(out_dir, frame_index, depth, intr, extr)
        if DEBUG:
            print("Dumped frame PCD:", str(out_dir / f"frame_pcd_{frame_index:06d}.ply"))

    # Build RGBD (dummy color)
    depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
    dummy_color = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    color_o3d = o3d.geometry.Image(dummy_color)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=1.0,
        depth_trunc=2.0,
        convert_rgb_to_intensity=False,
    )

    # Integrate ONCE
    st.tsdf.integrate(rgbd, intr, extr)

    st.integrate_count += 1
    st.last_frame_index = max(st.last_frame_index, int(frame_index))
    st.last_T_wc = T_wc


def extract_mesh_job(session_id: str) -> None:
    st = get_session(session_id)
    out_dir = BASE / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_ply_path = out_dir / "result_mesh.ply"

    with st.lock:
        mesh = st.tsdf.extract_triangle_mesh()

    if len(mesh.vertices) == 0:
        print("[TSDF] Mesh empty; not writing.")
        st.ready = False
        st.result_path = None
        return

    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(raw_ply_path), mesh)
    print("[TSDF] Wrote raw mesh:", str(raw_ply_path))

    # ---- Clean mesh using pose skeleton as guide ----
    pose_path = out_dir / "pose.json"
    averaged_joints = []
    if pose_path.exists():
        try:
            averaged_joints = load_averaged_joints(pose_path)
            print(f"[TSDF] Loaded {len(averaged_joints)} averaged joints for cleaning.")
        except Exception as e:
            print(f"[TSDF] Warning: could not load pose.json for cleaning: {e}")
    else:
        print("[TSDF] Warning: no pose.json found — skeleton-guided crop will be skipped.")

    try:
        clean_pcd, clean_mesh = clean_mesh_and_sample(
            mesh,
            averaged_joints,
            out_dir=out_dir,
            keep_n_components=1,
            floor_margin_m=0.05,
            body_radius_m=0.45,
            body_height_pad_m=0.15,
            outlier_nb_neighbors=20,
            outlier_std_ratio=2.0,
            laplacian_iters=1,
            sample_points=50_000,
        )
        print(f"[cleaner] Clean mesh: {len(clean_mesh.vertices)} verts")
        print(f"[cleaner] Clean pcd:  {len(clean_pcd.points)} points")
        st.ready       = True
        st.result_path = str(out_dir / "cleaned_mesh.ply")
    except Exception as e:
        print(f"[cleaner] Cleaning failed ({e}); falling back to raw mesh.")
        st.ready       = True
        st.result_path = str(raw_ply_path)

    # ---- Fit STAR body model ----
    _run_star_fitting(session_id, out_dir)



def _run_star_fitting(session_id: str, out_dir: Path) -> None:
    """
    Runs STAR fitting in a background thread after mesh cleaning completes.
    Failures are non-fatal — the scan result is already saved independently.
    """
    def _job():
        print(f"[STAR] Starting fitting for session {session_id[:8]}...")
        try:
            result = fit_star_to_session(
                out_dir,
                model_path=STAR_MODEL_PATH,
                stage1_iters=600,
                stage2_iters=500,
            )
            if result["ok"]:
                m = result["measurements"]
                print(f"[STAR] Fitting complete. "
                      f"Height: {m.get('height_cm')}cm  "
                      f"Chest: {m.get('chest_circ_cm')}cm  "
                      f"Waist: {m.get('waist_circ_cm')}cm  "
                      f"Hip: {m.get('hip_circ_cm')}cm")
                print("\n[STAR] ===== FIT COMPLETE =====")
                print("[STAR] Mesh path:", Path(result["star_mesh_path"]).resolve())
                print("[STAR] Params path:", (out_dir / "star_params.json").resolve())
                print("[STAR] Measurements path:", (out_dir / "star_measurements.json").resolve())
                print("[STAR] =========================\n")
            else:
                print(f"[STAR] Fitting failed: {result['error']}")
        except Exception as e:
            print(f"[STAR] Unexpected error during fitting: {e}")

    t = threading.Thread(target=_job, daemon=True)
    t.start()


# -----------------------------
# API
# -----------------------------

@app.post("/reset")
async def reset_session(session_id: str = Form(...)):
    # blow away session (fresh TSDF)
    SESSIONS.pop(session_id, None)
    return {"ok": True, "reset": session_id}

@app.post("/pose")
async def upload_pose(
    session_id: str = Form(...),
    pose_json: UploadFile = File(...),
):
    """Receive the on-device pose landmarks JSON and store it next to the TSDF session.

    iPhone (Swift) should POST multipart:
      - session_id (form field)
      - pose_json (file field, application/json)

    We store:
      sessions/<session_id>/pose.json
    """
    out_dir = BASE / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pose.json"

    try:
        data = await pose_json.read()
        out_path.write_bytes(data)
        return {"ok": True, "session_id": session_id, "bytes": len(data), "path": str(out_path)}
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"failed to write pose.json: {e}"}, status_code=500)


@app.post("/frame")
async def upload_frame(
    session_id: str = Form(...),
    frame_index: int = Form(...),
    meta_json: str = Form(...),
    depth_bin: UploadFile = File(...),
    conf_bin: UploadFile = File(None),
):
    st = get_session(session_id)
    start_live_viewer_if_needed(session_id, st, mode=LIVE_VIEW_MODE)

    try:
        meta = json.loads(meta_json)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"bad meta_json: {e}"}, status_code=400)

    # tag session id so dump_frame_pcd uses correct folder
    meta["session_id_for_dump"] = session_id

    depth_bytes = await depth_bin.read()
    conf_bytes = await conf_bin.read() if conf_bin is not None else None

    out_dir = BASE / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optionally persist raw inputs for offline debugging — disabled by
    # default, as this writes two files per frame.
    if DEBUG:
        (out_dir / f"meta_{frame_index:06d}.json").write_text(json.dumps(meta), encoding="utf-8")
        (out_dir / f"depth_{frame_index:06d}.bin").write_bytes(depth_bytes)
        if conf_bytes is not None:
            (out_dir / f"conf_{frame_index:06d}.bin").write_bytes(conf_bytes)

    try:
        with st.lock:
            integrate_into_tsdf(st, meta, int(frame_index), depth_bytes, conf_bytes)
            st.ready = False
            st.result_path = None
    except Exception as e:
        print("INTEGRATE FAILED:", repr(e))
        return JSONResponse({"ok": False, "error": f"integrate failed: {e}"}, status_code=500)

    return {"ok": True, "integrated_frames": st.integrate_count}


@app.post("/stop")
async def stop_session(session_id: str = Form(...)):
    st = get_session(session_id)
    with st.lock:
        st.ready = False
        st.result_path = None

    t = threading.Thread(target=extract_mesh_job, args=(session_id,), daemon=True)
    t.start()
    return {"ok": True, "session_id": session_id, "processing": True}


@app.get("/status")
async def status(session_id: str):
    st = SESSIONS.get(session_id)
    if not st:
        return JSONResponse({"ready": False, "error": "unknown session"}, status_code=404)
    return {"ready": st.ready, "integrated_frames": st.integrate_count, "last_frame_index": st.last_frame_index}


@app.get("/result")
async def result(session_id: str):
    st = SESSIONS.get(session_id)
    if not st or not st.ready or not st.result_path:
        return JSONResponse({"ready": False}, status_code=404)
    return FileResponse(st.result_path, media_type="application/octet-stream", filename="result_mesh.ply")

# ------------------------------------------------------------------
# STAR body model results
# ------------------------------------------------------------------

@app.get("/star_measurements")
async def get_star_measurements(session_id: str):
    """
    Returns derived body measurements in cm after STAR fitting completes.

    Example response:
      {
        "ready": true,
        "measurements": {
          "height_cm": 178.2,
          "chest_circ_cm": 98.4,
          "waist_circ_cm": 84.1,
          "hip_circ_cm": 101.2,
          "inseam_cm": 81.3,
          "shoulder_width_cm": 44.6,
          "torso_length_cm": 52.1
        }
      }

    Returns {"ready": false} if fitting is still running or failed.
    """
    path = BASE / session_id / "star_measurements.json"
    if not path.exists():
        return JSONResponse({"ready": False,
                             "message": "STAR fitting not complete yet. "
                                        "Check back in ~30 seconds after /stop."})
    try:
        data = json.loads(path.read_bytes())
        return JSONResponse({"ready": True, "measurements": data})
    except Exception as e:
        return JSONResponse({"ready": False, "error": str(e)}, status_code=500)


@app.get("/star_params")
async def get_star_params(session_id: str):
    """
    Returns the raw STAR shape/pose parameters.
    Use these to re-render the fitted body or simulate clothing.
    """
    path = BASE / session_id / "star_params.json"
    if not path.exists():
        return JSONResponse({"ready": False, "message": "STAR params not available yet."})
    try:
        data = json.loads(path.read_bytes())
        return JSONResponse({"ready": True, "params": data})
    except Exception as e:
        return JSONResponse({"ready": False, "error": str(e)}, status_code=500)


@app.get("/star_mesh")
async def get_star_mesh(session_id: str):
    """
    Download the fitted STAR body mesh as a PLY file.
    This is the clean, watertight 1:1 body model for this user.
    """
    path = BASE / session_id / "star_fit.ply"
    if not path.exists():
        return JSONResponse({"ready": False, "message": "STAR mesh not available yet."})
    return FileResponse(str(path), media_type="application/octet-stream",
                        filename="star_body.ply")