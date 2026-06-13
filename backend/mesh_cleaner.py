# mesh_cleaner.py
#
# Cleans the raw TSDF output mesh to isolate the human body,
# then samples a clean point cloud from it for SMPL/STAR fitting.
#
# Pipeline:
#   1. Remove floating blobs (largest connected component)
#   2. Floor removal (remove points below estimated floor plane)
#   3. Skeleton-guided bounding crop (remove anything far from the person)
#   4. Statistical outlier removal (remove sparse noise)
#   5. Smooth (optional, light Laplacian)
#   6. Sample point cloud from cleaned mesh
#
# Usage (standalone):
#   python mesh_cleaner.py --session sessions/<uuid> --joints sessions/<uuid>/pose.json
#
# Usage (from server.py):
#   from mesh_cleaner import clean_mesh_and_sample
#   clean_pcd, clean_mesh = clean_mesh_and_sample(mesh, averaged_joints, session_dir)
#

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------------
# Public entry point (called from server.py)
# ---------------------------------------------------------------------------

def clean_mesh_and_sample(
    mesh: o3d.geometry.TriangleMesh,
    averaged_joints: list[dict],           # list of {name, x, y, z, ...} dicts
    out_dir: Optional[Path] = None,        # if set, writes cleaned_mesh.ply + clean_pcd.ply
    *,
    # Tuning knobs — safe defaults for a single person at 1-2m
    keep_n_components: int = 1,            # how many connected components to keep
    floor_margin_m: float = 0.05,          # remove points this far below lowest joint
    body_radius_m: float = 0.45,           # horizontal crop radius around skeleton
    body_height_pad_m: float = 0.15,       # extra height above/below skeleton AABB
    outlier_nb_neighbors: int = 20,        # for statistical outlier removal
    outlier_std_ratio: float = 2.0,
    laplacian_iters: int = 1,              # 0 = no smoothing; 1-3 = light smoothing
    sample_points: int = 50_000,           # point cloud density for SMPL fitting
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]:
    """
    Returns (clean_point_cloud, clean_mesh).
    clean_point_cloud is what you feed into the Chamfer loss during SMPL fitting.
    clean_mesh is what you export as the final avatar surface.
    """

    if len(mesh.vertices) == 0:
        raise ValueError("Input mesh is empty.")

    joints_xyz = _parse_joints(averaged_joints)

    # ---- Step 1: Keep largest N connected components ----
    mesh = _keep_n_components(mesh, keep_n_components)
    print(f"[cleaner] After component filter: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

    # ---- Step 2: Floor removal ----
    if joints_xyz is not None:
        mesh = _remove_floor(mesh, joints_xyz, margin=floor_margin_m)
        print(f"[cleaner] After floor removal: {len(mesh.vertices)} verts")

    # ---- Step 3: Skeleton bounding crop ----
    if joints_xyz is not None:
        mesh = _crop_to_skeleton(mesh, joints_xyz,
                                 radius=body_radius_m,
                                 height_pad=body_height_pad_m)
        print(f"[cleaner] After skeleton crop: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

    # ---- Step 4: Statistical outlier removal on vertices ----
    # Convert to pcd temporarily, filter, then remove verts from mesh
    mesh = _outlier_filter_mesh(mesh,
                                nb_neighbors=outlier_nb_neighbors,
                                std_ratio=outlier_std_ratio)
    print(f"[cleaner] After outlier filter: {len(mesh.vertices)} verts")

    # ---- Step 5: Light smoothing (optional) ----
    if laplacian_iters > 0:
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=laplacian_iters)
        mesh.compute_vertex_normals()

    # ---- Step 6: Sample point cloud from cleaned mesh ----
    pcd = mesh.sample_points_poisson_disk(number_of_points=sample_points)
    print(f"[cleaner] Sampled point cloud: {len(pcd.points)} points")

    # ---- Optional: write outputs ----
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        mesh_path = out_dir / "cleaned_mesh.ply"
        pcd_path  = out_dir / "clean_pcd.ply"

        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        o3d.io.write_point_cloud(str(pcd_path), pcd)
        print(f"[cleaner] Wrote {mesh_path}")
        print(f"[cleaner] Wrote {pcd_path}")

    return pcd, mesh


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------

def _keep_n_components(mesh: o3d.geometry.TriangleMesh,
                       n: int) -> o3d.geometry.TriangleMesh:
    """Keep the N largest connected triangle clusters."""
    if len(mesh.triangles) == 0:
        return mesh

    clusters, n_tris, _ = mesh.cluster_connected_triangles()
    clusters  = np.asarray(clusters)
    n_tris    = np.asarray(n_tris)

    # Get indices of the N largest clusters, sorted descending
    sorted_idx = np.argsort(n_tris)[::-1]
    keep_clusters = set(sorted_idx[:n].tolist())

    remove_mask = np.array([c not in keep_clusters for c in clusters])
    mesh.remove_triangles_by_mask(remove_mask)
    mesh.remove_unreferenced_vertices()
    return mesh


def _remove_floor(mesh: o3d.geometry.TriangleMesh,
                  joints_xyz: np.ndarray,
                  margin: float) -> o3d.geometry.TriangleMesh:
    """
    Remove vertices below the lowest observed joint minus a small margin.
    ARKit Y is up, so we threshold on Y.

    If your world has a different up axis, change the axis index here.
    """
    UP_AXIS = 1  # Y is up in ARKit world space

    floor_y = float(joints_xyz[:, UP_AXIS].min()) - margin

    verts = np.asarray(mesh.vertices)
    keep  = verts[:, UP_AXIS] >= floor_y

    return _filter_vertices(mesh, keep)


def _crop_to_skeleton(mesh: o3d.geometry.TriangleMesh,
                      joints_xyz: np.ndarray,
                      radius: float,
                      height_pad: float) -> o3d.geometry.TriangleMesh:
    """
    Remove vertices that are far from the skeleton in the horizontal plane (XZ),
    and outside a padded height range in Y.

    This eliminates background walls, floor patches, and other people.
    """
    UP_AXIS = 1  # Y up

    verts = np.asarray(mesh.vertices)  # (N, 3)

    # ---- Height filter (Y axis) ----
    min_y = float(joints_xyz[:, UP_AXIS].min()) - height_pad
    max_y = float(joints_xyz[:, UP_AXIS].max()) + height_pad
    in_height = (verts[:, UP_AXIS] >= min_y) & (verts[:, UP_AXIS] <= max_y)

    # ---- Horizontal distance filter (XZ plane) ----
    # For each vertex, compute min distance to any joint projected onto XZ.
    # Vectorised: (N_verts, 1, 2) vs (1, N_joints, 2)
    horiz_axes = [i for i in range(3) if i != UP_AXIS]  # [0, 2] = X, Z
    verts_xz  = verts[:, horiz_axes]                    # (N, 2)
    joints_xz = joints_xyz[:, horiz_axes]               # (M, 2)

    # Squared distances: (N, M)
    diff      = verts_xz[:, None, :] - joints_xz[None, :, :]   # (N, M, 2)
    dist_sq   = (diff ** 2).sum(axis=2)                         # (N, M)
    min_dist  = np.sqrt(dist_sq.min(axis=1))                    # (N,)

    in_radius = min_dist <= radius

    keep = in_height & in_radius
    return _filter_vertices(mesh, keep)


def _outlier_filter_mesh(mesh: o3d.geometry.TriangleMesh,
                         nb_neighbors: int,
                         std_ratio: float) -> o3d.geometry.TriangleMesh:
    """
    Uses Open3D's statistical outlier removal on the vertex point cloud,
    then removes those vertices from the mesh.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))

    _, inlier_idx = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )

    inlier_set = set(inlier_idx)
    n_verts    = len(mesh.vertices)
    keep       = np.array([i in inlier_set for i in range(n_verts)])

    return _filter_vertices(mesh, keep)


def _filter_vertices(mesh: o3d.geometry.TriangleMesh,
                     keep_mask: np.ndarray) -> o3d.geometry.TriangleMesh:
    """
    Remove vertices where keep_mask is False, and remove any triangle
    that references a removed vertex.
    """
    keep_idx   = np.where(keep_mask)[0]
    remove_idx = np.where(~keep_mask)[0]

    if len(remove_idx) == 0:
        return mesh

    # Build old→new index remap
    remap = np.full(len(keep_mask), -1, dtype=np.int64)
    remap[keep_idx] = np.arange(len(keep_idx), dtype=np.int64)

    # Filter triangles: keep only those where ALL three vertices survive
    tris    = np.asarray(mesh.triangles)
    if len(tris) == 0:
        return mesh

    new_v0  = remap[tris[:, 0]]
    new_v1  = remap[tris[:, 1]]
    new_v2  = remap[tris[:, 2]]
    tri_ok  = (new_v0 >= 0) & (new_v1 >= 0) & (new_v2 >= 0)

    new_tris  = np.stack([new_v0[tri_ok], new_v1[tri_ok], new_v2[tri_ok]], axis=1)
    new_verts = np.asarray(mesh.vertices)[keep_idx]

    out = o3d.geometry.TriangleMesh()
    out.vertices  = o3d.utility.Vector3dVector(new_verts)
    out.triangles = o3d.utility.Vector3iVector(new_tris)

    # Carry over vertex colours if present
    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors)[keep_idx]
        out.vertex_colors = o3d.utility.Vector3dVector(colors)

    out.remove_unreferenced_vertices()
    out.compute_vertex_normals()
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_joints(averaged_joints: list[dict]) -> Optional[np.ndarray]:
    """Extract (N, 3) xyz array from the averagedJoints3D list."""
    if not averaged_joints:
        return None
    pts = []
    for j in averaged_joints:
        try:
            pts.append([float(j["x"]), float(j["y"]), float(j["z"])])
        except (KeyError, TypeError, ValueError):
            continue
    return np.array(pts, dtype=np.float64) if pts else None


def load_averaged_joints(pose_json_path: Path) -> list[dict]:
    """Load averagedJoints3D from a pose.json file."""
    data = json.loads(pose_json_path.read_bytes())
    return data.get("averagedJoints3D", [])


# ---------------------------------------------------------------------------
# CLI (for offline testing / inspection)
# ---------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(description="Clean TSDF mesh and sample body point cloud.")
    parser.add_argument("--session", required=True,
                        help="Path to session directory, e.g. sessions/<uuid>")
    parser.add_argument("--mesh",
                        help="Path to mesh PLY (default: <session>/result_mesh.ply)")
    parser.add_argument("--joints",
                        help="Path to pose.json (default: <session>/pose.json)")
    parser.add_argument("--radius", type=float, default=0.45,
                        help="Horizontal crop radius around skeleton (metres, default 0.45)")
    parser.add_argument("--floor-margin", type=float, default=0.05,
                        help="Remove verts this far below lowest joint (default 0.05)")
    parser.add_argument("--sample-points", type=int, default=50000,
                        help="Point cloud density (default 50000)")
    parser.add_argument("--no-smooth", action="store_true",
                        help="Disable Laplacian smoothing")
    args = parser.parse_args()

    session_dir  = Path(args.session)
    mesh_path    = Path(args.mesh) if args.mesh else session_dir / "result_mesh.ply"
    joints_path  = Path(args.joints) if args.joints else session_dir / "pose.json"

    if not mesh_path.exists():
        sys.exit(f"Mesh not found: {mesh_path}")
    if not joints_path.exists():
        sys.exit(f"pose.json not found: {joints_path}")

    print(f"Loading mesh from {mesh_path} ...")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()
    print(f"  {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

    joints = load_averaged_joints(joints_path)
    print(f"Loaded {len(joints)} averaged joints from {joints_path}")

    pcd, clean_mesh = clean_mesh_and_sample(
        mesh,
        joints,
        out_dir=session_dir,
        body_radius_m=args.radius,
        floor_margin_m=args.floor_margin,
        sample_points=args.sample_points,
        laplacian_iters=0 if args.no_smooth else 1,
    )

    print("\nDone.")
    print(f"  Cleaned mesh:  {session_dir}/cleaned_mesh.ply")
    print(f"  Clean pcd:     {session_dir}/clean_pcd.ply")
    print(f"  Open in MeshLab / CloudCompare to inspect.")


if __name__ == "__main__":
    _cli()
