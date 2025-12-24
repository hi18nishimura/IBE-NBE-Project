import argparse
import os
import time
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d


def load_df_with_error(
    csv_path: str,
    error_cols: Optional[List[str]] = None,
):
    """CSV を読み込み、誤差マグニチュードを計算して DataFrame を返す。"""
    df = pd.read_csv(csv_path)

    required = {"time", "node_id", "x", "y", "z"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {required}")

    if error_cols is None:
        # default
        error_cols = ["x_error", "y_error", "z_error"]

    # compute per-row error magnitude
    present_error_cols = [c for c in error_cols if c in df.columns]
    if len(present_error_cols) == 0:
        # try stress-error fallback
        stress_cols = [
            "Sxx_error",
            "Syy_error",
            "Szz_error",
            "Sxy_error",
            "Syz_error",
            "Szx_error",
        ]
        present_error_cols = [c for c in stress_cols if c in df.columns]

    if len(present_error_cols) == 0:
        # if still none, set zero errors
        df["__err_mag"] = 0.0
    else:
        # default combine: euclidean norm across specified columns
        df["__err_mag"] = np.sqrt((df[present_error_cols].fillna(0.0) ** 2).sum(axis=1))
    
    return df

def generate_frames_and_colors(
    df: pd.DataFrame,
    cmap_name: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
):
    """DataFrame からフレームと色を作成する。"""
    times = sorted(df["time"].unique())

    if vmin is None:
        vmin = float(df["__err_mag"].min())
    if vmax is None:
        vmax = float(df["__err_mag"].max())
    
    cmap = plt.get_cmap(cmap_name)

    frames = []
    colors = []
    for t in times:
        df_t = df[df["time"] == t].sort_values(by="node_id").reset_index(drop=True)
        pts = df_t[["x", "y", "z"]].to_numpy()
        frames.append(pts)

        mags = df_t["__err_mag"].to_numpy()
        if vmax > vmin:
            norm = (mags - vmin) / (vmax - vmin)
        else:
            norm = np.zeros_like(mags)

        vertex_colors = cmap(norm)[:, :3]
        colors.append(vertex_colors)

    return frames, colors, times, vmin, vmax


def create_mesh_or_pointcloud_from_frames(frames: List[np.ndarray], tet_file: str = ""):
    """frames の最初のフレームを参照して三角形メッシュ（四面体ファイルがあれば）を作成。
    見つからなければ点群を返す。戻り値は (geom, line_set, is_mesh)
    """
    initial = frames[0]
    if tet_file and os.path.exists(tet_file):
        try:
            # Try to read as a normal CSV with header (e.g. element_id,n1,n2,n3,n4)
            try:
                tet_df = pd.read_csv(tet_file)
                # common header names: n1,n2,n3,n4
                if {"n1", "n2", "n3", "n4"}.issubset(set(tet_df.columns)):
                    tets = tet_df[["n1", "n2", "n3", "n4"]].to_numpy(dtype=int) - 1
                else:
                    # not the expected CSV header: raise to fall back to whitespace parse
                    raise ValueError("tet file not in (n1..n4) CSV format")
            except Exception:
                # fallback: whitespace separated file without header (old style)
                tet_df = pd.read_csv(tet_file, sep="\s+", header=None, skiprows=1)
                tets = tet_df.iloc[:, 1:5].to_numpy(dtype=int) - 1

            faces = []
            for tet in tets:
                # tet indices -> form 4 triangular faces
                tet_faces_idx = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
                p = initial[tet]
                centroid = np.mean(p, axis=0)
                for fi in tet_faces_idx:
                    v0, v1, v2 = p[fi]
                    normal = np.cross(v1 - v0, v2 - v0)
                    face_center = (v0 + v1 + v2) / 3.0
                    if np.dot(normal, centroid - face_center) > 0:
						# flip
                        face_vids = [tet[fi[0]], tet[fi[2]], tet[fi[1]]]
                    else:
                        face_vids = [tet[fi[0]], tet[fi[1]], tet[fi[2]]]
                    faces.append(face_vids)

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(initial)
            mesh.triangles = o3d.utility.Vector3iVector(np.array(faces, dtype=int))
            mesh.compute_vertex_normals()
            mesh.orient_triangles()

            line_set = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
            return mesh, line_set, True
        except Exception as e:
            print(f"四面体ファイルの読み込みで例外: {e}; 点群表示にフォールバックします。")
            # fall through to pointcloud fallback below
            pass

    # fallback: point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(initial)
    return pcd, None, False


def animate_compare(
    frames1: List[np.ndarray], colors1: List[np.ndarray], 
    frames2: List[np.ndarray], colors2: List[np.ndarray],
    times: List, tet_file: str = "", fps: float = 8.0, 
    offset_axis: int = 0, offset_amount: float = None
):
    """2つのアニメーションを並べて表示する"""
    
    # Create geometries for both
    geom1, lines1, is_mesh1 = create_mesh_or_pointcloud_from_frames(frames1, tet_file)
    geom2, lines2, is_mesh2 = create_mesh_or_pointcloud_from_frames(frames2, tet_file)

    # Calculate offset if not provided
    if offset_amount is None:
        # Get bounding box of first frame of first mesh
        bbox = geom1.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        offset_amount = extent[offset_axis] * 1.1 # 50% padding

    # Apply initial offset to geom2
    translation = np.zeros(3)
    translation[offset_axis] = offset_amount
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Deformation Comparison")
    vis.get_render_option().light_on = False
    
    vis.add_geometry(geom1)
    if is_mesh1 and lines1:
        vis.add_geometry(lines1)
        
    vis.add_geometry(geom2)
    if is_mesh2 and lines2:
        vis.add_geometry(lines2)

    # Initial update for geom2 to apply offset
    pts2_init = frames2[0] + translation
    if is_mesh2:
        geom2.vertices = o3d.utility.Vector3dVector(pts2_init)
        geom2.compute_vertex_normals()
        if lines2:
            lines2.points = geom2.vertices
    else:
        geom2.points = o3d.utility.Vector3dVector(pts2_init)
    
    vis.update_geometry(geom2)
    if lines2:
        vis.update_geometry(lines2)

    delay = 1.0 / float(max(1.0, fps))
    idx = 0
    
    min_len = min(len(frames1), len(frames2))

    def cb(vis):
        nonlocal idx
        
        # Update 1
        pts1 = frames1[idx]
        clr1 = colors1[idx]
        
        if is_mesh1:
            geom1.vertices = o3d.utility.Vector3dVector(pts1)
            geom1.vertex_colors = o3d.utility.Vector3dVector(clr1)
            geom1.compute_vertex_normals()
            vis.update_geometry(geom1)
            if lines1:
                lines1.points = geom1.vertices
                vis.update_geometry(lines1)
        else:
            geom1.points = o3d.utility.Vector3dVector(pts1)
            geom1.colors = o3d.utility.Vector3dVector(clr1)
            vis.update_geometry(geom1)

        # Update 2
        pts2 = frames2[idx] + translation # Apply offset
        clr2 = colors2[idx]
        
        if is_mesh2:
            geom2.vertices = o3d.utility.Vector3dVector(pts2)
            geom2.vertex_colors = o3d.utility.Vector3dVector(clr2)
            geom2.compute_vertex_normals()
            vis.update_geometry(geom2)
            if lines2:
                lines2.points = geom2.vertices
                vis.update_geometry(lines2)
        else:
            geom2.points = o3d.utility.Vector3dVector(pts2)
            geom2.colors = o3d.utility.Vector3dVector(clr2)
            vis.update_geometry(geom2)

        idx = (idx + 1) % min_len
        if idx == 0:
            time.sleep(1.0)
        else:
            time.sleep(delay)
        return True

    vis.register_animation_callback(cb)
    vis.run()
    vis.destroy_window()


def save_colorbar_image(vmin, vmax, cmap_name, filename="colorbar.png"):
    """Save a colorbar image using matplotlib."""
    fig, ax = plt.subplots(figsize=(1.5, 6))
    
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    plt.colorbar(sm, cax=ax, orientation='vertical', label='Error Magnitude')
    plt.title(f'Range:\n{vmin:.2f}\n|\n{vmax:.2f}')
    
    plt.savefig(filename, bbox_inches='tight')
    print(f"Colorbar image saved to {filename}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare deformation animation from two CSVs with error coloring")
    parser.add_argument("input_csv1", type=str, help="First CSV file")
    parser.add_argument("input_csv2", type=str, help="Second CSV file")
    parser.add_argument("--tet_file", type=str, default="/workspace/dataset/liver_model_info/tetra_connectivity.csv", help="optional tetra elements file for mesh topology")
    parser.add_argument("--cmap", type=str, default="viridis", help="matplotlib colormap name")
    parser.add_argument("--fps", type=float, default=12.0, help="frames per second")
    parser.add_argument("--error_cols", type=str, default="", help="comma separated error columns to combine (default: x_error,y_error,z_error)")
    parser.add_argument("--offset_axis", type=int, default=0, help="Axis to offset the second model (0=x, 1=y, 2=z)")
    parser.add_argument("--offset_amount", type=float, default=None, help="Amount to offset the second model")
    args = parser.parse_args()

    error_cols = None
    if args.error_cols:
        error_cols = [c.strip() for c in args.error_cols.split(",") if c.strip()]

    # Load DataFrames
    print(f"Loading {args.input_csv1}...")
    df1 = load_df_with_error(args.input_csv1, error_cols=error_cols)
    print(f"Loading {args.input_csv2}...")
    df2 = load_df_with_error(args.input_csv2, error_cols=error_cols)

    # Calculate global min/max for consistent coloring
    max1 = df1["__err_mag"].max()
    max2 = df2["__err_mag"].max()
    print(f"Max Error File 1: {max1:.4f}")
    print(f"Max Error File 2: {max2:.4f}")

    vmin = min(df1["__err_mag"].min(), df2["__err_mag"].min())
    vmax = max(max1, max2)
    print(f"Global Error Range: {vmin:.4f} - {vmax:.4f}")

    # Save colorbar
    save_colorbar_image(vmin, vmax, args.cmap, "error_colorbar.png")

    # Generate frames and colors
    frames1, colors1, times1, _, _ = generate_frames_and_colors(df1, cmap_name=args.cmap, vmin=vmin, vmax=vmax)
    frames2, colors2, times2, _, _ = generate_frames_and_colors(df2, cmap_name=args.cmap, vmin=vmin, vmax=vmax)

    # Check time consistency
    if len(times1) != len(times2):
        print(f"Warning: Time steps mismatch. {len(times1)} vs {len(times2)}. Animation will loop on shorter length.")

    animate_compare(
        frames1, colors1, 
        frames2, colors2, 
        times1, 
        tet_file=args.tet_file, 
        fps=args.fps,
        offset_axis=args.offset_axis,
        offset_amount=args.offset_amount
    )


if __name__ == "__main__":
    main()
