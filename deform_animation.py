
import argparse
import os
import time
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d


def load_frames_and_colors(
	csv_path: str,
	error_cols: Optional[List[str]] = None,
	cmap_name: str = "viridis",
):
	"""CSV からフレーム（各時刻の頂点座標）と色を作成して返す。

	CSV は最低限 `time`, `node_id`, `x`, `y`, `z` を含むこと。
	誤差カラム群が指定されなければデフォルトで `x_error,y_error,z_error` を使い、
	各頂点の誤差マグニチュードを sqrt(sum squares) で計算する。
	"""
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

	times = sorted(df["time"].unique())

	vmin = float(df["__err_mag"].min())
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


def animate(frames: List[np.ndarray], colors: List[np.ndarray], times: List, tet_file: str = "", fps: float = 8.0, overlay: bool = False, show_true: bool = False):
	"""Open3D を用いてアニメーション表示する。frames/colors は予測または単一シーケンス用。
	`show_true` はここでは未使用だが CLI と合わせるため残している。
	"""
	geom, lines, is_mesh = create_mesh_or_pointcloud_from_frames(frames, tet_file)

	vis = o3d.visualization.Visualizer()
	vis.create_window(window_name=f"Deformation Animation - time {times[0]}")
	vis.add_geometry(geom)
	if is_mesh and lines is not None:
		vis.add_geometry(lines)

	delay = 1.0 / float(max(1.0, fps))
	idx = 0

	def cb(vis):
		nonlocal idx
		pts = frames[idx]
		clr = colors[idx]

		if is_mesh:
			geom.vertices = o3d.utility.Vector3dVector(pts)
			geom.vertex_colors = o3d.utility.Vector3dVector(clr)
			geom.compute_vertex_normals()
			vis.update_geometry(geom)
			if lines is not None:
				lines.points = geom.vertices
				vis.update_geometry(lines)
		else:
			geom.points = o3d.utility.Vector3dVector(pts)
			geom.colors = o3d.utility.Vector3dVector(clr)
			vis.update_geometry(geom)

		#time.sleep(delay)
		idx = (idx + 1) % len(frames)
		if idx == 1:
			time.sleep(1.0)  # pause at end
		else:
			time.sleep(delay)
		
		return True

	vis.register_animation_callback(cb)
	vis.run()
	vis.destroy_window()


def main():
	parser = argparse.ArgumentParser(description="Deformation animation from CSV with error coloring")
	parser.add_argument("input_csv", type=str, help="CSV file containing time,node_id,x,y,z and error columns")
	parser.add_argument("--tet_file", type=str, default="/workspace/dataset/liver_model_info/tetra_connectivity.csv", help="optional tetra elements file for mesh topology")
	parser.add_argument("--cmap", type=str, default="viridis", help="matplotlib colormap name")
	parser.add_argument("--fps", type=float, default=8.0, help="frames per second")
	parser.add_argument("--error_cols", type=str, default="", help="comma separated error columns to combine (default: x_error,y_error,z_error)")
	parser.add_argument("--overlay", action="store_true", help="overlay mode (not used here, kept for compatibility)")
	args = parser.parse_args()

	error_cols = None
	if args.error_cols:
		error_cols = [c.strip() for c in args.error_cols.split(",") if c.strip()]

	frames, colors, times, vmin, vmax = load_frames_and_colors(args.input_csv, error_cols=error_cols, cmap_name=args.cmap)
	animate(frames, colors, times, tet_file=args.tet_file, fps=args.fps, overlay=args.overlay)


if __name__ == "__main__":
	main()
