
import argparse
import os
import time
import re
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d


def load_frames_and_colors(
	csv_path: str,
	error_cols: Optional[List[str]] = None,
	cmap_name: str = "viridis",
	load_correct: bool = False,
	force_node_id: Optional[int] = None,
):
	"""CSV からフレーム（各時刻の頂点座標）と色を作成して返す。

	CSV は最低限 `time`, `node_id`, `x`, `y`, `z` を含むこと。
	誤差カラム群が指定されなければデフォルトで `x_error,y_error,z_error` を使い、
	各頂点の誤差マグニチュードを sqrt(sum squares) で計算する。

	load_correct=True の場合、戻り値に correct_frames (正解座標のリスト) が追加される。
	正解列名は x_correct, y_correct, z_correct を想定。
	
	force_node_id が指定された場合、その節点のインデックスも返す。
	"""
	df = pd.read_csv(csv_path)

	# Optional debug prints
	print(df[['x_error','y_error','z_error','dx_error','dy_error','dz_error','Sxx_error','Syy_error','Szz_error','Sxy_error','Syz_error','Szx_error']].abs().max())
	print(df[['x_error','y_error','z_error','dx_error','dy_error','dz_error','Sxx_error','Syy_error','Szz_error','Sxy_error','Syz_error','Szx_error']].abs().mean())

	required = {"time", "node_id", "x", "y", "z"}
	if not required.issubset(set(df.columns)):
		raise ValueError(f"CSV must contain columns: {required}")
	
	force_node_index = None
	if force_node_id is not None:
		unique_nodes = np.sort(df["node_id"].unique())
		if force_node_id in unique_nodes:
			force_node_index = np.searchsorted(unique_nodes, force_node_id)
		else:
			print(f"Warning: force node {force_node_id} not found in data.")

	has_correct = False
	if load_correct:
		correct_required = {"x_correct", "y_correct", "z_correct"}
		if correct_required.issubset(set(df.columns)):
			has_correct = True
		else:
			print("Warning: correct columns (x_correct, y_correct, z_correct) not found. Disabling comparison.")

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
	correct_frames = []

	for t in times:
		df_t = df[df["time"] == t].sort_values(by="node_id").reset_index(drop=True)
		pts = df_t[["x", "y", "z"]].to_numpy()
		frames.append(pts)

		if has_correct:
			c_pts = df_t[["x_correct", "y_correct", "z_correct"]].to_numpy()
			correct_frames.append(c_pts)

		mags = df_t["__err_mag"].to_numpy()
		if vmax > vmin:
			norm = (mags - vmin) / (vmax - vmin)
		else:
			norm = np.zeros_like(mags)

		vertex_colors = cmap(norm)[:, :3]
		colors.append(vertex_colors)

	if load_correct:
		return frames, colors, times, vmin, vmax, correct_frames if has_correct else None, force_node_index
	
	return frames, colors, times, vmin, vmax, force_node_index


def create_mesh_or_pointcloud_from_frames(frames: List[np.ndarray], tet_file: str = ""):
	"""frames の最初のフレームを参照して三角形メッシュ（四面体ファイルがあれば）を作成。
	見つからなければ点群を返す。戻り値は (geom, line_set, is_mesh, tets)
	tets は四面体インデックス配列（ない場合は None）
	"""
	initial = frames[0]
	tets = None
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
			return mesh, line_set, True, tets
		except Exception as e:
			print(f"四面体ファイルの読み込みで例外: {e}; 点群表示にフォールバックします。")
			# fall through to pointcloud fallback below
			pass

	# fallback: point cloud
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(initial)
	return pcd, None, False, None


def animate(
	frames: List[np.ndarray], 
	colors: List[np.ndarray], 
	times: List, 
	tet_file: str = "", 
	fps: float = 8.0, 
	overlay: bool = False, 
	show_true: bool = False,
	correct_frames: Optional[List[np.ndarray]] = None,
	force_node_index: Optional[int] = None
):
	"""Open3D を用いてアニメーション表示する。frames/colors は予測または単一シーケンス用。
	correct_frames があり、かつ show_true=True の場合、右隣に正解を表示する。
	force_node_index がある場合、その節点位置に赤い球を表示する。
	"""
	# --- Main Geometry (Prediction) ---
	geom, lines, is_mesh, tets = create_mesh_or_pointcloud_from_frames(frames, tet_file)
	
	# Determine scale for force point
	mins = np.min(frames[0], axis=0)
	maxs = np.max(frames[0], axis=0)
	bbox_diag = np.linalg.norm(maxs - mins)
	radius = bbox_diag * 0.01

	geom_force = None
	if force_node_index is not None and force_node_index < len(frames[0]):
		geom_force = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
		geom_force.paint_uniform_color([1.0, 0.0, 0.0])
		# Initialize position
		pt = frames[0][force_node_index]
		geom_force.translate(pt, relative=False)

	# --- Correct Geometry (if enabled) ---
	geom_c, lines_c = None, None
	geom_force_c = None
	if show_true and correct_frames:
		# Copy geometry structure from main
		if is_mesh:
			# Re-create mesh manually or copy
			# Using create function again is safer to get a fresh instance
			g, l, _, _ = create_mesh_or_pointcloud_from_frames(correct_frames, tet_file)
			geom_c = g
			lines_c = l
		else:
			geom_c = o3d.geometry.PointCloud()
			geom_c.points = o3d.utility.Vector3dVector(correct_frames[0])

		if force_node_index is not None and force_node_index < len(correct_frames[0]):
			geom_force_c = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
			geom_force_c.paint_uniform_color([1.0, 0.0, 0.0])
			pt_c = correct_frames[0][force_node_index]
			geom_force_c.translate(pt_c, relative=False)

	vis = o3d.visualization.Visualizer()
	vis.create_window(window_name=f"Deformation Animation - time {times[0]}")
	
	vis.add_geometry(geom)
	if is_mesh and lines is not None:
		vis.add_geometry(lines)
	
	if geom_force:
		vis.add_geometry(geom_force)

	if geom_c:
		# Shift correct geometry to the right
		# Find rough bounding box size to determine offset
		width = maxs[0] - mins[0]
		offset = np.array([width*1.1, 0, 0])
		
		# Apply initial offset
		if is_mesh:
			# Translate mesh vertices
			vs = np.asarray(geom_c.vertices)
			geom_c.vertices = o3d.utility.Vector3dVector(vs + offset)
			geom_c.compute_vertex_normals()
			
			# Paint it gray
			geom_c.paint_uniform_color([0.5, 0.5, 0.5])
			
			if lines_c:
				# Translate line set points
				ls_pts = np.asarray(lines_c.points)
				lines_c.points = o3d.utility.Vector3dVector(ls_pts + offset)
				lines_c.paint_uniform_color([0.3, 0.3, 0.3]) # darker gray for lines
				vis.add_geometry(lines_c)
		else:
			# Translate pcd points
			pts = np.asarray(geom_c.points)
			geom_c.points = o3d.utility.Vector3dVector(pts + offset)
			geom_c.paint_uniform_color([0.5, 0.5, 0.5])
			
		vis.add_geometry(geom_c)

		if geom_force_c:
			geom_force_c.translate(offset, relative=True)
			vis.add_geometry(geom_force_c)

	# Reset camera to look at the center of all geometries
	vis.poll_events()
	vis.update_renderer()
	ctr = vis.get_view_control()
	# First, let Open3D auto-center based on added geometries
	ctr.set_lookat(np.mean(frames[0], axis=0)) # Rough center, but let's do better
	
	# Explicitly calculate center of bounding box union
	all_mins = np.min(frames[0], axis=0)
	all_maxs = np.max(frames[0], axis=0)
	if geom_c:
		# If comparing, account for the second model
		c_mins = all_mins + offset
		c_maxs = all_maxs + offset
		all_mins = np.minimum(all_mins, c_mins)
		all_maxs = np.maximum(all_maxs, c_maxs)
	
	center = (all_mins + all_maxs) / 2.0
	ctr.set_lookat(center)

	delay = 1.0 / float(max(1.0, fps))
	idx = 0
	
	# If geom_c exists, we need to apply offset every update
	offset = np.array([0.0, 0.0, 0.0])
	if geom_c:
		mins = np.min(frames[0], axis=0)
		maxs = np.max(frames[0], axis=0)
		width = maxs[0] - mins[0]
		offset = np.array([width*1.1, 0, 0])

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

		if geom_force:
			pt = frames[idx][force_node_index]
			# translate to new position
			# Since we want to move TO pt, and current center is unknown (accumulated),
			# better to recreate or translate from previous?
			# translate(v, relative=False) moves center TO v.
			geom_force.translate(pt, relative=False)
			vis.update_geometry(geom_force)
			
		# Update correct geometry
		if geom_c and idx < len(correct_frames):
			pts_c = correct_frames[idx]
			if is_mesh:
				geom_c.vertices = o3d.utility.Vector3dVector(pts_c + offset)
				geom_c.compute_vertex_normals()
				vis.update_geometry(geom_c)
				if lines_c:
					lines_c.points = geom_c.vertices
					vis.update_geometry(lines_c)
			else:
				geom_c.points = o3d.utility.Vector3dVector(pts_c + offset)
				vis.update_geometry(geom_c)

			if geom_force_c:
				pt_c = correct_frames[idx][force_node_index]
				geom_force_c.translate(pt_c + offset, relative=False)
				vis.update_geometry(geom_force_c)

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
	parser.add_argument("--fps", type=float, default=12.0, help="frames per second")
	parser.add_argument("--error_cols", type=str, default="", help="comma separated error columns to combine (default: x_error,y_error,z_error)")
	parser.add_argument("--overlay", action="store_true", help="overlay mode (not used here, kept for compatibility)")
	parser.add_argument("--compare", action="store_true", help="Show ground truth side-by-side (requires x_correct, y_correct, z_correct in CSV)")
	parser.add_argument("--add_force_point", action="store_true", help="Extract nodeID from filename (node{id}) and plot a red point on it")
	args = parser.parse_args()

	error_cols = None
	if args.error_cols:
		error_cols = [c.strip() for c in args.error_cols.split(",") if c.strip()]

	force_id = None
	if args.add_force_point:
		# Tries to find "node123" in the filename
		match = re.search(r"node(\d+)", args.input_csv)
		if match:
			force_id = int(match.group(1))
			print(f"Detected force node ID: {force_id} from filename.")
		else:
			print("Warning: --add_force_point set but regex 'node(\d+)' failed on input filename.")

	res = load_frames_and_colors(args.input_csv, error_cols=error_cols, cmap_name=args.cmap, load_correct=args.compare, force_node_id=force_id)
	
	correct_frames = None
	force_index = None

	if args.compare:
		# Unpack with correct frames + force_index
		frames, colors, times, vmin, vmax, correct_frames, force_index = res
	else:
		frames, colors, times, vmin, vmax, force_index = res

	animate(frames, colors, times, tet_file=args.tet_file, fps=args.fps, overlay=args.overlay, show_true=args.compare, correct_frames=correct_frames, force_node_index=force_index)


if __name__ == "__main__":
	main()
