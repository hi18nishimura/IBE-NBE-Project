from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
	from Dataset import feature_coodinate_nodal
except ImportError:
	import feature_coodinate_nodal  # type: ignore

try:
	from Dataset import feature_force
except ImportError:
	import feature_force  # type: ignore


def compute_force_distance_features(
	result_file: Path,
	*,
	encoding: str = "utf-8",
	condition_name: str = "liver_forced",
) -> pd.DataFrame:
	"""強制変位を与えている節点への距離(各軸成分)を計算する。"""

	coords_df = feature_coodinate_nodal.load_initial_coordinates(result_file, encoding=encoding)
	force_df = feature_force.extract_force_dataframe(
		result_file,
		encoding=encoding,
		condition_name=condition_name,
	)

	force_node_ids = force_df["node_id"].unique()
	force_coords = coords_df[coords_df["node_id"].isin(force_node_ids)]
	if force_coords.empty:
		raise ValueError("強制変位節点の座標が取得できませんでした。")

	node_points = coords_df[["node_id", "x", "y", "z"]].to_numpy(dtype=float)
	force_points = force_coords[["node_id", "x", "y", "z"]].to_numpy(dtype=float)
	force_vectors = force_points[:, 1:]

	features: list[dict[str, float | int]] = []
	for node_row in node_points:
		node_id = int(node_row[0])
		point = node_row[1:]
		diff = force_vectors - point
		dists = np.linalg.norm(diff, axis=1)
		min_idx = int(np.argmin(dists))
		closest_force_id = int(force_points[min_idx, 0])
		axis_dist = np.abs(diff[min_idx])
		features.append(
			{
				"node_id": node_id,
				"force_node_id": closest_force_id,
				"f_x": float(axis_dist[0]),
				"f_y": float(axis_dist[1]),
				"f_z": float(axis_dist[2]),
			}
		)

	return pd.DataFrame(features)
