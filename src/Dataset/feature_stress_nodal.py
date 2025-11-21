from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


STRESS_COLUMNS = ["Sxx", "Syy", "Szz", "Sxy", "Syz", "Szx"]
DEFAULT_CONNECTIVITY = Path("dataset/liver_model_info/tetra_connectivity.csv")


class StressComputationError(RuntimeError):
	"""Raised when nodal stress interpolation cannot be completed."""


def _augment_coordinates(coords: np.ndarray) -> np.ndarray:
	"""Append a ones column to coordinate matrix."""
	return np.hstack([coords, np.ones((coords.shape[0], 1), dtype=coords.dtype)])


def _safe_inverse(matrix: np.ndarray) -> np.ndarray:
	"""Return matrix inverse (or pseudo-inverse if singular)."""
	try:
		return np.linalg.inv(matrix)
	except np.linalg.LinAlgError:
		return np.linalg.pinv(matrix)


def _build_connectivity_lookup(df: pd.DataFrame) -> Dict[int, Tuple[int, int, int, int]]:
	required_columns = {"element_id", "n1", "n2", "n3", "n4"}
	missing = required_columns - set(df.columns)
	if missing:
		raise ValueError(f"tetra_connectivity.csv に必要な列がありません: {sorted(missing)}")
	lookup: Dict[int, Tuple[int, int, int, int]] = {}
	for row in df.itertuples(index=False):
		lookup[int(row.element_id)] = (int(row.n1), int(row.n2), int(row.n3), int(row.n4))
	return lookup


def _build_coordinate_lookup(df: pd.DataFrame) -> Dict[Tuple[float, int], np.ndarray]:
	required_columns = {"time", "node_id", "x", "y", "z"}
	missing = required_columns - set(df.columns)
	if missing:
		raise ValueError(f"disp_with_coords.csv に必要な列がありません: {sorted(missing)}")
	lookup: Dict[Tuple[float, int], np.ndarray] = {}
	for row in df.itertuples(index=False):
		key = (float(row.time), int(row.node_id))
		lookup[key] = np.array([float(row.x), float(row.y), float(row.z)], dtype=float)
	return lookup


def _fetch_node_coordinates(
	time_value: float,
	node_ids: Sequence[int],
	coord_lookup: Dict[Tuple[float, int], np.ndarray],
) -> np.ndarray:
	coords: List[np.ndarray] = []
	for node_id in node_ids:
		key = (time_value, int(node_id))
		try:
			coords.append(coord_lookup[key])
		except KeyError as exc:
			raise StressComputationError(
				f"時刻 {time_value} の節点 {node_id} の座標が disp_with_coords.csv に見つかりません"
			) from exc
	return np.vstack(coords)


def _prepare_group_matrix(group: pd.DataFrame) -> np.ndarray:
	points = group[["x", "y", "z"]].to_numpy(dtype=float)
	if points.shape != (4, 3):
		raise StressComputationError(
			f"element_id={group.element_id.iloc[0]} time={group.time.iloc[0]} で積分点が {points.shape[0]} 個しかありません"
		)
	return points


def compute_nodal_stresses(
	stress_df: pd.DataFrame,
	coord_lookup: Dict[Tuple[float, int], np.ndarray],
	connectivity_lookup: Dict[int, Tuple[int, int, int, int]],
) -> pd.DataFrame:
	if not {"time", "element_id", "point_id", "x", "y", "z"}.issubset(stress_df.columns):
		raise ValueError("stress_elements.csv に必要な列 (time, element_id, point_id, x, y, z) が不足しています。")
	missing_stress = [col for col in STRESS_COLUMNS if col not in stress_df.columns]
	if missing_stress:
		raise ValueError(f"stress_elements.csv に応力列が不足しています: {missing_stress}")

	results: List[dict[str, float | int]] = []
	grouped = stress_df.groupby(["time", "element_id"], sort=True)
	for (time_value, element_id), group in grouped:
		element_id = int(element_id)
		if element_id not in connectivity_lookup:
			raise StressComputationError(f"element_id={element_id} が tetra_connectivity.csv に存在しません")
		node_ids = connectivity_lookup[element_id]
		group = group.sort_values("point_id")
		integration_points = _prepare_group_matrix(group)
		integration_matrix = _augment_coordinates(integration_points)
		inv_integration = _safe_inverse(integration_matrix)
		stress_values = group[STRESS_COLUMNS].to_numpy(dtype=float)
		node_coords = _fetch_node_coordinates(float(time_value), node_ids, coord_lookup)
		node_matrix = _augment_coordinates(node_coords)
		coefficients = inv_integration @ stress_values
		node_stresses = node_matrix @ coefficients
		for node_id, (x, y, z), stresses in zip(node_ids, node_coords, node_stresses):
			row = {
				"time": float(time_value),
				"node_id": int(node_id),
				"x": float(x),
				"y": float(y),
				"z": float(z),
			}
			for col, value in zip(STRESS_COLUMNS, stresses):
				row[col] = float(value)
			results.append(row)

	if not results:
		raise StressComputationError("節点応力データを生成できませんでした。入力を確認してください。")
	return pd.DataFrame(results)


def compute_average_stresses(nodal_df: pd.DataFrame) -> pd.DataFrame:
	if not {"time", "node_id"}.issubset(nodal_df.columns):
		raise ValueError("節点応力データに time, node_id 列が存在しません。")
	agg_columns = {column: "mean" for column in STRESS_COLUMNS if column in nodal_df.columns}
	avg_df = (
		nodal_df.groupby(["time", "node_id"], as_index=False)
		.agg(agg_columns)
		.sort_values(["time", "node_id"])
		.reset_index(drop=True)
	)
	return avg_df


def _default_output_path(stress_path: Path) -> Path:
	stem = stress_path.with_suffix("").name
	return stress_path.parent / f"{stem}_nodal.csv"


def _default_average_output_path(stress_path: Path) -> Path:
	stem = stress_path.with_suffix("").name
	return stress_path.parent / f"{stem}_nodal_average.csv"


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"要素積分点の応力CSVと節点座標CSVから、節点ごとの応力値を計算して出力します。"
		)
	)
	parser.add_argument("stress_elements", type=Path, help="要素ごとの応力CSV (stress_elements.csv)")
	parser.add_argument("disp_with_coords", type=Path, help="節点座標付き変位CSV (disp_with_coords.csv)")
	parser.add_argument(
		"--connectivity",
		type=Path,
		default=DEFAULT_CONNECTIVITY,
		help="要素の節点IDを定義する tetra_connectivity.csv (既定: %(default)s)",
	)
	parser.add_argument("-o", "--output", type=Path, help="出力CSVパス")
	parser.add_argument(
		"--encoding",
		default="utf-8",
		help="入力CSVの文字エンコーディング (既定: %(default)s)",
	)
	parser.add_argument(
		"--average-output",
		type=Path,
		help="時刻×節点で平均化した応力CSVの出力パス (省略時は _nodal_average.csv)",
	)
	return parser


def main(argv: Optional[List[str]] = None) -> int:
	parser = build_parser()
	args = parser.parse_args(argv)
	stress_path: Path = args.stress_elements
	disp_path: Path = args.disp_with_coords
	connectivity_path: Path = args.connectivity
	encoding: str = args.encoding

	if not stress_path.exists():
		raise FileNotFoundError(f"stress_elements.csv が見つかりません: {stress_path}")
	if not disp_path.exists():
		raise FileNotFoundError(f"disp_with_coords.csv が見つかりません: {disp_path}")
	if not connectivity_path.exists():
		raise FileNotFoundError(f"tetra_connectivity.csv が見つかりません: {connectivity_path}")

	stress_df = pd.read_csv(stress_path, encoding=encoding)
	disp_df = pd.read_csv(disp_path, encoding=encoding)
	connectivity_df = pd.read_csv(connectivity_path, encoding=encoding)

	coord_lookup = _build_coordinate_lookup(disp_df)
	connectivity_lookup = _build_connectivity_lookup(connectivity_df)
	nodal_df = compute_nodal_stresses(stress_df, coord_lookup, connectivity_lookup)
	output_path = args.output or _default_output_path(stress_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	nodal_df.to_csv(output_path, index=False)
	print(f"Wrote {len(nodal_df)} nodal stress rows to {output_path}")

	average_output_path = args.average_output or _default_average_output_path(stress_path)
	average_output_path.parent.mkdir(parents=True, exist_ok=True)
	average_df = compute_average_stresses(nodal_df)
	average_df.to_csv(average_output_path, index=False)
	print(f"Wrote {len(average_df)} averaged nodal stress rows to {average_output_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
