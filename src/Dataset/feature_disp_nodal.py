from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterator, Optional, TextIO

import pandas as pd

try:
	from Dataset import feature_coodinate_nodal
except ImportError:  # 実行パスによってはローカルインポートが必要
	import feature_coodinate_nodal  # type: ignore

try:
	from Dataset import feature_force_disp
except ImportError:  # 実行パスによってはローカルインポートが必要
	import feature_force_disp  # type: ignore

try:
	from Dataset import feature_fixed_disp
except ImportError:  # 実行パスによってはローカルインポートが必要
	import feature_fixed_disp  # type: ignore


TIME_PATTERN = re.compile(r"total transient time\s*=\s*([0-9.+\-Ee]+)")
NODE_LINE_PATTERN = re.compile(r"^\s*\d+")
DEFAULT_OUTPUT_SUFFIX = "_incremental_disp.csv"
DEFAULT_COORDINATE_SUFFIX = "_coordinates.csv"
DEFAULT_COMBINED_SUFFIX = "_disp_with_coords.csv"



def iter_incremental_displacements(stream: TextIO) -> Iterator[dict[str, float | int]]:
	"""各インクリメントにおける節点の変位レコードを順次生成する。

	MSC Marc の出力ストリームを読み進め、"total transient time" の行を
	起点として "incremental displacements" テーブルを検出する。
	テーブル内の数値行ごとに、インクリメント番号 (1, 2, 3...), 節点ID、
	x/y/z 方向の変位量 (dx, dy, dz) を含む辞書を返す。
	"""

	current_time: Optional[int] = None
	pending_time_marker: Optional[float] = None
	increment_index = 0
	in_block = False

	for raw_line in stream:
		line = raw_line.rstrip("\n")
		stripped = line.strip()

		if not stripped:
			# 空行はセクション区切りやヘッダ直後に挿入されるため無視する
			continue

		time_match = TIME_PATTERN.search(line)
		if time_match:
			pending_time_marker = float(time_match.group(1))
			continue

		normalized = stripped.replace(" ", "").lower()

		if "incrementaldisplacements" in normalized:
			if pending_time_marker is None:
				raise ValueError(
					"\"total transient time\" の行より前に変位データを検出しました。"
				)
			increment_index += 1
			current_time = increment_index
			pending_time_marker = None
			in_block = True
			continue

		if not in_block:
			continue

		if not NODE_LINE_PATTERN.match(line):
			# 数字から始まらない行が来たらテーブル終端とみなす
			in_block = False
			continue

		parts = stripped.split()
		if len(parts) < 4:
			# 節点IDと3成分の変位が揃っていない行はスキップ
			continue

		try:
			node_id = int(parts[0])
			dx = float(parts[1])
			dy = float(parts[2])
			dz = float(parts[3])
		except ValueError:
			# 数値化できない場合もスキップして処理を継続
			continue

		yield {
			"time": current_time,
			"node_id": node_id,
			"dx": dx,
			"dy": dy,
			"dz": dz,
		}


def _ensure_parent(path: Path) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	return path


def build_displacement_dataframe(input_path: Path, encoding: str = "utf-8") -> pd.DataFrame:
	"""結果ファイルからインクリメンタル変位データを抽出しDataFrameで返す。"""

	if not input_path.exists():
		raise FileNotFoundError(f"入力ファイルが見つかりません: {input_path}")

	records: list[dict[str, float | int]] = []
	with input_path.open("r", encoding=encoding, errors="ignore") as source:
		for record in iter_incremental_displacements(source):
			records.append(record)

	if not records:
		raise ValueError(
			"インクリメンタル変位データが見つかりません。"
			"入力ファイルに該当セクションが含まれているか確認してください。"
		)

	return pd.DataFrame(records, columns=["time", "node_id", "dx", "dy", "dz"])


def _default_output_path(input_path: Path) -> Path:
	stem = input_path.with_suffix("").name
	return input_path.parent / f"{stem}{DEFAULT_OUTPUT_SUFFIX}"


def _default_coordinate_path(input_path: Path) -> Path:
	stem = input_path.with_suffix("").name
	return input_path.parent / f"{stem}{DEFAULT_COORDINATE_SUFFIX}"


def _default_combined_path(input_path: Path) -> Path:
	stem = input_path.with_suffix("").name
	return input_path.parent / f"{stem}{DEFAULT_COMBINED_SUFFIX}"


def combine_displacement_and_coordinates(
	displacement_df: pd.DataFrame,
	coordinate_df: pd.DataFrame,
) -> pd.DataFrame:
	"""変位データと座標データをマージしたDataFrameを返す。"""

	merged = coordinate_df.merge(displacement_df, on=["time", "node_id"], how="left")
	for axis in ("dx", "dy", "dz"):
		if axis not in merged:
			merged[axis] = 0.0
		else:
			merged[axis] = merged[axis].fillna(0.0)

	return merged[["time", "node_id", "x", "y", "z", "dx", "dy", "dz"]]


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"MSC Marc 出力から \"incremental displacements\" テーブルを抽出し、"
			"座標と結合した単一のCSVを生成します (必要に応じて中間ファイルも保存可能)。"
		)
	)
	parser.add_argument("input", type=Path, help="Marc の .out 結果ファイルへのパス")
	parser.add_argument(
		"-o",
		"--output",
		type=Path,
		help="変位CSVを保存したい場合のパス (省略時は保存しません)",
	)
	parser.add_argument(
		"-c",
		"--coordinate-output",
		type=Path,
		help="座標CSVを保存したい場合のパス (省略時は保存しません)",
	)
	parser.add_argument(
		"-m",
		"--merged-output",
		type=Path,
		help="変位+座標を結合したCSVの保存先 (省略時は <入力名>_disp_with_coords.csv)",
	)
	parser.add_argument(
		"--force-condition",
		default="liver_forced",
		help="結合したい力境界条件名 (省略時は %(default)s)",
	)
	parser.add_argument(
		"--fixed-csv",
		type=Path,
		help="固定節点情報CSVのパス (既定: dataset/liver_model_info/fixed_nodes.csv)",
	)
	parser.add_argument(
		"--encoding",
		default="utf-8",
		help="入力ファイルの文字エンコーディング (既定: %(default)s)",
	)
	return parser


def main(argv: Optional[list[str]] = None) -> int:
	parser = build_parser()
	args = parser.parse_args(argv)

	input_path: Path = args.input

	disp_df = build_displacement_dataframe(input_path, encoding=args.encoding)
	if args.output:
		_ensure_parent(args.output)
		disp_df.to_csv(args.output, index=False)
		print(f"Saved displacement CSV to {args.output}")

	initial_df = feature_coodinate_nodal.load_initial_coordinates(input_path, encoding=args.encoding)
	coord_df = feature_coodinate_nodal.build_coordinate_timeseries(initial_df, disp_df)
	if args.coordinate_output:
		_ensure_parent(args.coordinate_output)
		coord_df.to_csv(args.coordinate_output, index=False)
		print(f"Saved coordinate CSV to {args.coordinate_output}")

	combined_df = combine_displacement_and_coordinates(disp_df, coord_df)
	force_features = feature_force_disp.compute_force_distance_features(
		input_path,
		encoding=args.encoding,
		condition_name=args.force_condition,
	)
	combined_df = combined_df.merge(force_features, on="node_id", how="left")
	fixed_csv = args.fixed_csv or Path("dataset/liver_model_info/fixed_nodes.csv")
	fixed_features = feature_fixed_disp.compute_fixed_node_distances(
		combined_df,
		fixed_csv,
	)
	combined_df = combined_df.merge(fixed_features, on=["time", "node_id"], how="left")
	merged_path = args.merged_output or _default_combined_path(input_path)
	_ensure_parent(merged_path)
	combined_df.to_csv(merged_path, index=False)
	print(f"Wrote merged displacement+coordinate CSV to {merged_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
