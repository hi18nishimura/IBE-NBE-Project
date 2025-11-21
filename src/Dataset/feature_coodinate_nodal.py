from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


COORDINATE_HEADER = "node      coordinates"
COORD_LINE_PATTERN = re.compile(r"^\s*(\d+)\s+([0-9.+\-Ee]+)\s+([0-9.+\-Ee]+)\s+([0-9.+\-Ee]+)")
COORDINATE_SUFFIX = "_coordinates.csv"


def _ensure_path(path: Path) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	return path


def load_initial_coordinates(result_file: Path, encoding: str = "utf-8") -> pd.DataFrame:
	"""結果ファイルから初期節点座標を抽出する。"""
	data: list[tuple[int, float, float, float]] = []
	with result_file.open("r", encoding=encoding, errors="ignore") as stream:
		capture = False
		for raw in stream:
			line = raw.rstrip("\n")
			if not capture:
				if COORDINATE_HEADER in line:
					capture = True
				continue
			if not line.strip():
				if capture:
					break
				continue
			match = COORD_LINE_PATTERN.match(line)
			if not match:
				break
			node_id = int(match.group(1))
			x = float(match.group(2))
			y = float(match.group(3))
			z = float(match.group(4))
			data.append((node_id, x, y, z))

	if not data:
		raise ValueError("座標データが見つかりませんでした。")

	return pd.DataFrame(data, columns=["node_id", "x", "y", "z"])


def build_coordinate_timeseries(initial_df: pd.DataFrame, displacement_df: pd.DataFrame) -> pd.DataFrame:
	"""初期座標と変位データから各時刻の節点座標を算出する。"""
	initial_df = initial_df.copy()
	initial_df["time"] = 0
	columns = ["time", "node_id", "x", "y", "z"]

	if displacement_df.empty:
		return initial_df[columns]

	disp_df = displacement_df.copy()
	disp_df = disp_df.sort_values(["node_id", "time"])
	disp_df[["cum_dx", "cum_dy", "cum_dz"]] = disp_df.groupby("node_id")[
		["dx", "dy", "dz"]
	].cumsum()

	merged = disp_df.merge(initial_df[["node_id", "x", "y", "z"]], on="node_id", how="left")
	for axis, col in zip(("x", "y", "z"), ("cum_dx", "cum_dy", "cum_dz")):
		merged[axis] = merged[axis] + merged[col]

	position_df = merged[["time", "node_id", "x", "y", "z"]]
	result = pd.concat([initial_df[columns], position_df], ignore_index=True)
	return result.sort_values(["time", "node_id"]).reset_index(drop=True)


def _default_output_path(result_file: Path) -> Path:
	stem = result_file.with_suffix("").name
	return result_file.parent / f"{stem}{COORDINATE_SUFFIX}"


def generate_coordinate_csv(
	result_file: Path,
	displacement_csv: Path,
	output_path: Path | None = None,
	encoding: str = "utf-8",
) -> Path:
	"""初期座標と変位CSVから時系列座標CSVを生成する。"""
	initial_df = load_initial_coordinates(result_file, encoding=encoding)
	disp_df = pd.read_csv(displacement_csv)
	coordinate_df = build_coordinate_timeseries(initial_df, disp_df)
	output = output_path or _default_output_path(result_file)
	_ensure_path(output)
	coordinate_df.to_csv(output, index=False)
	return output


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="結果ファイルと変位CSVから各時刻の節点座標を生成して保存します。"
	)
	parser.add_argument("result", type=Path, help="Marc結果ファイル (.out)")
	parser.add_argument("displacement", type=Path, help="feature_disp_nodalで生成したCSV")
	parser.add_argument(
		"-o",
		"--output",
		type=Path,
		help="出力CSVパス (既定: <結果ファイル>_coordinates.csv)",
	)
	parser.add_argument(
		"--encoding",
		default="utf-8",
		help="入力ファイルの文字エンコーディング (既定: %(default)s)",
	)
	return parser


def main(argv: Iterable[str] | None = None) -> int:
	parser = build_parser()
	args = parser.parse_args(argv)
	output = generate_coordinate_csv(
		args.result,
		args.displacement,
		output_path=args.output,
		encoding=args.encoding,
	)
	print(f"Wrote coordinate timeseries to {output}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
