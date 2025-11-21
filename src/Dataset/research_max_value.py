"""Feather 形式の節点データから絶対値ベースの最大値を抽出するユーティリティ。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from pandas.api.types import is_numeric_dtype


_IGNORE_COLUMNS = {"node_id", "time", "x", "y", "z", "force_node_id", "nearest_fixed_node_id"}


class CoordinateAnalysisError(RuntimeError):
	"""Feather ファイル解析に失敗した場合に送出されるエラー。"""


def _validate_directory(path: Path) -> Path:
	"""入力ディレクトリの存在を確認して絶対パスを返す。"""

	resolved = Path(path).resolve()
	if not resolved.is_dir():
		raise NotADirectoryError(f"指定ディレクトリが存在しません: {resolved}")
	return resolved


def _ensure_output_directory(path: Path) -> Path:
	"""出力先ディレクトリを作成して絶対パスを返す。"""

	resolved = Path(path).resolve()
	resolved.mkdir(parents=True, exist_ok=True)
	return resolved


def _iter_feather_files(root: Path, pattern: str) -> list[Path]:
	"""対象ディレクトリ内の Feather ファイル一覧を取得する。"""

	files = sorted(p for p in root.glob(pattern) if p.is_file())
	if not files:
		raise FileNotFoundError(f"入力ディレクトリ {root} に一致する Feather ファイルがありません ({pattern}).")
	return files


def _update_global_max(global_max: dict[str, float], frame: pd.DataFrame, columns: list[str]) -> None:
	"""全節点での最大値辞書を更新する (絶対値ベース)。"""

	for column in columns:
		value = frame[column].abs().max(skipna=True)
		if pd.isna(value):
			continue
		stored = global_max.get(column)
		if stored is None or value > stored:
			global_max[column] = float(value)


def _update_node_max(node_max: dict[int, dict[str, float]], frame: pd.DataFrame, columns: list[str]) -> None:
	"""各節点の最大値辞書を更新する (絶対値ベース)。"""

	if "node_id" not in frame.columns:
		raise CoordinateAnalysisError("Feather に node_id 列が存在しません。最大値解析を実行できません。")

	absolute_frame = frame.loc[:, ["node_id", *columns]].copy()
	absolute_frame[columns] = absolute_frame[columns].abs()
	grouped = absolute_frame.groupby("node_id", sort=False)[columns].max()
	for node_id, row in grouped.iterrows():
		record = node_max.setdefault(node_id, {})
		for column, value in row.items():
			if pd.isna(value):
				continue
			stored = record.get(column)
			if stored is None or value > stored:
				record[column] = float(value)


def analyze_feather_directory(
	feather_dir: Path,
	*,
	output_dir: Optional[Path] = None,
	glob_pattern: str = "*.feather",
) -> dict[str, Path]:
	"""指定ディレクトリ内の Feather を解析し絶対値最大の集計をファイルへ保存する。

	Parameters
	----------
	feather_dir:
		解析対象となる Feather ファイルを格納したディレクトリ。
	output_dir:
		結果を書き出すディレクトリ。未指定の場合は ``feather_dir`` を用いる。
	glob_pattern:
		読み込む Feather ファイルを決めるグロブパターン。

	Returns
	-------
	dict[str, Path]
		生成したファイルパスをキー ``overall`` と ``per_node`` で返す。

	Notes
	-----
	全節点・各節点ともに各特徴量の絶対値が最大となる値を採用する。
	"""

	input_dir = _validate_directory(Path(feather_dir))
	destination = _ensure_output_directory(Path(output_dir) if output_dir else input_dir)
	files = _iter_feather_files(input_dir, glob_pattern)

	global_max: dict[str, float] = {}
	node_max: dict[int, dict[str, float]] = {}
	observed_columns: set[str] = set()

	for file_path in files:
		print(f"[research_max_value] 解析中: {file_path}")
		frame = pd.read_feather(file_path)
		candidate_columns = [
			column
			for column in frame.columns
			if column not in _IGNORE_COLUMNS and is_numeric_dtype(frame[column])
		]
		if not candidate_columns:
			continue

		observed_columns.update(candidate_columns)
		_update_global_max(global_max, frame, candidate_columns)
		_update_node_max(node_max, frame, candidate_columns)

	if not observed_columns:
		raise CoordinateAnalysisError("最大値を算出できる数値列が見つかりませんでした。")

	sorted_columns = sorted(observed_columns)
	overall_series = pd.Series(global_max).reindex(sorted_columns)
	overall_df = overall_series.rename("max_value").to_frame()
	overall_path = destination / "summary_overall_max_values.csv"
	overall_df.to_csv(overall_path, index_label="feature")

	node_records = []
	for node_id, maxima in node_max.items():
		record = {"node_id": node_id}
		record.update(maxima)
		node_records.append(record)

	if not node_records:
		raise CoordinateAnalysisError("節点ごとの最大値を算出できませんでした。")

	per_node_df = pd.DataFrame(node_records)
	per_node_df = per_node_df.set_index("node_id").sort_index()
	per_node_df = per_node_df.reindex(columns=sorted_columns)
	per_node_path = destination / "summary_per_node_max_values.csv"
	per_node_df.to_csv(per_node_path, index_label="node_id")

	print(f"[research_max_value] 集計完了: overall={overall_path}, per_node={per_node_path}")
	return {"overall": overall_path, "per_node": per_node_path}


__all__ = [
	"CoordinateAnalysisError",
	"analyze_feather_directory",
]
