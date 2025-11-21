"""CSV データセットを Feather 形式へ変換するためのユーティリティ集。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


class ConversionError(RuntimeError):
	"""CSV から Feather への変換時に発生するエラー。"""


_REQUIRED_TIMES = tuple(range(1, 21))


def _validate_directory(path: Path, *, must_exist: bool) -> Path:
	path = path.resolve()
	if must_exist and not path.is_dir():
		raise NotADirectoryError(f"指定ディレクトリが存在しません: {path}")
	if not must_exist:
		path.mkdir(parents=True, exist_ok=True)
	return path


def _iter_csv_files(root: Path, pattern: str) -> list[Path]:
	files = sorted(p for p in root.glob(pattern) if p.is_file())
	if not files:
		raise FileNotFoundError(f"入力ディレクトリ {root} に一致する CSV がありません ({pattern}).")
	return files


def _convert_single_csv(
	csv_path: Path,
	feather_path: Path,
	*,
	encoding: str,
	compression: Optional[str],
) -> Optional[Path]:
	try:
		frame = pd.read_csv(csv_path, encoding=encoding)
	except Exception as exc:  # pragma: no cover - pandas の例外メッセージが詳細なため
		raise ConversionError(f"CSV の読み込みに失敗しました: {csv_path}") from exc

	if "time" not in frame.columns:
		raise ConversionError(f"CSV に time 列が存在しません: {csv_path}")

	time_series = pd.to_numeric(frame["time"], errors="coerce")
	missing_times = [t for t in _REQUIRED_TIMES if not (time_series == t).any()]
	if missing_times:
		missing_label = ",".join(str(t) for t in missing_times)
		print(
			f"[convert_csv_feather] Skip {csv_path.name} (time に {missing_label} が含まれていません)"
		)
		return None

	mask = time_series.isin(_REQUIRED_TIMES)
	frame = frame.loc[mask].copy()
	frame["time"] = time_series.loc[mask].astype(int)

	feather_path.parent.mkdir(parents=True, exist_ok=True)
	try:
		frame.to_feather(feather_path, compression=compression)
	except ImportError as exc:  # pragma: no cover - 依存不足は起動時に気付きたい
		raise ConversionError(
			"Feather 形式で保存するには `pyarrow` が必要です。`pip install pyarrow` で"
			"インストールしてください。"
		) from exc
	except Exception as exc:  # pragma: no cover - pandas の例外メッセージが詳細なため
		raise ConversionError(f"Feather の書き込みに失敗しました: {feather_path}") from exc

	return feather_path


def convert_directory(
	csv_dir: Path,
	output_dir: Path,
	*,
	glob_pattern: str = "*.csv",
	encoding: str = "utf-8",
	compression: Optional[str] = None,
	skip_existing: bool = False,
) -> list[Path]:
	"""ディレクトリ内の CSV ファイルを Feather 形式へ一括変換する。

	Parameters
	----------
	csv_dir:
		変換元 CSV が格納されたディレクトリ。
	output_dir:
		Feather ファイルを書き出すディレクトリ。
	glob_pattern:
		変換対象 CSV を絞り込むグロブパターン (既定値: ``"*.csv"``)。
	encoding:
		``pandas.read_csv`` に渡すテキストエンコーディング。
	compression:
		Feather 保存時に使用する圧縮方式 (``"snappy"``/``"lz4"``/``"zstd"``/``None``)。
	skip_existing:
		``True`` の場合、既存の Feather ファイルは上書きせずにスキップする。

	Notes
	-----
	変換時には ``time`` 列に 1 から 20 までの値がすべて揃っている CSV のみを採用し、
	該当列を 1~20 の範囲でフィルタリングしたうえで Feather へ書き出す。

	Returns
	-------
	list[Path]
		生成した Feather ファイルのパスをソート順で返す (スキップ分は含まない)。
	"""

	source = _validate_directory(Path(csv_dir), must_exist=True)
	destination = _validate_directory(Path(output_dir), must_exist=False)

	csv_files = _iter_csv_files(source, glob_pattern)
	created: list[Path] = []

	for csv_path in csv_files:
		feather_path = destination / f"{csv_path.stem}.feather"
		if skip_existing and feather_path.exists():
			print(f"[convert_csv_feather] Skip {csv_path.name} (exists)")
			continue

		result = _convert_single_csv(
			csv_path,
			feather_path,
			encoding=encoding,
			compression=compression,
		)
		if result is not None:
			print(f"[convert_csv_feather] Convert {csv_path} -> {feather_path}")
			created.append(result)

	return created


__all__ = [
	"ConversionError",
	"convert_directory",
]
