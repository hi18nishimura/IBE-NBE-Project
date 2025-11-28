from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_COLUMNS = ["dx", "dy", "dz", "Sxx", "Syy", "Szz", "Sxy", "Syz", "Szx"]


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Oka 正規化を行いヒストグラムを作成")
	parser.add_argument("input_dir", type=Path, help=".feather ファイルを含むディレクトリ")
	parser.add_argument(
		"-c",
		"--columns",
		nargs="+",
		help="正規化・可視化対象の列名 (既定: dx,dy,dz,Sxx〜Szx)",
	)
	parser.add_argument(
		"-o",
		"--output-dir",
		type=Path,
		default=Path("plots/oka"),
		help="出力画像/要約CSVの保存先 (既定: ./plots/oka)",
	)
	parser.add_argument(
		"--glob",
		default="*.feather",
		help='探索パターン (既定: "*.feather"。サブディレクトリまで探す場合は "**/*.feather")',
	)
	parser.add_argument(
		"--bins",
		type=int,
		default=50,
		help="ヒストグラムのビン数 (既定: %(default)s)",
	)
	parser.add_argument(
		"--alpha",
		type=float,
		default=8.0,
		help="Oka 正規化の alpha (既定: 8.0)。alpha>0 を指定してください",
	)
	parser.add_argument(
		"--pwidth",
		type=float,
		help="全列に共通で使う pwidth を指定 (省略時は列ごとの最大絶対値を使用)",
	)
	parser.add_argument(
		"--log-plot",
		action="store_true",
		help="縦軸を対数スケールで表示します",
	)
	return parser


def read_feather_files(directory: Path, pattern: str) -> pd.DataFrame:
	if not directory.exists() or not directory.is_dir():
		raise NotADirectoryError(f"入力ディレクトリが見つかりません: {directory}")
	files = sorted(directory.glob(pattern))
	if not files:
		raise FileNotFoundError(f"指定パターンに一致する .feather ファイルが見つかりません: {pattern}")
	dfs = []
	for fp in files:
		try:
			df = pd.read_feather(fp)
			df = df.reset_index(drop=True)
			df["__source_file"] = fp.name
			dfs.append(df)
		except Exception as exc:
			raise RuntimeError(f"Feather の読み込みに失敗しました: {fp}: {exc}") from exc
	combined = pd.concat(dfs, ignore_index=True)
	return combined


def oka_normalize_series(series: pd.Series, pwidth: float, alpha: float) -> pd.Series:
	"""Oka の式に従って series を正規化して返す。

	p_out = sign(p_in) * (0.4 / pwidth^(1/alpha)) * |p_in|^(1/alpha) + 0.5
	"""
	if alpha <= 0:
		raise ValueError("alpha は正の値を指定してください")
	vals = series.to_numpy(dtype=float)
	signs = np.sign(vals)
	absvals = np.abs(vals)
	# pwidth が 0 のときはすべて 0.5 とする（情報がないため中央に置く）
	if pwidth == 0 or np.isnan(pwidth):
		return pd.Series(np.full_like(vals, 0.5, dtype=float), index=series.index)
	factor = 0.4 / (pwidth ** (1.0 / alpha))
	# avoid negative/NaN issues: keep NaN as NaN
	with np.errstate(invalid='ignore'):
		normed = signs * factor * (absvals ** (1.0 / alpha)) + 0.5
	return pd.Series(normed, index=series.index)


def plot_histograms(df: pd.DataFrame, columns: List[str], output_dir: Path, bins: int, alpha: float, log_plot: bool = False) -> List[Path]:
	output_dir.mkdir(parents=True, exist_ok=True)
	paths: List[Path] = []
	for column in columns:
		series = df[column].dropna()
		fig, ax = plt.subplots(figsize=(6, 4))
		ax.hist(series, bins=bins, color="#4e79a7", alpha=0.85)
		if log_plot:
			ax.set_yscale('log')
		ax.set_title(f"Histogram of {column} (Oka-normalized)")
		ax.set_xlabel(column)
		ax.set_ylabel("Frequency")
		ax.grid(True, linestyle=":", alpha=0.4)
		fig.tight_layout()
		output_path = output_dir / f"hist_{column}_{alpha}.png"
		fig.savefig(output_path, dpi=200)
		plt.close(fig)
		paths.append(output_path)
	return paths


def summarize(df: pd.DataFrame, columns: List[str], output_dir: Path) -> Path:
	summary = df[columns].describe().transpose()
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "summary_statistics.csv"
	summary.to_csv(output_path)
	return output_path


def main(argv: Optional[List[str]] = None) -> int:
	parser = build_parser()
	args = parser.parse_args(argv)

	combined = read_feather_files(args.input_dir, args.glob)
	columns = args.columns or DEFAULT_COLUMNS
	# validate columns exist
	missing = [col for col in columns if col not in combined.columns]
	if missing:
		raise ValueError(f"入力データに存在しない列があります: {missing}")

	# determine pwidth per column (use provided pwidth for all if specified)
	pwidth_map: dict[str, float] = {}
	for col in columns:
		if args.pwidth is not None:
			pwidth_map[col] = float(args.pwidth)
		else:
				# use max value in data for the column (p_width はその変数の最大値)
				pw = float(combined[col].max()) if not combined[col].empty else 0.0
				# if maximum is non-positive, treat as missing (map to 0 -> center 0.5)
				if pw <= 0.0 or np.isnan(pw):
					pw = 0.0
				pwidth_map[col] = pw

	# apply oka normalization column-wise and build normalized df
	normed_df = combined[[]].copy()
	for col in columns:
		pwidth = pwidth_map[col]
		normed = oka_normalize_series(combined[col], pwidth=pwidth, alpha=args.alpha)
		# name normalized column same as original (overwrite) so plotting and summary use normalized values
		normed_df[col] = normed

	print(f"Loaded {len(combined)} rows from {args.input_dir} (pattern={args.glob})")

	image_paths = plot_histograms(normed_df, columns, args.output_dir, args.bins, args.alpha, log_plot=args.log_plot)
	summary_path = summarize(normed_df, columns, args.output_dir)

	# also save normalized combined dataframe as feather for downstream use
	try:
		out_feather = args.output_dir / "normalized_dataset.feather"
		normed_df.reset_index(drop=True).to_feather(out_feather)
	except Exception:
		# non-fatal
		out_feather = None

	print("Generated the following histogram images:")
	for path in image_paths:
		print(f" - {path}")
	print(f"Summary statistics saved to: {summary_path}")
	if out_feather is not None:
		print(f"Normalized dataset saved to: {out_feather}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
