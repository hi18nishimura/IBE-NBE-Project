from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_COLUMNS = ["dx","dy","dz","Sxx", "Syy", "Szz", "Sxy", "Syz", "Szx"]


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="ディレクトリ内の .feather を結合して応力分布のヒストグラムを作成")
	parser.add_argument("input_dir", type=Path, help=".feather ファイルを含むディレクトリ")
	parser.add_argument(
		"-c",
		"--columns",
		nargs="+",
		help="可視化対象の列名 (既定: Sxx〜Szx)",
	)
	parser.add_argument(
		"-o",
		"--output-dir",
		type=Path,
		default=Path("plots/dataset"),
		help="出力画像/要約CSVの保存先 (既定: ./plots/dataset)",
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
		"--log-plot",
		action="store_true",
		help="縦軸を対数スケールで表示します",
	)
	return parser


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> List[str]:
	missing = [col for col in columns if col not in df.columns]
	if missing:
		raise ValueError(f"入力データに存在しない列があります: {missing}")
	return list(columns)


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


def plot_histograms(df: pd.DataFrame, columns: List[str], output_dir: Path, bins: int, log_plot: bool = False) -> List[Path]:
	output_dir.mkdir(parents=True, exist_ok=True)
	paths: List[Path] = []
	for column in columns:
		series = df[column].dropna()
		fig, ax = plt.subplots(figsize=(6, 4))
		ax.hist(series, bins=bins, color="#4e79a7", alpha=0.85)
		if log_plot:
			ax.set_yscale('log')
		ax.set_title(f"Histogram of {column}")
		ax.set_xlabel(column)
		ax.set_ylabel("Frequency")
		ax.grid(True, linestyle=":", alpha=0.4)
		fig.tight_layout()
		if log_plot:
			output_path = output_dir / f"hist_{column}_log.png"
		else:
			output_path = output_dir / f"hist_{column}.png"
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
	columns = ensure_columns(combined, args.columns or DEFAULT_COLUMNS)
	print(f"Loaded {len(combined)} rows from {args.input_dir} (pattern={args.glob})")

	image_paths = plot_histograms(combined, columns, args.output_dir, args.bins, log_plot=args.log_plot)
	summary_path = summarize(combined, columns, args.output_dir)

	print("Generated the following histogram images:")
	for path in image_paths:
		print(f" - {path}")
	print(f"Summary statistics saved to: {summary_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
