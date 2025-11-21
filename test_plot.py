from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_COLUMNS = ["Sxx", "Syy", "Szz", "Sxy", "Syz", "Szx"]


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="CSV から応力分布のヒストグラムを作成")
	parser.add_argument("csv", type=Path, help="入力CSV (例: *_stress_elements_nodal.csv)")
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
		default=Path("plots"),
		help="出力画像の保存先 (既定: ./plots)",
	)
	parser.add_argument(
		"--bins",
		type=int,
		default=50,
		help="ヒストグラムのビン数 (既定: %(default)s)",
	)
	return parser


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
	missing = [col for col in columns if col not in df.columns]
	if missing:
		raise ValueError(f"入力CSVに存在しない列があります: {missing}")
	return list(columns)


def plot_histograms(df: pd.DataFrame, columns: list[str], output_dir: Path, bins: int) -> list[Path]:
	output_dir.mkdir(parents=True, exist_ok=True)
	paths: list[Path] = []
	for column in columns:
		series = df[column].dropna()
		fig, ax = plt.subplots(figsize=(6, 4))
		ax.hist(series, bins=bins, color="#4e79a7", alpha=0.85)
		ax.set_title(f"Histogram of {column}")
		ax.set_xlabel(column)
		ax.set_ylabel("Frequency")
		ax.grid(True, linestyle=":", alpha=0.4)
		fig.tight_layout()
		output_path = output_dir / f"hist_{column}.png"
		fig.savefig(output_path, dpi=200)
		plt.close(fig)
		paths.append(output_path)
	return paths


def summarize(df: pd.DataFrame, columns: list[str], output_dir: Path) -> Path:
	summary = df[columns].describe().transpose()
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "summary_statistics.csv"
	summary.to_csv(output_path)
	return output_path


def main(argv: Optional[list[str]] = None) -> int:
	parser = build_parser()
	args = parser.parse_args(argv)

	if not args.csv.exists():
		raise FileNotFoundError(f"入力CSVが見つかりません: {args.csv}")

	df = pd.read_csv(args.csv)
	columns = ensure_columns(df, args.columns or DEFAULT_COLUMNS)
	print(f"Loaded {len(df)} rows from {args.csv}")

	image_paths = plot_histograms(df, columns, args.output_dir, args.bins)
	summary_path = summarize(df, columns, args.output_dir)

	print("Generated the following histogram images:")
	for path in image_paths:
		print(f" - {path}")
	print(f"Summary statistics saved to: {summary_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
