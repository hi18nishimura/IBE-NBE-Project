from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Iterator, Optional

import pandas as pd


FLOAT_PATTERN = re.compile(r"[+\-]?\d+(?:\.\d*)?(?:[Ee][+\-]?\d+)?")
INCREMENT_PATTERN = re.compile(
	 r"output for increment\s+(?P<increment>\d+)",
	 re.IGNORECASE,
)
ELEMENT_PATTERN = re.compile(
	 r"element\s+(?P<element>\d+)\s+point\s+(?P<point>\d+).*?coordinate=\s*"
	 r"(?P<x>[+\-]?\d+(?:\.\d*)?(?:[Ee][+\-]?\d+)?)\s+"
	 r"(?P<y>[+\-]?\d+(?:\.\d*)?(?:[Ee][+\-]?\d+)?)\s+"
	 r"(?P<z>[+\-]?\d+(?:\.\d*)?(?:[Ee][+\-]?\d+)?)",
	 re.IGNORECASE,
)
PK_PATTERN = re.compile(r"PK2str\s+(?P<values>.+)", re.IGNORECASE)


def _extract_floats(text: str) -> list[float]:
	return [float(token) for token in FLOAT_PATTERN.findall(text)]


def iter_element_stresses(stream: Iterable[str]) -> Iterator[dict[str, float | int | None]]:
	current_increment: Optional[int] = None
	pending: Optional[dict[str, float | int | None]] = None

	for raw_line in stream:
		line = raw_line.rstrip("\n")

		increment_match = INCREMENT_PATTERN.search(line)
		if increment_match:
			current_increment = int(increment_match.group("increment"))
			continue

		element_match = ELEMENT_PATTERN.search(line)
		if element_match:
			pending = {
				"time": current_increment,
				"element_id": int(element_match.group("element")),
				"point_id": int(element_match.group("point")),
				"x": float(element_match.group("x")),
				"y": float(element_match.group("y")),
				"z": float(element_match.group("z")),
			}
			continue

		if pending:
			pk_match = PK_PATTERN.search(line)
			if pk_match:
				values = _extract_floats(pk_match.group("values"))
				if len(values) >= 6:
					sxx, syy, szz, sxy, syz, szx = values[-6:]
					row = dict(pending)
					row.update(
						{
							"Sxx": sxx,
							"Syy": syy,
							"Szz": szz,
							"Sxy": sxy,
							"Syz": syz,
							"Szx": szx,
						}
					)
					yield row
				pending = None


def build_stress_dataframe(rows: Iterable[dict[str, float | int | None]]) -> pd.DataFrame:
	data = list(rows)
	if not data:
		raise ValueError("応力データが見つかりませんでした。")
	df = pd.DataFrame(data)
	sort_columns = [col for col in ["time", "element_id", "point_id"] if col in df.columns]
	return df.sort_values(sort_columns).reset_index(drop=True)


def extract_element_stresses(
	result_file: Path,
	*,
	encoding: str = "utf-8",
) -> pd.DataFrame:
	if not result_file.exists():
		raise FileNotFoundError(f"入力ファイルが見つかりません: {result_file}")
	with result_file.open("r", encoding=encoding, errors="ignore") as stream:
		rows = iter_element_stresses(stream)
		return build_stress_dataframe(rows)


def _default_output_path(input_path: Path) -> Path:
	stem = input_path.with_suffix("").name
	return input_path.parent / f"{stem}_stress_elements.csv"


def _ensure_parent(path: Path) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	return path


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Marc の .out ファイルから各要素・積分点の応力 (PK2) を抽出し、"
			"Sxx〜Szx を含むCSVに出力します。"
		)
	)
	parser.add_argument("input", type=Path, help="Marc の .out 結果ファイルへのパス")
	parser.add_argument("-o", "--output", type=Path, help="出力CSVパス")
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
	stress_df = extract_element_stresses(input_path, encoding=args.encoding)
	output_path = args.output or _default_output_path(input_path)
	_ensure_parent(output_path)
	stress_df.to_csv(output_path, index=False)
	print(f"Wrote {len(stress_df)} stress rows to {output_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
