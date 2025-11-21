from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


NAME_PATTERN = re.compile(r"name of boundary condition\s+(?P<name>\S+)", re.IGNORECASE)
PRESCRIBED_PATTERN = re.compile(
	r"prescribed displacement for dof\s+(?P<dof>\d+)\s*=\s*(?P<value>[0-9.+\-Ee]+)",
	re.IGNORECASE,
)
APPLIED_PATTERN = re.compile(r"applied to node ids", re.IGNORECASE)
ALPHA_PATTERN = re.compile(r"[A-Za-z]")


@dataclass
class BoundaryCondition:
	name: str
	displacements: dict[int, float] = field(default_factory=dict)
	node_ids: list[int] = field(default_factory=list)

	def displacement_vector(self) -> tuple[float, float, float]:
		return (
			self.displacements.get(1, 0.0),
			self.displacements.get(2, 0.0),
			self.displacements.get(3, 0.0),
		)


def _parse_node_ids(line: str) -> list[int]:
	text = line.strip().replace(",", " ")
	if not text or ALPHA_PATTERN.search(text):
		return []
	return [int(token) for token in text.split() if token]


def iter_boundary_conditions(stream: Iterable[str]) -> list[BoundaryCondition]:
	conditions: list[BoundaryCondition] = []
	current_name: Optional[str] = None
	displacements: dict[int, float] = {}
	node_ids: list[int] = []
	capturing_nodes = False
	node_block_started = False

	def finalize_current() -> None:
		nonlocal current_name, displacements, node_ids, capturing_nodes, node_block_started
		if current_name is not None:
			conditions.append(
				BoundaryCondition(
					name=current_name,
					displacements=dict(displacements),
					node_ids=list(node_ids),
				)
			)
		current_name = None
		displacements = {}
		node_ids = []
		capturing_nodes = False
		node_block_started = False

	for raw_line in stream:
		line = raw_line.rstrip("\n")
		name_match = NAME_PATTERN.search(line)
		if name_match:
			if current_name is not None:
				finalize_current()
			current_name = name_match.group("name")
			continue

		if current_name is None:
			continue

		prescribed_match = PRESCRIBED_PATTERN.search(line)
		if prescribed_match:
			dof = int(prescribed_match.group("dof"))
			value = float(prescribed_match.group("value"))
			displacements[dof] = value
			continue

		if APPLIED_PATTERN.search(line):
			capturing_nodes = True
			node_block_started = False
			continue

		if capturing_nodes:
			stripped = line.strip()
			if not stripped:
				if node_block_started:
					capturing_nodes = False
					node_block_started = False
				continue

			nodes = _parse_node_ids(line)
			if nodes:
				node_ids.extend(nodes)
				node_block_started = True
				continue

			if node_block_started:
				capturing_nodes = False
				node_block_started = False
			continue

	finalize_current()
	return conditions


def build_force_dataframe(
	conditions: list[BoundaryCondition], target_name: Optional[str] = None
) -> pd.DataFrame:
	rows: list[dict[str, float | int | str]] = []
	for condition in conditions:
		if target_name and condition.name.lower() != target_name.lower():
			continue
		if not condition.node_ids:
			continue
		dx, dy, dz = condition.displacement_vector()
		for node_id in condition.node_ids:
			rows.append(
				{
					"condition_name": condition.name,
					"node_id": node_id,
					"dx": dx,
					"dy": dy,
					"dz": dz,
				}
			)

	if not rows:
		raise ValueError(
			"指定した境界条件の情報が見つかりませんでした。" if target_name else "節点に適用された境界条件が見つかりませんでした。"
		)

	return pd.DataFrame(rows).sort_values(["condition_name", "node_id"]).reset_index(drop=True)


def extract_force_dataframe(
	result_file: Path,
	*,
	encoding: str = "utf-8",
	condition_name: str = "liver_forced",
) -> pd.DataFrame:
	"""結果ファイルから特定の境界条件の力データを抽出して返す。"""

	if not result_file.exists():
		raise FileNotFoundError(f"入力ファイルが見つかりません: {result_file}")

	with result_file.open("r", encoding=encoding, errors="ignore") as stream:
		conditions = iter_boundary_conditions(stream)

	return build_force_dataframe(conditions, target_name=condition_name)


def _default_output_path(input_path: Path, condition_name: Optional[str]) -> Path:
	stem = input_path.with_suffix("").name
	if condition_name:
		return input_path.parent / f"{stem}_force_{condition_name}.csv"
	return input_path.parent / f"{stem}_force.csv"


def _ensure_parent(path: Path) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	return path


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Marc の .out ファイルから境界条件 (Prescribed Displacement) を抽出し、"
			"節点IDごとの移動量リストをCSVに出力します。"
		)
	)
	parser.add_argument("input", type=Path, help="Marc の .out 結果ファイルへのパス")
	parser.add_argument(
		"-n",
		"--condition-name",
		help="抽出したい境界条件名 (既定: liver_forced)",
	)
	parser.add_argument(
		"-o",
		"--output",
		type=Path,
		help="出力CSVパス (省略時は <入力名>_force_<境界条件>.csv)",
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
	condition_name = args.condition_name or "liver_forced"

	if not input_path.exists():
		raise FileNotFoundError(f"入力ファイルが見つかりません: {input_path}")

	with input_path.open("r", encoding=args.encoding, errors="ignore") as stream:
		conditions = iter_boundary_conditions(stream)

	force_df = build_force_dataframe(conditions, target_name=condition_name)
	output_path = args.output or _default_output_path(input_path, condition_name)
	_ensure_parent(output_path)
	force_df.to_csv(output_path, index=False)
	print(
		f"Wrote {len(force_df)} rows for boundary condition '{condition_name}' to {output_path}"
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

