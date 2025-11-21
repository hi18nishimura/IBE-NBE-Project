"""Dataset orchestration CLI.

現在は `pattern` と `marc_parse` サブコマンドを提供し、いずれも Hydra ベースで
各処理 (`Dataset.pattern.generate_dataset` や Marc出力パース) を実行する。
将来的に他のデータセット生成処理を追加したい場合は、このスクリプトに
サブコマンドを増やしていく想定。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import pandas as pd
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


# Dataset.* モジュールをインポートできるように src/ ディレクトリを sys.path へ追加
REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
	sys.path.append(str(SRC_DIR))

try:
	from Dataset.pattern import MaterialProps, generate_dataset
except ModuleNotFoundError as exc:  # pragma: no cover - 起動時に気付きやすくする
	raise RuntimeError('Dataset.pattern をインポートできません。src/ 配下の構成を確認してください。') from exc

try:
	from Dataset import feature_disp_nodal, feature_stress_elements, feature_stress_nodal
except ModuleNotFoundError as exc:  # pragma: no cover - 同様に起動時に知らせる
	raise RuntimeError('Dataset.feature_* モジュールをインポートできません。src/ 配下の構成を確認してください。') from exc


def _resolve_path(path_like: str) -> Path:
	"""Hydra 実行時に変わるカレントディレクトリを考慮して絶対パス化する"""

	return Path(to_absolute_path(path_like)).resolve()


def _resolve_optional_path(path_like: Optional[str | Path]) -> Optional[Path]:
	if path_like in (None, '', 'null'):
		return None
	return _resolve_path(str(path_like))


CONFIG_DIR = REPO_ROOT / 'config' / 'dataset'


def _invoke_cli(main_func: Callable[[Optional[List[str]]], int], argv: List[str], label: str) -> None:
	"""Call Dataset.* CLI main functions and ensure success."""

	try:
		exit_code = main_func(argv)
	except SystemExit as exc:  # 万一 main 内で SystemExit を投げた場合に備える
		exit_code = exc.code if isinstance(exc.code, int) else 1
	if exit_code != 0:
		raise RuntimeError(f"{label} の実行に失敗しました (exit={exit_code}).")


def _iter_target_files(root: Path, pattern: str) -> list[Path]:
	files = sorted(p for p in root.glob(pattern) if p.is_file())
	if not files:
		raise FileNotFoundError(f"入力ディレクトリ {root} に一致するファイルがありません ({pattern}).")
	return files


def _prepare_output_dirs(base_dir: Path) -> dict[str, Path]:
	subdirs = {
		'disp': base_dir / 'disp_nodal',
		'stress_elements': base_dir / 'stress_elements',
		'stress_nodal': base_dir / 'stress_nodal',
		'all_feature': base_dir / 'all_feature',
	}
	for path in subdirs.values():
		path.mkdir(parents=True, exist_ok=True)
	return subdirs


def _run_feature_disp(
	result_file: Path,
	output_dir: Path,
	*,
	encoding: str,
	force_condition: str,
	fixed_csv: Optional[Path],
) -> Path:
	stem = result_file.stem
	merged_output = output_dir / f"{stem}_disp_with_coords.csv"
	argv: List[str] = [
		str(result_file),
		'--merged-output',
		str(merged_output),
		'--encoding',
		encoding,
		'--force-condition',
		force_condition,
	]
	if fixed_csv:
		argv.extend(['--fixed-csv', str(fixed_csv)])
	_invoke_cli(feature_disp_nodal.main, argv, 'feature_disp_nodal.py')
	return merged_output


def _run_feature_stress_elements(
	result_file: Path,
	output_dir: Path,
	*,
	encoding: str,
) -> Path:
	stem = result_file.stem
	output_path = output_dir / f"{stem}_stress_elements.csv"
	argv = [
		str(result_file),
		'--encoding',
		encoding,
		'--output',
		str(output_path),
	]
	_invoke_cli(feature_stress_elements.main, argv, 'feature_stress_elements.py')
	return output_path


def _run_feature_stress_nodal(
	stress_elements_csv: Path,
	disp_with_coords_csv: Path,
	output_dir: Path,
	*,
	encoding: str,
	connectivity_csv: Path,
) -> Tuple[Path, Path]:
	stem = stress_elements_csv.stem.replace('_stress_elements', '')
	nodal_output = output_dir / f"{stem}_nodal.csv"
	avg_output = output_dir / f"{stem}_nodal_average.csv"
	argv = [
		str(stress_elements_csv),
		str(disp_with_coords_csv),
		'--encoding',
		encoding,
		'--connectivity',
		str(connectivity_csv),
		'--output',
		str(nodal_output),
		'--average-output',
		str(avg_output),
	]
	_invoke_cli(feature_stress_nodal.main, argv, 'feature_stress_nodal.py')
	return nodal_output, avg_output


def _merge_feature_tables(
	displacement_csv: Path,
	stress_csv: Path,
	destination: Path,
	*,
	how: str,
) -> Path:
	disp_df = pd.read_csv(displacement_csv)
	stress_df = pd.read_csv(stress_csv)
	required = {'time', 'node_id'}
	if not required.issubset(disp_df.columns):
		missing = required - set(disp_df.columns)
		raise ValueError(f"disp CSV に必要な列が不足しています: {sorted(missing)}")
	if not required.issubset(stress_df.columns):
		missing = required - set(stress_df.columns)
		raise ValueError(f"stress CSV に必要な列が不足しています: {sorted(missing)}")

	drop_overlap = [col for col in ('x', 'y', 'z') if col in stress_df.columns]
	if drop_overlap:
		stress_df = stress_df.drop(columns=drop_overlap)

	merged = disp_df.merge(stress_df, on=['time', 'node_id'], how=how)
	destination.parent.mkdir(parents=True, exist_ok=True)
	merged.to_csv(destination, index=False)
	return destination


def run_marc_parse_job(
	*,
	input_dir: Path,
	output_dir: Optional[Path],
	glob_pattern: str,
	encoding: str,
	connectivity: Path,
	force_condition: str,
	fixed_csv: Optional[Path],
	merge_how: str,
	skip_existing: bool,
) -> None:
	input_dir = input_dir.resolve()
	if not input_dir.is_dir():
		raise NotADirectoryError(f"入力ディレクトリが存在しません: {input_dir}")

	if merge_how not in {'inner', 'left'}:
		raise ValueError(f"merge_how には 'inner' か 'left' を指定してください (受領: {merge_how})")

	if not connectivity.exists():
		raise FileNotFoundError(f"tetra_connectivity.csv が見つかりません: {connectivity}")
	if fixed_csv is not None and not fixed_csv.exists():
		raise FileNotFoundError(f"固定節点CSVが見つかりません: {fixed_csv}")

	resolved_output = (output_dir or (input_dir / 'marc_parse_output')).resolve()
	dirs = _prepare_output_dirs(resolved_output)
	files = _iter_target_files(input_dir, glob_pattern)
	for file_path in files:
		stem = file_path.stem
		print(f"[marc_parse] Processing {file_path}")

		disp_csv = dirs['disp'] / f"{stem}_disp_with_coords.csv"
		stress_elem_csv = dirs['stress_elements'] / f"{stem}_stress_elements.csv"
		nodal_csv = dirs['stress_nodal'] / f"{stem}_nodal.csv"
		avg_csv = dirs['stress_nodal'] / f"{stem}_nodal_average.csv"
		nodal_all_feature = dirs['all_feature'] / f"{stem}_nodal_features.csv"
		avg_all_feature = dirs['all_feature'] / f"{stem}_nodal_average_features.csv"

		if skip_existing and all(
			path.exists()
			for path in (disp_csv, stress_elem_csv, nodal_csv, avg_csv, nodal_all_feature, avg_all_feature)
		):
			print(f"[marc_parse] Skip {file_path.name} (all outputs exist)")
			continue

		disp_csv = _run_feature_disp(
			file_path,
			dirs['disp'],
			encoding=encoding,
			force_condition=force_condition,
			fixed_csv=fixed_csv,
		)
		stress_elem_csv = _run_feature_stress_elements(
			file_path,
			dirs['stress_elements'],
			encoding=encoding,
		)
		nodal_csv, avg_csv = _run_feature_stress_nodal(
			stress_elem_csv,
			disp_csv,
			dirs['stress_nodal'],
			encoding=encoding,
			connectivity_csv=connectivity,
		)

		_merge_feature_tables(
			disp_csv,
			nodal_csv,
			nodal_all_feature,
			how=merge_how,
		)
		_merge_feature_tables(
			disp_csv,
			avg_csv,
			avg_all_feature,
			how=merge_how,
		)
		print(f"[marc_parse] Done {file_path.name}")


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name='pattern')
def _pattern_hydra_entry(cfg: DictConfig) -> None:
	"""Hydra で読み込んだ設定を generate_dataset に橋渡しする"""

	print('[hydra] 使用設定:\n' + OmegaConf.to_yaml(cfg))

	input_dat = _resolve_path(cfg.input)
	liver_props = MaterialProps(young=float(cfg.liver.E), poisson=float(cfg.liver.nu))
	tumor_props = MaterialProps(young=float(cfg.tumor.E), poisson=float(cfg.tumor.nu))

	max_nodes = cfg.get('max_nodes')
	max_nodes = int(max_nodes) if max_nodes is not None else None

	seed = cfg.get('seed')
	seed = int(seed) if seed is not None else None

	plot_random = bool(cfg.get('plot_random', False))

	generate_dataset(
		mode=str(cfg.mode),
		output_name=str(cfg.output_name),
		input_dat=input_dat,
		liver_props=liver_props,
		tumor_props=tumor_props,
		max_disp=float(cfg.max_disp),
		train_divisions=int(cfg.train_divisions),
		valid_count=int(cfg.valid_count),
		test_count=int(cfg.test_count),
		max_nodes=max_nodes,
		seed=seed,
		plot_random=plot_random,
	)


def run_pattern_hydra(overrides: List[str]) -> None:
	"""Hydra エントリポイントを直接呼び出し、override を渡して実行する"""

	# Hydra は sys.argv の内容をそのまま解析するため、一時的に差し替える
	original_argv = sys.argv
	sys.argv = [__file__] + overrides
	try:
		_pattern_hydra_entry()
	finally:
		sys.argv = original_argv


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name='marc_parse')
def _marc_hydra_entry(cfg: DictConfig) -> None:
	"""Hydra 設定から marc_parse ジョブを実行する"""

	print('[hydra] 使用設定:\n' + OmegaConf.to_yaml(cfg))

	input_dir = _resolve_path(str(cfg.input_dir))
	output_dir = _resolve_optional_path(cfg.get('output_dir'))
	connectivity = _resolve_path(str(cfg.connectivity))
	fixed_csv = _resolve_optional_path(cfg.get('fixed_csv'))
	run_marc_parse_job(
		input_dir=input_dir,
		output_dir=output_dir,
		glob_pattern=str(cfg.glob),
		encoding=str(cfg.encoding),
		connectivity=connectivity,
		force_condition=str(cfg.force_condition),
		fixed_csv=fixed_csv,
		merge_how=str(cfg.merge_how),
		skip_existing=bool(cfg.skip_existing),
	)


def run_marc_parse_hydra(overrides: List[str]) -> None:
	"""Hydra エントリポイント (marc_parse) を override 付きで実行する"""

	original_argv = sys.argv
	sys.argv = [__file__] + overrides
	try:
		_marc_hydra_entry()
	finally:
		sys.argv = original_argv


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description='Dataset generation orchestrator (pattern, ...).')
	subparsers = parser.add_subparsers(dest='command', required=True)

	pattern_parser = subparsers.add_parser('pattern', help='Hydraベースの liver.dat パターン生成を実行')
	pattern_parser.add_argument(
		'overrides',
		nargs='*',
		help='Hydra の override 文字列 (例: mode=valid output_name=run02 max_nodes=2)'
	)

	marc_parser = subparsers.add_parser('marc_parse', help='Marc .out ファイルの一括パースを実行')
	marc_parser.add_argument(
		'overrides',
		nargs='*',
		help='Hydra の override 文字列 (例: input_dir=dataset/marc/sample glob="*.out")'
	)

	return parser


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()

	if args.command == 'pattern':
		run_pattern_hydra(args.overrides or [])
	elif args.command == 'marc_parse':
		run_marc_parse_hydra(args.overrides or [])
	else:  # pragma: no cover - サブコマンド追加時の安全策
		parser.error(f"Unknown command: {args.command}")


if __name__ == '__main__':
	main()
