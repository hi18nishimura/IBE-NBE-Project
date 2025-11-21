"""liver.dat の `liver_forced` ブロックを書き換えて大量の入力パターンを生成するスクリプト。

日本語コメント付きで、旧 make_patern.py の機能（train/valid/test 各モード）を
より柔軟に扱えるように実装している。生成された .dat ファイルはすべて
`dataset/liver_model_pattern/<出力ディレクトリ>` に保存される。
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import sys

import numpy as np
import pandas as pd

try:
	from Dataset import nodal_fixed
except ModuleNotFoundError:  # 直接実行時は親ディレクトリをパスに追加
	sys.path.append(str(Path(__file__).resolve().parents[1]))
	from Dataset import nodal_fixed


# ルートディレクトリと主要パスを定義
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DAT = REPO_ROOT / 'dataset' / 'liver_model' / 'liver.dat'
DEFAULT_OUTPUT_BASE = REPO_ROOT / 'dataset' / 'liver_model_pattern'
DEFAULT_FIXED_CSV = REPO_ROOT / 'dataset' / 'liver_model_info' / 'fixed_nodes.csv'


# 旧 make_patern.py に記載されていた外部節点の候補リスト
FORCE_NODE_CANDIDATES = [
	1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
	22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
	35, 36, 38,
	45, 46, 49,
	51, 52, 53, 54, 55,
	58, 59, 60, 61,
	63, 64,
	66, 67, 68,
	71,
	74, 75, 76, 77,
	79, 80,
	83, 85, 86,
	88, 89, 90, 91, 92,
	95, 96, 97,
	99, 100, 101,
	104, 105,
	108, 109, 110, 111,
	114, 115, 116,
	118, 119, 120, 121, 122,
	124, 125,
	127, 128, 129, 130, 131, 132, 133,
	135, 136, 137, 138, 139, 140,
]


@dataclass
class MaterialProps:
	"""材料パラメータをまとめる単純なデータクラス"""

	young: float
	poisson: float


def ensure_fixed_nodes_csv(target_csv: Path = DEFAULT_FIXED_CSV) -> None:
	"""固定節点CSVが存在しなければ nodal_fixed から生成する"""

	if target_csv.exists():
		return
	print(f"[info] 固定節点CSVが見つからないため新規作成します: {target_csv}")
	nodal_fixed.generate_fixed_nodes_csv(output_csv=str(target_csv))


def load_fixed_nodes(target_csv: Path = DEFAULT_FIXED_CSV) -> set[int]:
	"""固定節点CSVを読み込み固定節点IDの集合を返す"""

	ensure_fixed_nodes_csv(target_csv)
	df = pd.read_csv(target_csv)
	if 'node_id' not in df.columns:
		raise ValueError(f"固定節点CSVに node_id 列がありません: {target_csv}")

	flag_col = None
	for col in ('is_fixed', 'fixed'):
		if col in df.columns:
			flag_col = col
			break
	if flag_col is None:
		raise ValueError(f"固定節点CSVに固定フラグ列がありません (is_fixed/fixed)")

	return set(df.loc[df[flag_col].astype(bool), 'node_id'].astype(int).tolist())


def available_force_nodes() -> List[int]:
	"""固定節点を除外した力点候補リストを返す"""

	fixed = load_fixed_nodes()
	nodes = [node for node in FORCE_NODE_CANDIDATES if node not in fixed]
	if not nodes:
		raise RuntimeError('全ての候補節点が固定節点でした。データを確認してください。')
	return nodes


def edit_liver_dat(
	input_dat: Path,
	output_dat: Path,
	liver_props: MaterialProps,
	tumor_props: MaterialProps,
	force_node: int,
	displacement: Sequence[float],
) -> None:
	"""liver.dat を読み込み、材料物性と liver_forced ブロックを書き換えて保存する"""

	lines = input_dat.read_text(encoding='utf-8').splitlines(keepends=True)

	# 材料ブロックの物性を書き換えるヘルパー
	def update_material_block(keyword: str, props: MaterialProps) -> None:
		for idx, line in enumerate(lines):
			if keyword in line:
				target = idx + 2  # キーワードの2行後にヤング率とポアソン比が並ぶ
				if target >= len(lines):
					raise ValueError(f"{keyword} ブロックの書式が不正です")
				lines[target] = (
					f"   {props.young:.6f},  {props.poisson:.6f},        0.,        0.,        0.,        0.,        0.,        0.,\n"
				)
				return
		raise ValueError(f"{keyword} がファイル中に見つかりませんでした")

	update_material_block('"liver_material"', liver_props)
	update_material_block('"tumor_material"', tumor_props)

	# liver_forced ブロックの探索
	forced_idx = None
	for idx, line in enumerate(lines):
		if re.search(r'liver_forced\s*', line):
			forced_idx = idx
			break
	if forced_idx is None:
		raise ValueError('liver_forced ブロックが見つかりません')

	disp_line_idx = forced_idx + 1  # 変位量が書かれている行
	node_line_idx = forced_idx + 5  # 移動させる節点IDが書かれている行
	if node_line_idx >= len(lines):
		raise ValueError('liver_forced ブロックの書式が想定と異なります')

	fx, fy, fz = displacement
	lines[disp_line_idx] = f"   {fx:.4f},   {fy:.4f},   {fz:.4f},\n"
	lines[node_line_idx] = f"{int(force_node)}\n"

	output_dat.parent.mkdir(parents=True, exist_ok=True)
	output_dat.write_text(''.join(lines), encoding='utf-8')


def linspace_displacements(max_disp: float, divisions: int) -> List[Tuple[float, float, float]]:
	"""train モードで使用する等間隔グリッドの変位リストを生成"""

	values = np.linspace(-max_disp, max_disp, divisions)
	combos: List[Tuple[float, float, float]] = []
	for x in values:
		for y in values:
			for z in values:
				if math.isclose(x, 0.0) and math.isclose(y, 0.0) and math.isclose(z, 0.0):
					continue
				combos.append((float(x), float(y), float(z)))
	return combos


def random_displacements(
	rng: np.random.Generator,
	count: int,
	min_abs: float,
	max_abs: float,
) -> List[Tuple[float, float, float]]:
	"""valid/test モードで使用する乱数変位リストを生成"""

	disps: List[Tuple[float, float, float]] = []
	for _ in range(count):
		magnitude = rng.uniform(min_abs, max_abs, size=3)
		signs = rng.choice([-1, 1], size=3)
		vec = magnitude * signs
		disps.append(tuple(float(v) for v in vec))
	return disps


def format_output_name(force_node: int, disp: Sequence[float]) -> str:
	"""出力ファイル名を node/変位ごとに一意になるよう整形"""

	return f"liver_out_node{force_node}_dx{disp[0]:.1f}_dy{disp[1]:.1f}_dz{disp[2]:.1f}.dat"


def save_random_log(records: List[dict], output_dir: Path, mode: str, enable_plot: bool) -> None:
	"""valid/test で生成した乱数変位をCSVと散布図で保存"""

	if not records:
		return
	df = pd.DataFrame(records)
	csv_path = output_dir / f"{mode}_rand_move_list.csv"
	df.to_csv(csv_path, index=False)
	print(f"[info] 乱数変位の履歴を保存しました: {csv_path}")

	if not enable_plot:
		return
	try:
		import matplotlib.pyplot as plt
		from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
	except Exception as exc:  # pragma: no cover
		print(f"[warn] matplotlib の読み込みに失敗したため散布図をスキップします: {exc}")
		return

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(df['dx'], df['dy'], df['dz'], s=10, alpha=0.6)
	ax.set_xlabel('dx')
	ax.set_ylabel('dy')
	ax.set_zlabel('dz')
	ax.set_title(f'Random Displacements ({mode})')
	img_path = output_dir / f"{mode}_rand_move_scatter.png"
	fig.savefig(img_path, dpi=150, bbox_inches='tight')
	plt.close(fig)
	print(f"[info] 乱数散布図を保存しました: {img_path}")


def create_run_directory(name: str) -> Path:
	"""出力先の run ディレクトリを確実に作成して返す"""

	DEFAULT_OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
	run_dir = DEFAULT_OUTPUT_BASE / name
	run_dir.mkdir(parents=True, exist_ok=True)
	return run_dir


def generate_dataset(
	mode: str,
	output_name: str,
	input_dat: Path,
	liver_props: MaterialProps,
	tumor_props: MaterialProps,
	max_disp: float,
	train_divisions: int,
	valid_count: int,
	test_count: int,
	max_nodes: int | None,
	seed: int | None,
	plot_random: bool,
) -> None:
	"""モード別の変位を組み合わせて liver.dat を大量生成するメイン処理"""

	output_dir = create_run_directory(output_name)
	nodes = available_force_nodes()
	if max_nodes is not None:
		nodes = nodes[:max_nodes]
	if not nodes:
		raise RuntimeError('処理対象の節点が空です (max_nodes の制限に注意)。')

	rng = np.random.default_rng(seed)
	total = 0
	random_records: List[dict] = []

	if mode == 'train':
		disps = linspace_displacements(max_disp=max_disp, divisions=train_divisions)
		if train_divisions <= 1:
			step = 0.0
		else:
			step = (max_disp * 2) / (train_divisions - 1)
		print(
			f"[info] train: 1節点あたり {len(disps)} 通りの変位を生成します"
			f" / 各軸の刻み幅: {step:.4f} (max_disp={max_disp})"
		)
		for node in nodes:
			for disp in disps:
				fname = format_output_name(node, disp)
				edit_liver_dat(input_dat, output_dir / fname, liver_props, tumor_props, node, disp)
				total += 1
	else:
		if mode == 'valid':
			min_abs = 0.0
			count = valid_count
		elif mode == 'test':
			min_abs = max_disp / 2.0
			count = test_count
		else:
			raise ValueError(f"未知のモード指定です: {mode}")
		print(f"[info] {mode}: 各節点につき {count} 件の乱数変位を生成します")
		for node in nodes:
			disps = random_displacements(rng, count=count, min_abs=min_abs, max_abs=max_disp)
			for disp in disps:
				fname = format_output_name(node, disp)
				edit_liver_dat(input_dat, output_dir / fname, liver_props, tumor_props, node, disp)
				random_records.append({'force_node': node, 'dx': disp[0], 'dy': disp[1], 'dz': disp[2]})
				total += 1
		save_random_log(random_records, output_dir, mode, plot_random)

	print(f"[done] {output_dir} に {total} 件のファイルを出力しました")


def parse_args() -> argparse.Namespace:
	"""CLI 引数を定義"""

	parser = argparse.ArgumentParser(description='liver.dat の力点と変位を自動生成するツール')
	parser.add_argument('--mode', choices=['train', 'valid', 'test'], required=True, help='生成モード (train/valid/test)')
	parser.add_argument('--output-name', required=True, help='dataset/liver_model_pattern 以下に作成するディレクトリ名')
	parser.add_argument('--input', default=str(DEFAULT_INPUT_DAT), help='元となる liver.dat のパス (default: %(default)s)')
	parser.add_argument('--liver-e', type=float, default=0.683, help='肝臓材料のヤング率 (default: %(default)s)')
	parser.add_argument('--liver-nu', type=float, default=0.49, help='肝臓材料のポアソン比 (default: %(default)s)')
	parser.add_argument('--tumor-e', type=float, default=3.415, help='腫瘍材料のヤング率 (default: %(default)s)')
	parser.add_argument('--tumor-nu', type=float, default=0.40, help='腫瘍材料のポアソン比 (default: %(default)s)')
	parser.add_argument('--max-disp', type=float, default=30.0, help='変位の最大絶対値 (default: %(default)s)')
	parser.add_argument('--train-divisions', type=int, default=4, help='train モードで軸を何分割するか (default: %(default)s)')
	parser.add_argument('--valid-count', type=int, default=8, help='valid モードで1節点あたり何サンプル生成するか (default: %(default)s)')
	parser.add_argument('--test-count', type=int, default=5, help='test モードで1節点あたり何サンプル生成するか (default: %(default)s)')
	parser.add_argument('--max-nodes', type=int, help='処理する節点数の上限 (先頭から順に使用)')
	parser.add_argument('--seed', type=int, help='valid/test 用乱数シード')
	parser.add_argument('--plot-random', action='store_true', help='valid/test で乱数散布図を保存するか')
	return parser.parse_args()


def main() -> None:
	"""エントリポイント"""

	args = parse_args()
	input_dat = Path(args.input)
	if not input_dat.exists():
		raise FileNotFoundError(f"入力ファイルが存在しません: {input_dat}")

	liver_props = MaterialProps(young=args.liver_e, poisson=args.liver_nu)
	tumor_props = MaterialProps(young=args.tumor_e, poisson=args.tumor_nu)

	generate_dataset(
		mode=args.mode,
		output_name=args.output_name,
		input_dat=input_dat,
		liver_props=liver_props,
		tumor_props=tumor_props,
		max_disp=args.max_disp,
		train_divisions=args.train_divisions,
		valid_count=args.valid_count,
		test_count=args.test_count,
		max_nodes=args.max_nodes,
		seed=args.seed,
		plot_random=args.plot_random,
	)


if __name__ == '__main__':
	main()
