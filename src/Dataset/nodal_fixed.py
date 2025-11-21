"""Extract fixed node IDs from liver.dat and save a CSV labeling fixed/free nodes.

The file liver.dat contains a load case referencing ``liver_fixed`` where the
fixed node IDs appear two lines after the ``liver_fixed`` string. This module
parses those node IDs, cross-references all node IDs from
``dataset/liver_model_info/liver_coordinates.csv``, and saves a CSV summarizing
whether each node is fixed or not.

Default locations (used when arguments are omitted):
- liver.dat : ``dataset/liver_model/liver.dat``
- coordinates CSV : ``dataset/liver_model_info/liver_coordinates.csv``
- output CSV : ``dataset/liver_model_info/fixed_nodes.csv``

Usage example:

    python src/Dataset/nodal_fixed.py

This will produce a CSV with columns ``node_id``, ``is_fixed`` (boolean), and
``status`` ('fixed' or 'free').
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Set

import pandas as pd


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_paths() -> dict:
    repo_root = _default_repo_root()
    return {
        'dat': repo_root / 'dataset' / 'liver_model' / 'liver.dat',
        'coords': repo_root / 'dataset' / 'liver_model_info' / 'liver_coordinates.csv',
        'output': repo_root / 'dataset' / 'liver_model_info' / 'fixed_nodes.csv',
    }


def extract_ints_from_line(line: str) -> List[int]:
    return [int(x) for x in re.findall(r'-?\d+', line)]


def parse_fixed_nodes_from_dat(dat_path: str | Path | None = None, target_tag: str = 'liver_fixed') -> Set[int]:
    paths = default_paths()
    dat_file = Path(dat_path) if dat_path is not None else Path(paths['dat'])
    if not dat_file.exists():
        raise FileNotFoundError(f"liver.dat not found: {dat_file}")

    lines = dat_file.read_text().splitlines()
    fixed_nodes: Set[int] = set()
    for idx, line in enumerate(lines):
        if target_tag in line:
            candidate_idx = idx + 2
            if candidate_idx < len(lines):
                parsed_ids = extract_ints_from_line(lines[candidate_idx])
                fixed_nodes.update(parsed_ids)

    if not fixed_nodes:
        raise ValueError(f"Fixed node IDs not found using tag '{target_tag}' in {dat_file}")

    return fixed_nodes


def load_all_node_ids(coords_csv: str | Path | None = None) -> List[int]:
    paths = default_paths()
    csv_file = Path(coords_csv) if coords_csv is not None else Path(paths['coords'])
    if not csv_file.exists():
        raise FileNotFoundError(f"Coordinates CSV not found: {csv_file}")

    df = pd.read_csv(csv_file)
    if 'node_id' not in df.columns:
        raise ValueError(f"Coordinates CSV must contain 'node_id' column: {csv_file}")
    return df['node_id'].astype(int).tolist()


def build_fixed_nodes_dataframe(node_ids: Iterable[int], fixed_nodes: Set[int]) -> pd.DataFrame:
    rows = []
    fixed_nodes = set(int(n) for n in fixed_nodes)
    for nid in sorted(int(n) for n in node_ids):
        is_fixed = nid in fixed_nodes
        rows.append({
            'node_id': nid,
            'is_fixed': is_fixed,
            'status': 'fixed' if is_fixed else 'free',
        })
    return pd.DataFrame(rows)


def save_fixed_nodes_csv(df: pd.DataFrame, output_csv: str | Path | None = None) -> Path:
    paths = default_paths()
    out_file = Path(output_csv) if output_csv is not None else Path(paths['output'])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Saved fixed-node summary to: {out_file} (rows: {len(df)})")
    return out_file


def generate_fixed_nodes_csv(dat_path: str | Path | None = None,
                             coords_csv: str | Path | None = None,
                             output_csv: str | Path | None = None,
                             target_tag: str = 'liver_fixed') -> pd.DataFrame:
    fixed_nodes = parse_fixed_nodes_from_dat(dat_path=dat_path, target_tag=target_tag)
    node_ids = load_all_node_ids(coords_csv=coords_csv)
    df = build_fixed_nodes_dataframe(node_ids, fixed_nodes)
    save_fixed_nodes_csv(df, output_csv=output_csv)
    return df


def main():
    paths = default_paths()
    parser = argparse.ArgumentParser(description='Extract fixed node IDs from liver.dat and save CSV with fixed/free labels.')
    parser.add_argument('--dat', default=str(paths['dat']), help='Path to liver.dat (default: %(default)s)')
    parser.add_argument('--coords', default=str(paths['coords']), help='Path to liver_coordinates.csv (default: %(default)s)')
    parser.add_argument('--output', default=str(paths['output']), help='Path to output CSV (default: %(default)s)')
    parser.add_argument('--tag', default='liver_fixed', help='Identifier string that precedes the fixed node list (default: %(default)s)')

    args = parser.parse_args()
    generate_fixed_nodes_csv(dat_path=args.dat, coords_csv=args.coords, output_csv=args.output, target_tag=args.tag)


if __name__ == '__main__':
    main()