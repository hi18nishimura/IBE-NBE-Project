"""Build node adjacency from tetra connectivity and save per-node connection CSV.

This module reads a tetra connectivity CSV with columns ['mesh_id','n1','n2','n3','n4']
and builds an adjacency list where two nodes are connected if they appear in the
same tetra element. It can save a CSV with one row per node containing:

- node_id: integer node id
- degree: number of connected neighbor nodes
- neighbors: comma-separated neighbor node ids (sorted)
- neighbor_degrees: (optional) comma-separated degrees of each neighbor in same order

Default input file (if none provided):
  dataset/liver_model_info/tetra_connectivity.csv
Default output file (if none provided):
  dataset/liver_model_info/node_connections.csv

Example usage:
  python src/Dataset/nodal_connect.py

"""
from pathlib import Path
from typing import Dict, Set, Optional
import pandas as pd
import argparse


def build_adjacency_from_tetra_csv(tetra_csv: str) -> Dict[int, Set[int]]:
    """Read tetra connectivity CSV and build node adjacency (set of neighbor ids).

    Parameters
    ----------
    tetra_csv : str
        Path to CSV with columns ['mesh_id','n1','n2','n3','n4']

    Returns
    -------
    Dict[int, Set[int]]
        node_id -> set of neighbor node_ids
    """
    p = Path(tetra_csv)
    if not p.exists():
        raise FileNotFoundError(f"tetra connectivity CSV not found: {p}")

    df = pd.read_csv(p)
    required = {'n1', 'n2', 'n3', 'n4'}
    if not required.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns: {required}")

    adj: Dict[int, Set[int]] = {}
    for _, row in df.iterrows():
        nodes = [int(row['n1']), int(row['n2']), int(row['n3']), int(row['n4'])]
        for i, a in enumerate(nodes):
            adj.setdefault(a, set())
            for j, b in enumerate(nodes):
                if i == j:
                    continue
                adj[a].add(int(b))

    return adj


def compute_degrees(adj: Dict[int, Set[int]]) -> Dict[int, int]:
    return {nid: len(neigh) for nid, neigh in adj.items()}


def save_node_connections_csv(adj: Dict[int, Set[int]], output_csv: str, include_neighbor_degrees: bool = False) -> Path:
    """Save node adjacency to CSV.

    Output columns: node_id, degree, neighbors, [neighbor_degrees]
    """
    outp = Path(output_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)

    degrees = compute_degrees(adj)
    rows = []
    for nid in sorted(adj.keys()):
        neighbors = sorted(adj[nid])
        neigh_str = ','.join(str(x) for x in neighbors)
        row = {'node_id': int(nid), 'degree': int(degrees.get(nid, 0)), 'neighbors': neigh_str}
        if include_neighbor_degrees:
            neigh_deg = [degrees.get(x, 0) for x in neighbors]
            row['neighbor_degrees'] = ','.join(str(x) for x in neigh_deg)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(outp, index=False)
    print(f"Saved node connections to: {outp} (rows: {len(df)})")
    return outp


def default_paths():
    repo_root = Path(__file__).resolve().parents[2]
    default_in = repo_root / 'dataset' / 'liver_model_info' / 'tetra_connectivity.csv'
    default_out = repo_root / 'dataset' / 'liver_model_info' / 'node_connections.csv'
    return str(default_in), str(default_out)


def main():
    default_in, default_out = default_paths()
    parser = argparse.ArgumentParser(description='Build node adjacency from tetra connectivity CSV and save per-node CSV')
    parser.add_argument('--input', '-i', default=default_in, help='input tetra_connectivity.csv')
    parser.add_argument('--output', '-o', default=default_out, help='output node connections CSV')
    parser.add_argument('--neighbor-degrees', action='store_true', help='include neighbor degrees column')

    args = parser.parse_args()

    adj = build_adjacency_from_tetra_csv(args.input)
    save_node_connections_csv(adj, args.output, include_neighbor_degrees=args.neighbor_degrees)


if __name__ == '__main__':
    main()


def read_node_connections(file_path: str = None, return_type: str = 'dataframe'):
    """Read node connections CSV and return as DataFrame or dictionary.

    Parameters
    ----------
    file_path : str, optional
        Path to node connections CSV. If None, default is the same
        path used by save_node_connections_csv (dataset/liver_model_info/node_connections.csv).
    return_type : str, optional
        - 'dataframe' : return pandas.DataFrame (default)
        - 'dict_by_node' or 'by_node' : return dict mapping node_id -> list of neighbor ids

    Returns
    -------
    pd.DataFrame or dict
    """
    # default output path is the saved location; use it when file_path is None
    if file_path is None:
        _, default_out = default_paths()
        file_path = default_out

    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"node connections CSV not found: {p}")

    df = pd.read_csv(p)

    if return_type == 'dataframe':
        return df
    elif return_type in ('dict_by_node', 'by_node'):
        node_map = {}
        for _, row in df.iterrows():
            nid = int(row['node_id'])
            neighs = []
            if 'neighbors' in row and pd.notna(row['neighbors']):
                s = str(row['neighbors']).strip()
                if s:
                    neighs = [int(x) for x in s.split(',') if x != '']
            node_map[nid] = neighs
        return node_map
    else:
        raise ValueError("return_type must be one of: 'dataframe', 'dict_by_node' (or 'by_node')")
