from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _ensure_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def load_fixed_nodes(fixed_csv: Path) -> pd.DataFrame:
    _ensure_path(fixed_csv)
    df = pd.read_csv(fixed_csv)
    required_cols = {"node_id", "is_fixed"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{fixed_csv} must contain columns: {required_cols}")
    df["node_id"] = df["node_id"].astype(int)
    df["is_fixed"] = df["is_fixed"].astype(bool)
    return df[df["is_fixed"]]


def compute_fixed_node_distances(
    combine_df: pd.DataFrame,
    fixed_csv: Path,
    time_column: str = "time",
) -> pd.DataFrame:
    required_cols = {"time", "node_id", "x", "y", "z"}
    if not required_cols.issubset(combine_df.columns):
        raise ValueError("Coordinates DataFrame must include node_id,x,y,z columns")

    has_time = time_column in combine_df.columns
    selected_cols = ["node_id", "x", "y", "z"] + ([time_column] if has_time else [])
    coords = combine_df[selected_cols].copy()
    coords["node_id"] = coords["node_id"].astype(int)
    fixed_nodes = load_fixed_nodes(fixed_csv)
    if fixed_nodes.empty:
        raise ValueError("No fixed nodes found in fixed_nodes.csv")

    grouped = coords.groupby(time_column, sort=True) if has_time else [(None, coords)]

    rows: list[dict[str, float | int]] = []
    for time_value, group in grouped:
        fixed_coords = group.merge(fixed_nodes[["node_id"]], on="node_id", how="inner")
        if fixed_coords.empty:
            label = f" (time={time_value})" if has_time else ""
            raise ValueError(f"No fixed nodes present in coordinates{label}")

        fixed_points = fixed_coords[["node_id", "x", "y", "z"]].to_numpy()
        node_points = group[["node_id", "x", "y", "z"]].to_numpy()
        fixed_positions = fixed_points[:, 1:]

        for node in node_points:
            node_id = int(node[0])
            pos = node[1:]
            diff = fixed_positions - pos
            dists = np.linalg.norm(diff, axis=1)
            min_idx = int(np.argmin(dists))
            nearest_id = int(fixed_points[min_idx, 0])
            axis_dist = np.abs(diff[min_idx])
            record: dict[str, float | int] = {
                "node_id": node_id,
                "nearest_fixed_node_id": nearest_id,
                "fixed_fx": float(axis_dist[0]),
                "fixed_fy": float(axis_dist[1]),
                "fixed_fz": float(axis_dist[2]),
            }
            if has_time:
                record[time_column] = time_value
            rows.append(record)

    result = pd.DataFrame(rows)
    if has_time:
        column_order = [time_column, "node_id", "nearest_fixed_node_id", "fixed_fx", "fixed_fy", "fixed_fz"]
        return result[column_order].sort_values([time_column, "node_id"]).reset_index(drop=True)
    return result[["node_id", "nearest_fixed_node_id", "fixed_fx", "fixed_fy", "fixed_fz"]].sort_values("node_id").reset_index(drop=True)
