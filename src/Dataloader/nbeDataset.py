from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


DEFAULT_COLUMNS = ["dx", "dy", "dz", "Sxx", "Syy", "Szz", "Sxy", "Syz", "Szx"]


def oka_normalize_series(series: pd.Series, pwidth: float, alpha: float) -> pd.Series:
    """Oka normalization for a pandas Series.

    p_out = sign(p_in) * (0.4 / pwidth^(1/alpha)) * |p_in|^(1/alpha) + 0.5
    """
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    vals = series.to_numpy(dtype=float)
    signs = np.sign(vals)
    absvals = np.abs(vals)
    if pwidth == 0 or np.isnan(pwidth):
        return pd.Series(np.full_like(vals, 0.5, dtype=float), index=series.index)
    factor = 0.4 / (pwidth ** (1.0 / alpha))
    with np.errstate(invalid='ignore'):
        normed = signs * factor * (absvals ** (1.0 / alpha)) + 0.5
    return pd.Series(normed, index=series.index)


def oka_normalize_array(vals: np.ndarray, pwidths: np.ndarray, alpha: float) -> np.ndarray:
    """Vectorized Oka normalization for numpy arrays.

    - `vals` may contain NaN.
    - `pwidths` is broadcastable to `vals` shape (1D matching length of vals).
    """
    vals = np.asarray(vals, dtype=float)
    pwidths = np.asarray(pwidths, dtype=float)
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    signs = np.sign(vals)
    absvals = np.abs(vals)
    mask_bad = (pwidths == 0) | np.isnan(pwidths)
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = 0.4 / (pwidths ** (1.0 / alpha))
    with np.errstate(invalid='ignore'):
        normed = signs * factor * (absvals ** (1.0 / alpha)) + 0.5
    if mask_bad.any():
        maskb = mask_bad
        if maskb.shape != normed.shape:
            maskb = np.broadcast_to(maskb, normed.shape)
        normed = np.where(maskb, 0.5, normed)
    return normed

def oka_normalize_dataframe_fast(df: pd.DataFrame, pwidths: np.ndarray, alpha: float) -> pd.DataFrame:
    """
    Oka normalizationをNumPyのベクトル演算でDataFrame全体に適用する。

    :param df: 処理対象のDataFrame
    :param pwidths: 各列に対応するpwidthパラメータ (形状: (1, N_cols) または (N_rows, N_cols))
    :param alpha: スケーリングパラメータ
    :return: 正規化されたDataFrame
    """
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    # 1. データとパラメータをNumPy配列に変換
    vals = df.values
    
    # 2. ゼロ除算やNaN除外の条件処理 (ブロードキャストを利用)
    # 処理対象となる列のインデックスを特定 (pwidth > 0 かつ 有限値)
    valid_cols_mask = (pwidths > 0) & np.isfinite(pwidths)
    
    # 全ての列が無効な場合は、0.5で埋めて返す
    if not np.any(valid_cols_mask):
        return pd.DataFrame(np.full_like(vals, 0.5, dtype=float), index=df.index, columns=df.columns)

    # 3. ゼロ除算を避けるため、NaN/infを無視する設定で演算
    with np.errstate(divide='ignore', invalid='ignore'):
        # 4. 要素ごとのサインと絶対値を取得
        signs = np.sign(vals)
        absvals = np.abs(vals)
        
        # 5. ファクターとべき乗の計算 (ブロードキャストが自動的に適用される)
        power_term = 1.0 / alpha
        
        # pwidthsが有効な列でのみ計算を行う
        # invalid_cols (pwidth<=0) の factor は NaN となり、その列の結果も NaN になるため、
        # 後で 0.5 でマスクする。
        
        # factor = 0.4 / (pwidth ** (1.0 / alpha))
        # (1, N_cols) or (N_rows, N_cols) の形状の pwidths が vals (N_rows, N_cols) にブロードキャストされる
        factor = np.full_like(pwidths, np.nan, dtype=float)
        factor[:, valid_cols_mask[0, :]] = 0.4 / (pwidths[:, valid_cols_mask[0, :]] ** power_term)
        
        # normed = signs * factor * (absvals ** (1.0 / alpha)) + 0.5
        normed = signs * factor * (absvals ** power_term) + 0.5

    # 6. pwidth <= 0 (または NaN) の列を 0.5 で置き換える (元の関数の動作を再現)
    # np.where(条件, 真の場合の値, 偽の場合の値)
    # invalid_cols_mask (pwidth <= 0) の場合は 0.5、それ以外は normed の値
    result_vals = np.where(~valid_cols_mask, 0.5, normed)

    return pd.DataFrame(result_vals, index=df.index, columns=df.columns)

class NbeDataset(Dataset):
    """PyTorch Dataset for nodal time-series stored as .feather files.

    Behavior (implemented according to the header comments):
    - `data_dir` contains many .feather files; each file is treated as one sample.
    - `node_id` selects the central node for which we construct inputs/targets.
    - Inputs: timesteps 1..19 of central node and its neighbors (concatenated per timestep).
    - Targets: timesteps 2..20 of the central node.
    - Fixed neighbor nodes (from `fixed_nodes.csv`) have displacement components
      (`dx`,`dy`,`dz`) removed before concatenation.
    - Values are Oka-normalized using `summary_overall_max_values.csv` (feature -> max_value).

    The class attempts to be flexible about feather layout: it supports long-form tables
    with explicit `time`/`node_id` columns or wide-form columns like `dx_t1`, `dx_t2`, ...
    """

    def __init__(
        self,
        data_dir: str | Path,
        node_id: int,
        columns: Optional[List[str]] = None,
        extra_columns: Optional[List[str]] = None,
        preload: bool = False,
        glob: str = "*.feather",
        alpha: float = 8.0,
        summary_overall_max: Optional[str | Path] = None,
        node_connection_file: Optional[str | Path] = None,
        fixed_nodes_file: Optional[str | Path] = None,
        global_normalize: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise NotADirectoryError(f"data_dir not found: {self.data_dir}")
        self.files = sorted(self.data_dir.glob(glob))
        if not self.files:
            raise FileNotFoundError(f"No files found in {self.data_dir} matching {glob}")
        self.node_id = int(node_id)
        self.columns = list(columns or DEFAULT_COLUMNS)
        if extra_columns:
            self.columns.extend(extra_columns)
        self.preload = bool(preload)
        self.alpha = float(alpha)
        self.global_normalize = bool(global_normalize)

        # locate node_connections.csv and fixed_nodes.csv
        if node_connection_file is None:
            found = list(self.data_dir.rglob("node_connections.csv"))
            if found:
                node_connection_file = found[0]
            else:
                # fallback to workspace-wide liver_model_info path
                default_nc = Path("/workspace/dataset/liver_model_info/node_connections.csv")
                node_connection_file = default_nc if default_nc.exists() else None
        if fixed_nodes_file is None:
            found = list(self.data_dir.rglob("fixed_nodes.csv"))
            if found:
                fixed_nodes_file = found[0]
            else:
                # fallback to workspace-wide liver_model_info path
                default_fn = Path("/workspace/dataset/liver_model_info/fixed_nodes.csv")
                fixed_nodes_file = default_fn if default_fn.exists() else None

        self.node_connections = self._load_node_connections(Path(node_connection_file)) if node_connection_file else {}
        self.fixed_nodes = self._load_fixed_nodes(Path(fixed_nodes_file)) if fixed_nodes_file else {}

        self.neighbors = self.node_connections.get(self.node_id, [])

        # save node ordering used for constructing inputs: central node first, then neighbors
        self.node_order: List[int] = [self.node_id] + list(self.neighbors)
        self.have_fixed_nodes =[nid for nid in self.node_order if self.fixed_nodes.get(nid, False)]

        # locate summary_overall_max_values.csv if not explicitly given
        if summary_overall_max is None:
            if self.global_normalize:
                summary_overall_max = "/workspace/dataset/bin/toy_all_model/train/summary_overall_max_values.csv"
            else:
                summary_overall_max = "/workspace/dataset/bin/toy_all_model/train/summary_per_node_max_values.csv"
        self.summary_overall_max = summary_overall_max
        self.max_map: Dict[str, float] = {}
        if self.global_normalize:
            if self.summary_overall_max and Path(self.summary_overall_max).exists():
                df_max = pd.read_csv(self.summary_overall_max)
                for _, row in df_max.iterrows():
                    self.max_map[str(row['feature'])] = float(row['max_value'])
                self.pwidth_array_broadcasted = np.array([self.max_map.get(col, 0.0)  for col in self.columns])[np.newaxis, :]
            else:
                raise(FileNotFoundError(f"summary_overall_max_values.csv not found: {self.summary_overall_max}"))
        else:
            if self.summary_overall_max and Path(self.summary_overall_max).exists():
                df_max = pd.read_csv(self.summary_overall_max)
                df_max = df_max[df_max['node_id'].isin(self.node_order)]
                for col in self.columns:
                    self.max_map[str(col)] = df_max[col].max()
                self.pwidth_array_broadcasted = np.array([self.max_map.get(col, 0.0)  for col in self.columns])[np.newaxis, :]
            else:
                raise(FileNotFoundError(f"summary_overall_max_values.csv not found: {self.summary_overall_max}"))
        
        # precompute per-node feature counts (accounting for fixed nodes skipping dx/dy/dz)
        self.node_feature_counts: Dict[int, int] = {}
        for nid in self.node_order:
            is_fixed = self.fixed_nodes.get(nid, False)
            count = 0
            for col in self.columns:
                if is_fixed and col in ("dx", "dy", "dz"):
                    continue
                count += 1
            self.node_feature_counts[nid] = count

        # precompute slices (start inclusive, end exclusive) within the per-timestep concatenated vector
        self.input_slices: Dict[int, tuple[int, int]] = {}
        offset = 0
        for nid in self.node_order:
            cnt = self.node_feature_counts.get(nid, 0)
            self.input_slices[nid] = (offset, offset + cnt)
            offset += cnt
        self.input_feature_size: int = offset

        # 抽出するインデックスの情報を取得する
        self.extract_dataframe_idx,self.extract_fixed_idx = self._build_extraction_plan(pd.read_feather(self.files[0]))
        #print(self.node_order)
        # preload option: if requested, process each file now into tensors and
        # store a list of processed dicts {'inputs','targets'} in self._data_cache.
        # This makes __getitem__ return cached tensors immediately (fast), at the
        # cost of increased memory usage during preload. Show progress with tqdm.
        self._data_cache: Optional[List[Dict[str, torch.Tensor]]] = None

        if self.preload:
            processed_list: List[Dict[str, torch.Tensor]] = []
            for fp in tqdm(self.files, desc=f"preload node {self.node_id}"):
                try:
                    df = pd.read_feather(fp)
                    node_tables = self._extract_node_tables(df)
                    processed = self._transform_input_output(node_tables)
                    processed_list.append(processed)
                except Exception:
                    # if a single file fails, append a placeholder to keep indexing
                    # stable and allow runtime errors to surface later in training.
                    processed_list.append({"inputs": None, "targets": None})
            self._data_cache = processed_list

        # NOTE: extraction from feather files is intentionally left to the
        # caller to implement. No automatic extraction plan is built here.

    def _load_node_connections(self, path: Path) -> Dict[int, List[int]]:
        if not path.exists():
            return {}
        df = pd.read_csv(path)
        result: Dict[int, List[int]] = {}
        for _, row in df.iterrows():
            nid = int(row['node_id'])
            neigh_raw = row.get('neighbors')
            if pd.isna(neigh_raw):
                result[nid] = []
            else:
                if isinstance(neigh_raw, str):
                    items = [s.strip() for s in neigh_raw.split(',') if s.strip()]
                    result[nid] = [int(x) for x in items]
                else:
                    try:
                        result[nid] = list(map(int, neigh_raw))
                    except Exception:
                        result[nid] = []
        return result

    def _load_fixed_nodes(self, path: Path) -> Dict[int, bool]:
        if not path.exists():
            return {}
        df = pd.read_csv(path)
        return {int(row['node_id']): bool(row['is_fixed']) for _, row in df.iterrows()}

    def _build_extraction_plan(self, df: pd.DataFrame) -> list[int]:
        return df.index[df['node_id'].isin(self.node_order)].tolist(),df.loc[df['node_id'].isin(self.have_fixed_nodes), ['dx', 'dy', 'dz']].index.tolist()

    def __len__(self) -> int:
        return len(self.files)

    def _read_file(self, idx: int) -> pd.DataFrame:
        # If preload stored processed tensors, _read_file should not be used
        # to fetch them; callers should check self._data_cache directly. Here
        # keep backward compatibility: if cache is list of DataFrames return it,
        # otherwise always read from disk.
        if self._data_cache is not None and len(self._data_cache) > 0:
            first = self._data_cache[0]
            if isinstance(first, pd.DataFrame):
                return self._data_cache[idx]
        return pd.read_feather(self.files[idx])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # If we preloaded processed tensors, return cached item immediately.
        if self._data_cache is not None and len(self._data_cache) > 0:
            return self._data_cache[idx]

        # otherwise, read raw dataframe and process on-demand
        df = self._read_file(idx)
        node_tables = self._extract_node_tables(df)
        processed = self._transform_input_output(node_tables)
        return processed

    def _transform_input_output(self, node_tables) -> Dict[str, torch.Tensor]:
        """Process node_tables into normalized input/target tensors.

        This implementation follows the approach in `nbe_dataset.py`:
        - build a long-form DataFrame with columns `time`, `node_id` and features
        - for each timestep (1..19) flatten node-order (central then neighbors)
          into a single feature vector per timestep, applying masks for fixed nodes
          and vectorized Oka normalization
        - targets are central node values at t+1 (same masked column ordering)
        """
        inputs = torch.tensor(node_tables[:19,:],dtype=torch.float32)
        targets = torch.tensor(node_tables[1:20, :self.node_feature_counts[self.node_id]],dtype=torch.float32)

        return {"inputs": inputs, "targets": targets}

    def _extract_node_tables(self, df: pd.DataFrame):
        """Placeholder extractor.

        The implementation that extracted rows/columns from feather files was
        removed per user request so that the user can provide their own
        extraction logic. Implement this method to return a mapping
        {node_id: DataFrame} where each DataFrame has rows indexed by time
        (1-based ints) and columns for the features in ``self.columns``.
        """
        times = len(df['time'].unique())
        df = df[self.columns]
        df = df.loc[self.extract_dataframe_idx]
        # 正規化
        df = oka_normalize_dataframe_fast(df, self.pwidth_array_broadcasted, self.alpha)
        # 固定節点のdx,dy,dzをNaNにする
        df.loc[self.extract_fixed_idx, ['dx', 'dy', 'dz']] = np.nan
        data_arr = df.values

        data_arr = data_arr[~np.isnan(data_arr)]
        arr_reshaped = data_arr.reshape(times, self.input_feature_size)
        
        return arr_reshaped
