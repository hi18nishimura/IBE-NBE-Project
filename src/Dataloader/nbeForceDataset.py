from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
import sys
import os
from tqdm import tqdm

# Adjust path to import NbeDataset if necessary, assuming it's in the same package
from .nbeDataset import NbeDataset, oka_normalize_array, oka_normalize_dataframe_fast, DEFAULT_COLUMNS

class NbeForceDataset(NbeDataset):
    """
    Extension of NbeDataset that also returns information about the force node.
    
    Returns extra keys in __getitem__:
      - "force": Displacement (dx, dy, dz) of the force node at the input timesteps.
               Shape: (19, 3) (normalized)
      - "force_node_disp": Distance vector from the central node to the force node.
                         Shape: (3,) (normalized: x_nor, y_nor, z_nor from node_displacement_features.csv)
    """
    
    _disp_df_cache = None # Class-level cache for displacement dataframe

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
        node_displacement_file: Optional[str | Path] = None
    ) -> None:
        
        # Load node displacement features BEFORE calling super().__init__ because super().__init__ might trigger preload
        if node_displacement_file is None:
            node_displacement_file = "/workspace/dataset/liver_model_info/node_displacement_features.csv"
        
        self.node_displacement_file = Path(node_displacement_file)
        if not self.node_displacement_file.exists():
            raise FileNotFoundError(f"Node displacement file not found: {self.node_displacement_file}")
            
        # Use cached dataframe if available to avoid repeated IO
        if NbeForceDataset._disp_df_cache is None:
            print(f"Loading node displacement features from {self.node_displacement_file}...")
            NbeForceDataset._disp_df_cache = pd.read_csv(self.node_displacement_file)
        
        self.disp_df = NbeForceDataset._disp_df_cache
        
        # Create a lookup dictionary for faster access: (node_id, target_node_id) -> [x_nor, y_nor, z_nor]
        # Filtering for the current node_id to save memory
        self.disp_lookup = {}
        # We only need rows where node_id matches the dataset's central node_id
        # However, node_id is passed to __init__.
        # Let's filter now.
        relevant_disp = self.disp_df[self.disp_df['node_id'] == int(node_id)]
        for _, row in relevant_disp.iterrows():
            tid = int(row['target_node_id'])
            # Store extracted normalized features
            self.disp_lookup[tid] = np.array([row['x_nor'], row['y_nor'], row['z_nor']], dtype=np.float32)

        # Initialize parent with preload=False used temporarily to prevent premature data loading
        # Data loading requires self.force_pwidths which is initialized AFTER super().__init__
        super().__init__(
            data_dir=data_dir,
            node_id=node_id,
            columns=columns,
            extra_columns=extra_columns,
            preload=False, # Temporarily False
            glob=glob,
            alpha=alpha,
            summary_overall_max=summary_overall_max,
            node_connection_file=node_connection_file,
            fixed_nodes_file=fixed_nodes_file,
            global_normalize=global_normalize
        )
        
        # Precompute pwidths for force node displacement (dx, dy, dz)
        # Using the max map loaded by parent
        self.force_pwidths = np.array([self.max_map.get(c, 0.0) for c in ["dx", "dy", "dz"]])

        # Update input size to include force vector and force node displacement vector
        # Saves copy of original size for reshaping logic in _extract_node_tables
        self.original_input_feature_size = self.input_feature_size
        self.input_feature_size += 6 # 3 for force (dx, dy, dz), 3 for dist (x, y, z)
        
        # Now handle preload if requested
        self.preload = preload
        if self.preload:
            processed_list: List[Dict[str, torch.Tensor]] = []
            for fp in tqdm(self.files, desc=f"preload node {self.node_id}"):
                try:
                    df = pd.read_feather(fp)
                    node_tables = self._extract_node_tables(df)
                    processed = self._transform_input_output(node_tables)
                    processed_list.append(processed)
                except Exception as e:
                    # print(f"Error preloading {fp}: {e}")
                    # if a single file fails, append a placeholder to keep indexing
                    # stable and allow runtime errors to surface later in training.
                    processed_list.append({"inputs": None, "targets": None})
            self._data_cache = processed_list

    def _extract_node_tables(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Overrides parent method to extract main node tables AND force node information.
        
        Returns:
            (arr_reshaped, force_seq, force_node_disp_vec)
        """
        # --- 1. Identify Force Node and Extract its Data ---
        if 'force_node_id' in df.columns:
            fids = df['force_node_id'].unique()
            # Filter NaNs
            fids = fids[~np.isnan(fids)]
            if len(fids) > 0:
                fid = int(fids[0])
            else:
                fid = -1 # Should not happen in valid data
        else:
            fid = -1
            
        # Extract Force Node Displacement (Time Series)
        # We need dx, dy, dz for the force node at all timesteps.
        # df contains data for all nodes.
        if fid != -1:
            # Filter for force node
            force_rows = df[df['node_id'] == fid]
            
            # Ensure sorting by time - usually feather is sorted or we trust the order?
            # NbeDataset assumes implicit order or pre-sorted?
            # 'time' column exists. Let's sort to be safe, though might be slow.
            # Assuming the input df is the raw read from feather, which serves as a "block" of data.
            # However, df has mixed nodes. We need to pivot or just extract.
            # The parent implementation does not explicitly sort by time inside _extract_node_tables,
            # but relies on reshape.
            
            # Let's extract dx, dy, dz.
            # Note: We need to match the time steps of the main input.
            # The main input covers t=1..19 (19 steps).
            # The df usually contains t=1..20 or more.
            # We want t=1..19 for the "force" input (aligned with inputs).
            
            # Let's clean and sort just in case
            # But creating a copy of slice is needed.
            # Optimization: assumes df is grouped by time or node?
            # Usually these feather files are long format: time 1 (all nodes), time 2 (all nodes)...
            # Or Node 1 (all times), Node 2 (all times)...
            # Let's check 'nbeDataset.py' implementation idea.
            
            # For robustness, let's just get the values sorted by time.
            # Assuming 'time' column exists.
            force_vals = force_rows.sort_values('time')[['dx', 'dy', 'dz']].values
            
            # If force_vals is empty or missing steps?
            # We assume data quality is good.
            
            # Normalize force node displacement
            # Using oka_normalize_array with the precomputed pwidths
            force_vals_norm = oka_normalize_array(force_vals, self.force_pwidths, self.alpha)
        else:
            # Placeholder if no force node found
            # Assuming 20 timesteps based on parent logic (19 input, 1 target from 2..20)
            # Actually we don't know T exactly until we see the data, but usually it's fixed.
            # We will handle slicing in _transform
            force_vals_norm = np.zeros((20, 3), dtype=np.float32) # Enough buffer

        # --- 2. Get Distance Vector (Static) ---
        if fid != -1 and fid in self.disp_lookup:
            dist_vec = self.disp_lookup[fid]
        else:
            # Fallback (e.g. force node is self, or unknown)
            # If fid == self.node_id, distance is 0.
            # If unknown, 0?
            dist_vec = np.zeros(3, dtype=np.float32)

        # --- 3. Parent Logic for Main Features ---
        # We replicate the logic since we can't easily wrap the mid-processing of parent
        times = len(df['time'].unique())
        
        # Filter columns
        df_main = df[self.columns]
        
        # Filter rows (neighbors + self)
        df_main = df_main.loc[self.extract_dataframe_idx]
        
        # Normalize
        df_main = oka_normalize_dataframe_fast(df_main, self.pwidth_array_broadcasted, self.alpha)
        
        # Mask fixed nodes
        df_main.loc[self.extract_fixed_idx, ['dx', 'dy', 'dz']] = np.nan
        
        # Flatten
        data_arr = df_main.values
        data_arr = data_arr[~np.isnan(data_arr)]
        arr_reshaped = data_arr.reshape(times, self.original_input_feature_size)

        return (arr_reshaped, force_vals_norm, dist_vec)

    def _transform_input_output(self, node_tables: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Transforms the extracted tables into PyTorch tensors.
        """
        main_arr, force_seq, dist_vec = node_tables
        
        # Parent logic:
        # Inputs: timesteps 0..18 (representing t=1..19)
        # Targets: timesteps 1..19 (representing t=2..20)
        # Note: main_arr has 'times' rows. Usually 20.
        
        # Safety check on length
        T = main_arr.shape[0]
        # We assume T >= 20 based on parent hardcoded slicing [1:20] etc initially?
        # Parent uses:
        # inputs = node_tables[:19, :]
        # targets = node_tables[1:20, ...]
        # So it implies T=20.
        
        inputs = torch.tensor(main_arr[:19, :], dtype=torch.float32)
        targets = torch.tensor(main_arr[1:20, :self.node_feature_counts[self.node_id]], dtype=torch.float32)
        
        # Force Input
        # Align with inputs (t=1..19) -> indices 0..18
        # force_seq should be (T, 3).
        # If force_seq is shorter, we pad? Assuming it matches main_arr time dimension.
        if force_seq.shape[0] >= 19:
            force_tensor = torch.tensor(force_seq[:19, :], dtype=torch.float32)
        else:
            # Fallback padding
            pad_size = 19 - force_seq.shape[0]
            if pad_size > 0:
                pad = np.zeros((pad_size, 3), dtype=np.float32)
                f_data = np.concatenate([force_seq, pad], axis=0)
                force_tensor = torch.tensor(f_data, dtype=torch.float32)
            else:
                 force_tensor = torch.tensor(force_seq[:19, :], dtype=torch.float32)
        
        # Distance Vector
        # Shape (3,)
        force_node_disp_tensor = torch.tensor(dist_vec, dtype=torch.float32)
        
        # Concatenate to inputs: force (19,3) + force_node_disp (expanded to 19,3)
        force_node_disp_expanded = force_node_disp_tensor.unsqueeze(0).expand(inputs.shape[0], -1)
        
        inputs = torch.cat([inputs, force_tensor, force_node_disp_expanded], dim=1)

        return {
            "inputs": inputs,
            "targets": targets,
            "force": force_tensor,
            "force_node_disp": force_node_disp_tensor
        }

