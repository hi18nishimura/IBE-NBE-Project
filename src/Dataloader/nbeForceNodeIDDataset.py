from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
import torch
from tqdm import tqdm

# Import NbeDataset from the same package
from .nbeDataset import NbeDataset

class NbeForceNodeIDDataset(NbeDataset):
    """
    Dataset class that extends NbeDataset to filter files based on force_node_id.

    It only includes feather files where the 'force_node_id' column contains
    any of the values specified in target_force_node_ids.
    """

    def __init__(
        self,
        data_dir: str | Path,
        node_id: int,
        target_force_node_ids: List[int],
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
        # Initialize parent class
        # Force preload=False initially to prevent loading all files before filtering
        super().__init__(
            data_dir=data_dir,
            node_id=node_id,
            columns=columns,
            extra_columns=extra_columns,
            preload=False,
            glob=glob,
            alpha=alpha,
            summary_overall_max=summary_overall_max,
            node_connection_file=node_connection_file,
            fixed_nodes_file=fixed_nodes_file,
            global_normalize=global_normalize
        )

        self.target_force_node_ids = set(target_force_node_ids)
        self.preload_requested = preload

        # Filter files based on force_node_id
        self._filter_files_by_force_node_id()

        # If user requested preload, perform it now on the filtered files
        if self.preload_requested:
            self.preload = True
            self._do_preload()

    def _filter_files_by_force_node_id(self):
        """Check the contents of current self.files and keep only those matching force_node_id."""
        filtered_files = []
        
        # Read only force_node_id column to check each file
        for fp in tqdm(self.files, desc="Filtering files by force_node_id"):
            try:
                # Use pandas read_feather with columns argument for efficiency
                df_check = pd.read_feather(fp, columns=["force_node_id"])
                
                # Get unique force_node_ids in the file
                file_force_ids = set(df_check["force_node_id"].unique())
                
                # If there is any intersection with target_force_node_ids, keep the file
                if not file_force_ids.isdisjoint(self.target_force_node_ids):
                    filtered_files.append(fp)
            except Exception:
                # Skip if column doesn't exist or read fails
                continue
        
        self.files = sorted(filtered_files)
        # print(f"Filtered files: {len(self.files)} files match force_node_id {self.target_force_node_ids}")

    def _do_preload(self):
        """Execute preload on the filtered file list."""
        if not self.files:
            return

        processed_list: List[Dict[str, torch.Tensor]] = []
        
        for fp in tqdm(self.files, desc=f"preload node {self.node_id} (filtered)"):
            try:
                df = pd.read_feather(fp)
                node_tables = self._extract_node_tables(df)
                processed = self._transform_input_output(node_tables)
                processed_list.append(processed)
            except Exception:
                # Append placeholder on error to maintain indexing
                processed_list.append({"inputs": None, "targets": None})
        
        self._data_cache = processed_list
    
    def get_physical_values(self, idx: int) -> pd.DataFrame:
        """
        Get the raw physical values (dx, dy, dz, Sxx, Syy, Szz, Sxy, Syz, Szx)
        from a specific file without normalization.
        
        Args:
            idx: Index of the file to read.
            
        Returns:
            DataFrame containing physical columns and time/node_id info.
        """
        if idx < 0 or idx >= len(self.files):
            raise IndexError("Index out of bound")
            
        fp = self.files[idx]
        target_columns = ["time", "node_id", "dx", "dy", "dz", "Sxx", "Syy", "Szz", "Sxy", "Syz", "Szx"]
        
        # Read the file
        df = pd.read_feather(fp)
        
        # Filter for the target columns that exist in the dataframe
        available_cols = [c for c in target_columns if c in df.columns]
        df_subset = df[available_cols].copy()
        
        # Filter for the relevant nodes (self.node_order) just like in _extract_node_tables
        # This keeps only the central node and its neighbors
        if "node_id" in df_subset.columns:
            df_subset = df_subset[df_subset['node_id'].isin(self.node_order)]
            
        return df_subset.sort_values(["time", "node_id"]).reset_index(drop=True)
    
    def get_target_node_physical_values(self, idx: int) -> pd.DataFrame:
        """
        Get the raw physical values for ONLY the target node (self.node_id).
        
        Args:
            idx: Index of the file to read.
            
        Returns:
            DataFrame containing physical columns for the target node.
        """
        df_all = self.get_physical_values(idx)
        if "node_id" in df_all.columns:
            return df_all[df_all["node_id"] == self.node_id].copy()
        return pd.DataFrame() # Return empty if node_id missing (should not happen)
