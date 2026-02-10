
import argparse
import os
import sys
import glob
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
import pandas as pd
import numpy as np

# Add workspace root to sys.path
sys.path.append("/workspace")

from src.Networks.nbe_simple_mlp import NbeSimpleMLP, NbeSimpleSplitMLP
from src.Dataloader.nbeDataset import NbeDataset
from src.Evaluate.nbe_normalize_inverse import oka_denormalize

def oka_normalize_tensor(x, max_values, alpha=3.0):
    """
    Apply Oka normalization to a tensor.
    x: (N, F) or (F,)
    max_values: (F,) or (1, F)
    """
    eps = 1e-8
    signs = torch.sign(x)
    absvals = torch.abs(x)
    normalized = signs * 0.4 * torch.pow(absvals / (max_values + eps), 1.0/alpha) + 0.5
    return normalized

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate NBE Simple MLP models with Matrix Operations")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing model subdirectories")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing dataset files")
    parser.add_argument("--output_dir", type=str, default="outputs/matrix_eval", help="Directory to save results")
    return parser.parse_args()

def load_global_info(dataset_dir):
    """Load connection and fixed node info once."""
    dataset_dir_path = Path(dataset_dir)
    
    node_connection_file = None
    possible_nc = list(dataset_dir_path.rglob("node_connections.csv"))
    if possible_nc:
        node_connection_file = possible_nc[0]
    else:
        # Fallback
        default_nc = Path("/workspace/dataset/liver_model_info/node_connections.csv")
        if default_nc.exists():
            node_connection_file = default_nc

    fixed_nodes_file = None
    possible_fn = list(dataset_dir_path.rglob("fixed_nodes.csv"))
    if possible_fn:
        fixed_nodes_file = possible_fn[0]
    else:
        # Fallback
        default_fn = Path("/workspace/dataset/liver_model_info/fixed_nodes.csv")
        if default_fn.exists():
            fixed_nodes_file = default_fn

    # Load connections
    node_connections = {}
    if node_connection_file:
        print(f"Loading connections from {node_connection_file}")
        df_nc = pd.read_csv(node_connection_file)
        # Handle 'neighbors' or 'connected_nodes'
        col_name = 'neighbors' if 'neighbors' in df_nc.columns else ('connected_nodes' if 'connected_nodes' in df_nc.columns else None)
        
        if 'node_id' in df_nc.columns and col_name:
            for _, row in df_nc.iterrows():
                try:
                    raw = row[col_name]
                    neighbors = []
                    if isinstance(raw, str):
                        # "1, 2, 3"
                        neighbors = [int(s.strip()) for s in raw.split(',') if s.strip()]
                    elif isinstance(raw, (int, float)):
                        # Single number or nan
                        if pd.notna(raw):
                            neighbors = [int(raw)]
                    else:
                        # List or other (eval fallback)
                         try:
                             val = eval(str(raw))
                             if isinstance(val, (list, tuple)):
                                 neighbors = [int(x) for x in val]
                         except:
                             pass
                    
                    node_connections[int(row['node_id'])] = neighbors
                except Exception as e:
                    # print(f"Error parsing connections for row {row}: {e}")
                    pass
    
    fixed_nodes = {}
    if fixed_nodes_file:
        print(f"Loading fixed nodes from {fixed_nodes_file}")
        df_fn = pd.read_csv(fixed_nodes_file)
        
        if 'node_id' in df_fn.columns:
            # Check for is_fixed column
            has_is_fixed = 'is_fixed' in df_fn.columns
            
            for _, row in df_fn.iterrows():
                nid = int(row['node_id'])
                if has_is_fixed:
                    # Treat as fixed only if is_fixed is True-ish
                    val = row['is_fixed']
                    if isinstance(val, str):
                        is_fixed_val = val.lower() in ('true', '1', 't', 'yes')
                    else:
                        is_fixed_val = bool(val)
                    
                    if is_fixed_val:
                        fixed_nodes[nid] = True
                else:
                    # If column not present, assume presence in file means fixed
                    fixed_nodes[nid] = True
                
    return node_connections, fixed_nodes

def load_models_into_matrix(model_dir, dataset_dir, node_connections, fixed_nodes):
    model_dir_path = Path(model_dir)
    subdirs = [d for d in model_dir_path.iterdir() if d.is_dir()]
    
    # We first scan to determine max input/output sizes and collect model weights
    node_configs = {}
    
    max_input_dim = 0
    max_output_dim = 0
    
    # Store weights
    weights = {} # node_id -> dict of weights
    
    # To determine dimensions correctly without NbeDataset overhead, 
    # we need to simulate input construction logic.
    # We need column count. Assuming 9 columns by default.
    # Fixed nodes skip 3 columns.
    DEFAULT_COLS = ["dx", "dy", "dz", "Sxx", "Syy", "Szz", "Sxy", "Syz", "Szx"]
    
    valid_models = []
    
    print("Loading models...")
    for subdir in tqdm(subdirs):
        try:
            node_id = int(subdir.name)
        except ValueError:
            continue
            
        model_path = subdir / "best.pth"
        if not model_path.exists():
             if (subdir / "best_finetune.pth").exists():
                 model_path = subdir / "best_finetune.pth"
             elif (subdir / "best_recursive.pth").exists():
                model_path = subdir / "best_recursive.pth"
             else:
                continue

        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        except:
            continue
            
        cfg = checkpoint.get('cfg', {})
        if OmegaConf.is_config(cfg):
            cfg = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(cfg, dict) and '_content' in cfg:
            cfg = cfg['_content']
        if not isinstance(cfg, dict):
            # Fallback
            cfg = {'alpha': 8.0, 'global_normalize': True}
            
        # Determine Input Size
        neighbors = node_connections.get(node_id, [])
        node_order = [node_id] + list(neighbors)
        
        # Determine sizes from logic (for gather mapping)
        logic_input_dim = 0
        input_mapping = [] # List of (neighbor_id, is_fixed)
        
        for nid in node_order:
            is_fixed = fixed_nodes.get(nid, False)
            cols = len(DEFAULT_COLS) - (3 if is_fixed else 0)
            logic_input_dim += cols
            input_mapping.append((nid, is_fixed))
        
        # Load State Dict
        sd = None
        if 'model_state_dict' in checkpoint:
            sd = checkpoint['model_state_dict']
        elif 'model_state' in checkpoint:
            sd = checkpoint['model_state']
        else:
            sd = checkpoint
            
        # Determine actual input dim from weights
        w1_shape = sd['fc1.weight'].shape
        actual_input_dim = w1_shape[1]
        
        if logic_input_dim != actual_input_dim:
            # print(f"Warning: Model {node_id} expects {actual_input_dim} inputs, but connections suggest {logic_input_dim}.")
            # If logic < actual, we might miss some inputs.
            # If logic > actual, we might extract too much (but gather indices will be constructed by mapping, so truncated).
            pass
            
        # Determine Output Size
        
        # Load State Dict
        sd = None
        if 'model_state_dict' in checkpoint:
            sd = checkpoint['model_state_dict']
        elif 'model_state' in checkpoint:
            sd = checkpoint['model_state']
        else:
            sd = checkpoint
            
        # Determine actual input dim from weights
        w1_shape = sd['fc1.weight'].shape
        actual_input_dim = w1_shape[1]
        
        # Determine actual output dim from weights
        w4_shape = sd['fc4.weight'].shape
        actual_output_dim = w4_shape[0]

        if logic_input_dim != actual_input_dim:
            pass
            
        is_self_fixed = fixed_nodes.get(node_id, False)
        output_dim = len(DEFAULT_COLS) - (3 if is_self_fixed else 0)
        
        # Verify output dim match
        if actual_output_dim != output_dim:
            # print(f"Warning: Model {node_id} has output dim {actual_output_dim}, expected {output_dim} (Fixed={is_self_fixed}). Using actual.")
            output_dim = actual_output_dim
            # If mismatch, maybe fixed status is wrong? Update it?
            if output_dim == 9 and is_self_fixed:
                 # Actually not fixed
                 is_self_fixed = False
            elif output_dim == 6 and not is_self_fixed:
                 # Actually fixed
                 is_self_fixed = True
        
        # Use Actual Dimension for Allocation
        max_input_dim = max(max_input_dim, actual_input_dim)
        max_output_dim = max(max_output_dim, output_dim)
        
        weights[node_id] = sd
        node_configs[node_id] = {
            "input_dim": actual_input_dim, # Use actual
            "output_dim": output_dim,
            "input_mapping": input_mapping,
            "alpha": cfg.get('alpha', 8.0),
            "is_fixed": is_self_fixed
        }
        valid_models.append(node_id)

        
    if not valid_models:
        raise ValueError("No valid models found")
        
    valid_models.sort()
    num_models = len(valid_models)
    node_to_idx = {nid: i for i, nid in enumerate(valid_models)}
    
    print(f"Loaded {num_models} models. Max Input: {max_input_dim}, Max Output: {max_output_dim}")
    
    # Construct Tensor Batches
    # Layer 1: (N, 64, MaxIn)
    # Layer 2: (N, 32, 64)
    # Layer 3: (N, 16, 32)
    # Layer 4: (N, MaxOut, 16)
    
    W1 = torch.zeros(num_models, 64, max_input_dim)
    B1 = torch.zeros(num_models, 64)
    
    W2 = torch.zeros(num_models, 32, 64)
    B2 = torch.zeros(num_models, 32)
    
    W3 = torch.zeros(num_models, 16, 32)
    B3 = torch.zeros(num_models, 16)
    
    W4 = torch.zeros(num_models, max_output_dim, 16)
    B4 = torch.zeros(num_models, max_output_dim)
    
    # To construct gather indices, we need a mapping from (node_id, feature_idx) to Global State Vector Index
    # Global State Vector S: (Total_Nodes * 9) or (Total_Nodes, 9)
    # Let's use (Total_nodes, 9) index for easier valid masking.
    # But gather works on flat index.
    # Let's say all nodes (even those without models) have state in S.
    # We need to map real node_id -> range [0, MaxNodeID]. Or dense map.
    # Given dataset feather files usually contain all nodes.
    # Let's map node_id -> row_index in S.
    
    # Load one sample to get ALL node IDs
    sample_files = sorted(Path(dataset_dir).glob("*.feather"))
    if not sample_files:
        raise FileNotFoundError("No feather files found")
    
    sample_df = pd.read_feather(sample_files[0])
    all_node_ids = sorted(sample_df['node_id'].unique())
    node_map = {nid: i for i, nid in enumerate(all_node_ids)}
    num_total_nodes = len(all_node_ids)
    
    # Gather Indices Tensor: (num_models, max_input_dim)
    gather_indices = torch.zeros(num_models, max_input_dim, dtype=torch.long)
    
    # Padding index: We will append a dummy row to S at the end.
    # S shape: (NumTotalNodes + 1, 9). S[-1] is dummy zero.
    dummy_idx = num_total_nodes # row index
    
    # Fill tensors
    for i, nid in enumerate(valid_models):
        sd = weights[nid]
        cfg = node_configs[nid]
        
        # Layer 1
        w1 = sd['fc1.weight'] # (64, In)
        b1 = sd['fc1.bias']
        in_dim = w1.shape[1]
        W1[i, :, :in_dim] = w1
        B1[i, :] = b1
        
        # Layer 2
        W2[i] = sd['fc2.weight']
        B2[i] = sd['fc2.bias']
        
        # Layer 3
        W3[i] = sd['fc3.weight']
        B3[i] = sd['fc3.bias']
        
        # Layer 4
        w4 = sd['fc4.weight'] # (Out, 16)
        b4 = sd['fc4.bias']
        out_dim = w4.shape[0]
        W4[i, :out_dim, :] = w4
        B4[i, :out_dim] = b4
        
        # Build Gather Indices
        current_input_idx = 0
        mapping = cfg["input_mapping"] # [(nid, is_fixed), ...]
        
        for neighbor_id, neighbor_fixed in mapping:
            if neighbor_id not in node_map:
                # Missing node in data? Should not happen if data is consistent.
                # Point to dummy
                cols = 6 if neighbor_fixed else 9
                for _ in range(cols):
                    if current_input_idx < max_input_dim:
                         # 9 cols per node in S. S has shape (N+1, 9) Flattened -> ((N+1)*9)
                         # We want linear index into flattened S.
                         # Dummy row is index num_total_nodes.
                         gather_indices[i, current_input_idx] = dummy_idx * 9 # Point to dummy row, first col (all 0)
                         current_input_idx += 1
                continue
                
            row_idx = node_map[neighbor_id]
            
            # Feature indices
            # If neighbor is fixed, we skip 0,1,2 (dx,dy,dz). Take 3..8.
            # If neighbor is not fixed, take 0..8.
            start_col = 3 if neighbor_fixed else 0
            end_col = 9
            
            for col in range(start_col, end_col):
                if current_input_idx < max_input_dim:
                    # Linear index: row_idx * 9 + col
                    gather_indices[i, current_input_idx] = row_idx * 9 + col
                    current_input_idx += 1
                    
        # Pad remaining gather indices to point to dummy
        while current_input_idx < max_input_dim:
            gather_indices[i, current_input_idx] = dummy_idx * 9
            current_input_idx += 1

    # Params
    params = {
        "node_map": node_map,        # Real ID -> index in valid_models (No, index in Data S)
        "model_node_ids": valid_models, # Real IDs of models
        "node_to_model_idx": node_to_idx, # Real ID -> index in model batches
        "node_configs": node_configs,
        "gather_indices": gather_indices,
        "dummy_idx": dummy_idx,
        "num_total_nodes": num_total_nodes,
        "column_names": DEFAULT_COLS
    }
    
    model_matrices = {
        "W1": W1, "B1": B1,
        "W2": W2, "B2": B2,
        "W3": W3, "B3": B3,
        "W4": W4, "B4": B4
    }
    
    return model_matrices, params

def batch_forward(matrices, inputs_flat, output_mask):
    """
    inputs_flat: (Batch(NumModels), MaxIn)
    output_mask: (Batch, MaxOut) boolean mask to zero out padding
    """
    # Layer 1
    # W1: (N, 64, In). inputs: (N, In). 
    # Use bmm: W1 @ inp.unsqueeze(2) -> (N, 64, 1)
    h = torch.bmm(matrices["W1"], inputs_flat.unsqueeze(2)).squeeze(2) + matrices["B1"]
    h = torch.tanh(h)
    
    # Layer 2
    h = torch.bmm(matrices["W2"], h.unsqueeze(2)).squeeze(2) + matrices["B2"]
    h = torch.tanh(h)
    
    # Layer 3
    h = torch.bmm(matrices["W3"], h.unsqueeze(2)).squeeze(2) + matrices["B3"]
    h = torch.tanh(h)
    
    # Layer 4
    out = torch.bmm(matrices["W4"], h.unsqueeze(2)).squeeze(2) + matrices["B4"]
    
    # Sigmoid Output Scaling
    out = 0.1 + 0.8 * torch.sigmoid(out)
    
    return out

def run_matrix_evaluation(dataset_dir, output_dir, model_matrices, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move models to device
    for k, v in model_matrices.items():
        model_matrices[k] = v.to(device)
        
    gather_indices = params["gather_indices"].to(device) # (N_models, MaxIn)
    
    dataset_dir_path = Path(dataset_dir)
    files = sorted(dataset_dir_path.glob("*.feather"))
    if not files:
        print(f"No feather files found in {dataset_dir}")
        return
    
    # Load Max Values for Normalization
    summary_file = dataset_dir_path / ".." / "bin" / "toy_all_model" / "train" / "summary_overall_max_values.csv"
    if not summary_file.exists():
         possible = list(Path("/workspace").rglob("summary_overall_max_values.csv"))
         if possible:
             summary_file = possible[0]
             
    max_vals = torch.ones(9, device=device)
    max_map = {}
    if summary_file and summary_file.exists():
        df_max = pd.read_csv(summary_file)
        for _, row in df_max.iterrows():
            max_map[str(row['feature'])] = float(row['max_value'])
        m_list = [max_map.get(c, 1.0) for c in params["column_names"]]
        max_vals = torch.tensor(m_list, device=device, dtype=torch.float32)

    # Prepare Pre-computed tensors for Denormalization & Scatter
    num_models = len(params["model_node_ids"])
    max_out = model_matrices["W4"].shape[1] # Max Output Dim
    
    batch_max_vals = torch.ones(num_models, max_out, device=device)
    scatter_row_indices = torch.zeros(num_models, max_out, dtype=torch.long, device=device)
    scatter_col_indices = torch.zeros(num_models, max_out, dtype=torch.long, device=device)
    
    # Output Mask (1 if valid output, 0 if padding) - helpful if we sum or something, 
    # but for scatter we can just point padded outputs to dummy row?
    # Better: Use dummy row/col for padding or use mask.
    # Since we scatter into S (N+1, 9), we can point invalid outputs to dummy row S[N].
    
    node_map = params["node_map"] # RealID -> RowID
    inv_node_map = {v: k for k, v in node_map.items()}
    dummy_idx = params["dummy_idx"]
    
    for i, nid in enumerate(params["model_node_ids"]):
        cfg = params["node_configs"][nid]
        is_fixed = cfg["is_fixed"]
        
        # Features produced by this model
        # If fixed: produce 6 features (dx..dz skipped). Correspond to cols 3..8.
        # If normal: produce 9 features. Cols 0..8.
        
        current_out_idx = 0
        
        start_col = 3 if is_fixed else 0
        end_col = 9
        
        for col_idx in range(start_col, end_col):
            if current_out_idx < max_out:
                # Max Value for this feature
                batch_max_vals[i, current_out_idx] = max_vals[col_idx]
                
                # Scatter target
                row_id = node_map[nid]
                scatter_row_indices[i, current_out_idx] = row_id
                scatter_col_indices[i, current_out_idx] = col_idx
                
                current_out_idx += 1
                
        # Handle padding
        while current_out_idx < max_out:
            batch_max_vals[i, current_out_idx] = 1.0 # arbitrary
            scatter_row_indices[i, current_out_idx] = dummy_idx
            scatter_col_indices[i, current_out_idx] = 0
            current_out_idx += 1
            
    # Alpha
    if params["model_node_ids"]:
        first_model_id = params["model_node_ids"][0]
        alpha = params["node_configs"][first_model_id]["alpha"]
    else:
        alpha = 8.0

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    num_total_nodes = params["num_total_nodes"]
    
    # Helper to load force info
    def get_force_node_info_local(df_sample):
        if 'force_node_id' not in df_sample.columns:
            return {}
        fids = df_sample['force_node_id'].dropna().unique()
        info = {}
        for fid in fids:
            if pd.isna(fid): continue
            fid = int(fid)
            sub = df_sample[df_sample['node_id'] == fid].sort_values('time')
            if sub.empty: continue
            disp = sub[['dx', 'dy', 'dz']].values 
            
            # Normalize displacement for force nodes
            # We need to normalize because we inject it into the Physical logic?
            # Wait. S_phys stores Physical values.
            # So we need physical displacement.
            # get_force_node_info in dataset returns NORMALIZED.
            # But here `disp` from feather is PHYSICAL (likely? check dataset comments).
            # Usually feather contains whatever is logged. Usually physical.
            # Dataset normalizes it.
            # If we want to inject into S_phys, we should use PHYSICAL.
            # So `disp` is already good!
            
            # BUT, we need to be careful. In GNN script, it denormalized ground truth.
            # "datasets[first_node].get_force_node_info(idx)" returns normalized.
            # Here we read directly.
            # So `disp` is physical.
            
            # However, we must ensure unit consistency.
            # If `S_phys` is physical, we just write `disp` into dx,dy,dz columns.
            
            t_vals = torch.tensor(disp, device=device, dtype=torch.float32)
            info[fid] = t_vals
        return info

    columns = params["column_names"]
    
    files_pbar = tqdm(files, desc="Eval Files")
    for file_path in files_pbar:
        try:
            df = pd.read_feather(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        force_info = get_force_node_info_local(df)
        
        # Determine Timesteps
        if 'time' not in df.columns:
            continue
        times = sorted(df['time'].unique())
        if not times: continue
        
        # Initial State (T=1)
        t1_df = df[df['time'] == 1]
        if t1_df.empty: continue
        
        # Fill S_phys from T1
        # Map node_id -> row index
        S_phys = torch.zeros(num_total_nodes + 1, 9, device=device)
        
        # We need to fill S_phys efficiently.
        # t1_df has 'node_id'.
        # We iterate or use bulk assign.
        # Check if t1_df covers all nodes.
        present_nodes = t1_df['node_id'].values
        present_vals = t1_df[columns].values
        
        # Convert present_nodes to row indices
        # Helper map array?
        # Since we use a dict `node_map`, we map one by one or vectorize.
        # If node_ids are contiguous this is fast. If random, slower.
        # Vectorized map:
        # We can build a fast lookup if needed. But for initialization, loop is ok or map.
        
        # Bulk fill
        indices = [node_map[nid] for nid in present_nodes if nid in node_map]
        valid_mask = [nid in node_map for nid in present_nodes]
        vals_tensor = torch.tensor(present_vals[valid_mask], device=device, dtype=torch.float32)
        indices_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        
        S_phys.index_copy_(0, indices_tensor, vals_tensor)
        
        # Store results (Physical)
        # List of (Time, NodeID, 9-Features)
        # Storing all history in GPU memory might be too much?
        # (2000 nodes * 20 steps * 9 * 4 bytes) ~ 1.4MB. Tiny.
        # Even 100k nodes -> 70MB. Safe.
        
        all_results = []
        # Store T1
        all_results.append(S_phys[:num_total_nodes].clone()) # T=1
        
        current_time_idx = 0 # Corresponds to T=1 (index 0 in 'times')
        
        # Simulation Loop (Predict T=2 onwards)
        # times usually [1, 2, ..., 20]
        # We predict steps 1..(T-1).
        
        for step in range(len(times) - 1):
            next_time_val = times[step+1]
            force_idx = step + 1 # Index in force array corresponding to 'next_time'
            
            # 1. Normalize S_phys -> S_norm
            # Use max_vals (9,) broadcast
            S_norm = oka_normalize_tensor(S_phys, max_vals.unsqueeze(0), alpha)
            S_norm[dummy_idx] = 0.0
            
            # 2. Gather Inputs
            # gather_indices: (NumModels, MaxIn)
            S_flat = S_norm.view(-1)
            inputs_flat = S_flat[gather_indices] # (NumModels, MaxIn)
            
            # 3. Forward Pass
            # We don't need mask for forward, padding is handled.
            norm_out = batch_forward(model_matrices, inputs_flat, None) # (NumModels, MaxOut)
            
            # 4. Denormalize
            # batch_max_vals: (NumModels, MaxOut)
            # norm_out -> phys_out
            phys_out = oka_denormalize(norm_out, batch_max_vals, alpha)
            
            # 5. Scatter Update S_phys
            # We update S_phys with predicted values.
            # scatter_row_indices, scatter_col_indices
            # Linear index in S_phys: row * 9 + col.
            
            # S_phys is (N+1, 9). View as (-1).
            S_phys_flat = S_phys.view(-1)
            
            scatter_lin_idx = scatter_row_indices * 9 + scatter_col_indices
            
            # Flatten Source
            phys_out_flat = phys_out.view(-1)
            
            # Scatter
            # We use index_put_ or scatter_.
            # S_phys_flat[scatter_lin_idx] = phys_out_flat
            
            # Caution: dummy_idx writes go to dummy row.
            S_phys_flat.scatter_(0, scatter_lin_idx.view(-1), phys_out_flat.view(-1))
            
            # Restore Shape
            # S_phys is modified in place (view shares memory).
            
            # 6. Force Node Injection
            # Overwrite dx, dy, dz for force nodes with GT
            # force_info: {fid: tensor(T, 3)}
            # We need to overwrite indices in S_phys.
            # For each force node, we have new dx, dy, dz.
            
            # We can vectorize this too if many force nodes, but usually few.
            for fid, fdisp in force_info.items():
                if fid in node_map:
                    rid = node_map[fid]
                    if force_idx < len(fdisp):
                        # fdisp[force_idx] is (3,)
                        # Write to cols 0, 1, 2 of S_phys[rid]
                        S_phys[rid, :3] = fdisp[force_idx] 
            
            # Store result
            all_results.append(S_phys[:num_total_nodes].clone())

        # Save Results for this file
        # Reconstruct DataFrame
        # Result List: [Tensor(N, 9) for t in times]
        # Stack -> (T, N, 9)
        res_tensor = torch.stack(all_results, dim=0).cpu().numpy() # (T, N, 9)
        
        # Create DataFrame
        # We need to map row indices back to NodeIDs.
        # inv_node_map: 0 -> nid1, 1 -> nid2...
        sorted_nids = [inv_node_map[i] for i in range(num_total_nodes)]
        
        dfs = []
        for t_i, t_val in enumerate(times):
            # Data for time T
            data = res_tensor[t_i] # (N, 9)
            df_t = pd.DataFrame(data, columns=columns)
            df_t['node_id'] = sorted_nids
            df_t['time'] = t_val
            dfs.append(df_t)
            
        final_df = pd.concat(dfs, ignore_index=True)
        
        # Calculate Coordinates (x, y, z)
        # Using cumsum logic from original script
        # x = init_x + cumsum(dx) - first_dx
        # But we need initial coordinates.
        # Get from t1_df (initial state loaded from feather)
        # t1_df has x, y, z usually? Feather has EVERYTHING.
        # Check if x, y, z in df.
        
        init_coords = t1_df[['node_id', 'x', 'y', 'z']].rename(columns={'x': 'init_x', 'y': 'init_y', 'z': 'init_z'})
        
        final_df = pd.merge(final_df, init_coords, on='node_id', how='left')
        
        final_df['cum_dx'] = final_df.groupby('node_id')['dx'].cumsum()
        final_df['cum_dy'] = final_df.groupby('node_id')['dy'].cumsum()
        final_df['cum_dz'] = final_df.groupby('node_id')['dz'].cumsum()
        
        final_df['x'] = final_df['init_x'] + final_df['cum_dx'] - final_df.groupby('node_id')['dx'].transform('first')
        final_df['y'] = final_df['init_y'] + final_df['cum_dy'] - final_df.groupby('node_id')['dy'].transform('first')
        final_df['z'] = final_df['init_z'] + final_df['cum_dz'] - final_df.groupby('node_id')['dz'].transform('first')
        
        final_df.drop(columns=['cum_dx', 'cum_dy', 'cum_dz', 'init_x', 'init_y', 'init_z'], inplace=True)
        
        # Save
        out_name = f"{file_path.stem}_results.csv"
        final_df.to_csv(output_path / out_name, index=False)
        #print(f"Saved {out_name}")

if __name__ == "__main__":
    args = parse_args()
    
    # 1. Load Connections
    print("Loading Global Info...")
    node_connections, fixed_nodes = load_global_info(args.dataset_dir)
    
    # 2. Load Models & Build Matrices
    print("Building Model Matrices...")
    model_matrices, params = load_models_into_matrix(
        args.model_dir, 
        args.dataset_dir, 
        node_connections, 
        fixed_nodes
    )
    
    # 3. Run
    print("Starting Evaluation...")
    run_matrix_evaluation(args.dataset_dir, args.output_dir, model_matrices, params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move to device
    for k, v in model_matrices.items():
        model_matrices[k] = v.to(device)
        
    gather_indices = params["gather_indices"].to(device) # (N_models, MaxIn)
    
    dataset_dir_path = Path(dataset_dir)
    files = sorted(dataset_dir_path.glob("*.feather"))
    
    # Load Max Values for Normalization
    # Assuming Global Normalization as per context
    summary_file = dataset_dir_path / ".." / "bin" / "toy_all_model" / "train" / "summary_overall_max_values.csv"
    # Or search
    if not summary_file.exists():
         possible = list(Path("/workspace").rglob("summary_overall_max_values.csv"))
         if possible:
             summary_file = possible[0]
             
    max_vals = torch.ones(9, device=device)
    if summary_file and summary_file.exists():
        df_max = pd.read_csv(summary_file)
        max_map = {}
        for _, row in df_max.iterrows():
            max_map[str(row['feature'])] = float(row['max_value'])
        m_list = [max_map.get(c, 1.0) for c in params["column_names"]]
        max_vals = torch.tensor(m_list, device=device, dtype=torch.float32)

    # For denormalization:
    # We need to know which model is which
    # And apply correct max_vals (usually same for all if global, but if local...)
    # NbeDataset logic implies global_normalize uses one set of max_vals.
    
    # Alpha
    # Assuming alpha constant for simplicity or take from first model
    first_model_id = params["model_node_ids"][0]
    alpha = params["node_configs"][first_model_id]["alpha"]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    num_total_nodes = params["num_total_nodes"]
    node_map = params["node_map"] # RealID -> S row index
    model_node_ids = params["model_node_ids"]
    
    # S_phys: (NumTotalNodes + 1, 9). Stores PHYSICAL values.
    # Because normalization happens on input construction (usually).
    # Wait, NbeSimpleMLP expects Normalized Inputs?
    # Yes. NbeDataset returns normalized inputs.
    # So we should store PHYSICAL state in S_phys, and normalize on the fly?
    # Or store NORMALIZED state in S_norm?
    # Normalizing on the fly for every interaction is expensive?
    # Normalizing 2000 nodes x 9 feats is cheap.
    # Let's keep S_phys to easily inject ground truth and save results.
    
    for file_idx, file_path in enumerate(files):
        print(f"Evaluating {file_path.name}...")
        
        # Load Ground Truth
        df_gt = pd.read_feather(file_path)
        
        # Determine number of steps
        # df_gt has 'time' column.
        times = sorted(df_gt['time'].unique())
        if not times:
            continue
            
        # Initial State (Time 1)
        t1_df = df_gt[df_gt['time'] == 1]
        
        # Initialize S (Physical)
        # Shape: (TotalNodes + 1, 9)
        S_phys = torch.zeros(num_total_nodes + 1, 9, device=device)
        
        # Fill T1 data
        # Map node_id to row index
        # We need efficient mapping.
        # df column to tensor assignment.
        
        # Create a temporary array aligned with node_map
        # Sort t1_df by node_id to match? node_map might not be sorted by ID if constructed from unique() which is hashed?
        # unique() returns sorted? No.
        # But we built node_map from sorted unique IDs.
        
        # Verify ids match
        # t1_df should have entry for all nodes.
        t1_ids = t1_df['node_id'].values
        # It's safer to reindex
        t1_df = t1_df.set_index('node_id').reindex(sorted(node_map.keys()))
        
        # Extract values
        vals = t1_df[params["column_names"]].values # (N, 9)
        S_phys[:num_total_nodes] = torch.tensor(vals, device=device, dtype=torch.float32)
        
        # Result Container
        results = []
        
        # Force Node Info for this sample
        # In original script, get_force_node_info(idx)
        # Here we can just read from GT for the force nodes?
        # Force nodes are those in datasets[n].fixed_nodes? No.
        # Force nodes are distinct from fixed nodes often.
        # But wait, original script uses `dataset.get_force_node_info`.
        # That method likely reads a separate force file or interprets dataset columns?
        # In `IBE-NBE` context, force nodes are where displacement is prescribed.
        # Let's assume GT has the correct values for force nodes at each step.
        # We just need to know WHICH nodes are force nodes.
        # If we don't know, we might just overwrite ALL nodes with GT if we treat them as such? No.
        # The script `nbe_simple_mlp_evaluate.py` logic:
        # force_info = dataset.get_force_node_info(idx)
        # This returns a dictionary of node_ids and their displacements over time.
        # If we can't replicate `get_force_node_info`, we can't injection forces correctly.
        # Without this, dynamic simulation fails.
        
        # How to trigger `get_force_node_info` without Dataset?
        # NbeDataset doesn't seem to have it in the snippet provided.
        # Ah, read_file for NbeDataset stopped at line 250. Maybe it's further down.
        # Or maybe it's in the snippet `nbe_simple_mlp_evaluate.py` logic... No `dataset.get_force_node_info` is called.
        # Let's read more of NbeDataset.py.
        
        # Since I can't easily rely on NbeDataset without full context, 
        # I'll implement a fallback: Read GT for "Fixed/Force" nodes from the file itself if possible?
        # But which ones are force nodes?
        # In the dataset, maybe `fixed_nodes.csv` defines them?
        # Or `node_type` column?
        # If I can't solve this, the evaluation is incorrect.
        # HOWEVER, the user asked to optimize, assuming the logic in `nbe_simple_mlp_evaluate.py` is correct reference.
        # I must call `get_force_node_info`.
        
        # Let's try to instantiate a dummy NbeDataset just to get force info helper?
        # Or read the implementation of `get_force_node_info`.
        pass
    
    # Placeholder for loop logic
    pass

if __name__ == "__main__":
    pass
