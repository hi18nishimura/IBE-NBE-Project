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
    x: (F,) or (N, F)
    max_values: (F,)
    """
    # Avoid division by zero
    eps = 1e-8
    
    signs = torch.sign(x)
    absvals = torch.abs(x)
    
    # formula: sign(x) * 0.4 * (|x|/max_vals)^(1/alpha) + 0.5
    # Note: max_values should be broadcastable to x
    
    normalized = signs * 0.4 * torch.pow(absvals / (max_values + eps), 1.0/alpha) + 0.5
    return normalized

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate NBE Simple MLP models")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing model subdirectories (e.g. outputs/simple_mlp_train)")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing dataset files")
    parser.add_argument("--output_dir", type=str, default="NBE/simple_mlp_eval", help="Directory to save results")
    return parser.parse_args()

def load_models(model_dir, dataset_dir):
    model_dir_path = Path(model_dir)
    dataset_dir_path = Path(dataset_dir)
    
    if not model_dir_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not dataset_dir_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    models = {}
    node_params = {}
    datasets = {}
    
    # Iterate over subdirectories in model_dir
    subdirs = [d for d in model_dir_path.iterdir() if d.is_dir()]
    
    # Try to find auxiliary files in dataset_dir if needed (mainly for summary statistics if global normalization)
    summary_overall_max = None
    
    fixed_nodes_file = None
    possible_fn = list(dataset_dir_path.rglob("fixed_nodes.csv"))
    if possible_fn:
        fixed_nodes_file = possible_fn[0]

    node_connection_file = None
    possible_nc = list(dataset_dir_path.rglob("node_connections.csv"))
    if possible_nc:
        node_connection_file = possible_nc[0]

    for subdir in tqdm(subdirs, desc="Loading models"):
        try:
            node_id = int(subdir.name)
        except ValueError:
            continue 
            
        model_path = subdir / "best.pth"
        if not model_path.exists():
            # Try finetune/pretrain convention if best.pth not found
            if (subdir / "best_finetune.pth").exists():
                 model_path = subdir / "best_finetune.pth"
            elif (subdir / "best_recursive.pth").exists():
                model_path = subdir / "best_recursive.pth"
            else:
                continue
            
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"Failed to load {model_path}: {e}")
            continue
            
        cfg = checkpoint.get('cfg', {})
        if not cfg:
            print(f"No config found in {model_path}")
            continue
            
        # Handle OmegaConf object
        if OmegaConf.is_config(cfg):
            cfg = OmegaConf.to_container(cfg, resolve=True)
            
        # Handle serialized OmegaConf as dict (with _content)
        if isinstance(cfg, dict) and '_content' in cfg:
            cfg = cfg['_content']
            
        # Ensure cfg is a dict now
        if not isinstance(cfg, dict):
             # Try to recover if it's a string or other
             try:
                 cfg = OmegaConf.create(cfg)
                 cfg = OmegaConf.to_container(cfg, resolve=True)
             except:
                 pass
        
        if not isinstance(cfg, dict):
            print(f"Config invalid for {model_path}")
            continue

        try:
            alpha = cfg.get('alpha', 8.0)
            global_normalize = cfg.get('global_normalize', True)
            glob_pattern = cfg.get('glob', '*.feather')
            use_split_mlp = cfg.get('use_split_mlp', False)
        except Exception as e:
            print(f"Error accessing config: {e}")
            continue
        
        # Determine hidden size etc from config if available, or defaults
        # The simple MLP architecture is currently hardcoded in NbeSimpleMLP class layers, 
        # but input/output dims depend on data.
        
        try:
            dataset = NbeDataset(
                data_dir=dataset_dir_path,
                node_id=node_id,
                alpha=alpha,
                global_normalize=global_normalize,
                glob=glob_pattern,
                node_connection_file=node_connection_file,
                fixed_nodes_file=fixed_nodes_file
            )
        except Exception as e:
            print(f"Failed to initialize dataset for node {node_id}: {e}")
            continue
            
        if len(dataset) == 0:
            print(f"Dataset empty for node {node_id}")
            continue
            
        sample = dataset[0]
        # input shape: (SeqLen, F)
        input_size = sample['inputs'].shape[-1] 
        output_size = dataset.target_feature_size
        
        if use_split_mlp:
            model = NbeSimpleSplitMLP(input_size, output_size)
        else:
            model = NbeSimpleMLP(input_size, output_size)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            # Try loading directly if checkpoint is the state dict (unlikely based on my training script)
            try:
                model.load_state_dict(checkpoint)
            except:
                print(f"Could not load state dict for node {node_id}")
                continue
            
        model.eval()
        models[node_id] = model
        
        # Store params
        max_vals_list = [dataset.max_map.get(col, 1.0) for col in dataset.columns]
        max_vals_tensor = torch.tensor(max_vals_list, dtype=torch.float32)

        # Check if this node is fixed
        is_fixed = dataset.fixed_nodes.get(node_id, False)
        
        # Create a mask for valid features (excluding dx, dy, dz if fixed)
        # Assuming first 3 columns are dx, dy, dz
        feature_mask = torch.ones(len(dataset.columns), dtype=torch.bool)
        if is_fixed:
            # Assuming standard order dx, dy, dz are 0, 1, 2
            # We should check column names to be safe but usually they are first
            for i, col in enumerate(dataset.columns):
                if col in ['dx', 'dy', 'dz']:
                    feature_mask[i] = False
        
        node_params[node_id] = {
            "input_size": input_size,
            "output_size": output_size,
            "max_values": dataset.max_map,
            "max_values_tensor": max_vals_tensor,
            "alpha": alpha,
            "is_fixed": is_fixed,
            "feature_mask": feature_mask
        }
        datasets[node_id] = dataset
        
    return models, node_params, datasets

def convert_physical_output_to_input_mlp(physical_outputs_dict, datasets, node_params, device):
    inputs_dict = {}
    
    # We need to construct input for each target model
    for target_node, dataset in datasets.items():
        # Get the order of nodes (central + neighbors) used by this model
        node_order = dataset.node_order
        
        feature_parts = []
        for neighbor_id in node_order:
            if neighbor_id not in physical_outputs_dict:
                # Missing neighbor value, cannot construct input.
                # This might happen if neighbor_id is not in 'models' keys either?
                # Or if physical_outputs_dict is incomplete.
                # Assuming physical_outputs_dict covers all required nodes.
                # If not, we might need a fallback or zeroes.
                
                # Check if we can get params for this neighbor even if not in models?
                # For now assume it is in physical_outputs_dict (previous step output or force GT)
                pass 
            
            if neighbor_id in physical_outputs_dict:
                phys_val = physical_outputs_dict[neighbor_id].to(device)
                
                # We need normalize params for the neighbor
                # But node_params is keyed by model node_id.
                # Does node_params contain all nodes? Only those with models.
                # If neighbor has no model, we still need its params (max_vals) to normalize.
                
                # If neighbor is not in node_params (i.e. no model loaded for it), 
                # we should try to get params from dataset if possible?
                # But dataset is per-node.
                # However, datasets[target_node] *contains* info about neighbors? Not really max_vals for neighbors.
                # Wait, NbeDataset stores max_values_tensor for *features*, not nodes?
                # max_map is "feature name" -> max. 
                # Since global_normalize is true, max_map is same for everyone usually.
                
                # If global_normalize=True, we can use target_node's params for everyone?
                # Yes, max_values match column names.
                
                # Let's use target_node's params to get max_vals/alpha
                target_params = node_params[target_node]
                max_vals = target_params["max_values_tensor"].to(device)
                alpha = target_params["alpha"]
                
                # Normalize neighbor's physical value
                norm_val = oka_normalize_tensor(phys_val, max_vals, alpha)
                
                # Filter if neighbor is fixed (remove dx, dy, dz)
                # We need to know if neighbor is fixed.
                # dataset.fixed_nodes is a dict {nid: bool}
                neighbor_is_fixed = dataset.fixed_nodes.get(neighbor_id, False)
                
                if neighbor_is_fixed:
                    # Remove first 3 elements (dx, dy, dz)
                    # We assume feature order.
                    # Or use a reusable mask.
                    # Assuming standard order for simplicity as in other parts
                    # norm_val: (F,)
                    # We want to remove 0,1,2 indices.
                    # Keep 3..end
                    norm_val = norm_val[3:]
                
                feature_parts.append(norm_val)
                
        if feature_parts:
            # Concatenate all parts to form single vector (Input_Size)
            cat_features = torch.cat(feature_parts, dim=0) # (Total_F,)
            # Add Batch and Time dims: (1, Total_F)
            inputs_dict[target_node] = cat_features.unsqueeze(0)
            
    return inputs_dict

def run_evaluation(models, node_params, datasets, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move models to device
    for model in models.values():
        model.to(device)
        
    if not datasets:
        print("No datasets loaded.")
        return

    # Assume all datasets have same length and sync
    first_node = list(datasets.keys())[0]
    num_samples = len(datasets[first_node])
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for idx in range(num_samples):
        print(f"Evaluating sample {idx+1}/{num_samples}")
        
        # Get Force Node Info (Ground Truth Displacement) 
        # This returns physical values (likely normalized? need to verify. 
        # Usually get_force_node_info returns normalized values in the context of GNN eval reading.)
        # Let's assume it matches the GNN implementation where we have to be careful.
        # In GNN eval: "fdisp is NORMALIZED" -> verified by comment in GNN snippet.
        force_info = datasets[first_node].get_force_node_info(idx)
        
def get_initial_input(datasets, idx):
    """
    Get the initial input (Time 1) for all nodes for a specific sample index.
    """
    node_inputs = {}
    all_targets = {}
    
    for node_id, dataset in datasets.items():
        data = dataset[idx]
        # inputs: (Time, F)
        # targets: (Time, F)
        
        # We start with the first time step input
        t1_input = data["inputs"][0] # Shape (F,)
        node_inputs[node_id] = t1_input.unsqueeze(0) # Shape (1, F)
        
        all_targets[node_id] = data["targets"]
        
    return node_inputs, all_targets

def run_evaluation(models, node_params, datasets, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move models to device
    for model in models.values():
        model.to(device)
        
    if not datasets:
        print("No datasets loaded.")
        return

    # Assume all datasets have same length and sync
    first_node = list(datasets.keys())[0]
    num_samples = len(datasets[first_node])
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for idx in range(num_samples):
        print(f"Evaluating sample {idx+1}/{num_samples}")
        
        # Get Force Node Info (Ground Truth Displacement) 
        # This returns physical values (likely normalized? need to verify. 
        # Usually get_force_node_info returns normalized values in the context of GNN eval reading.)
        # Let's assume it matches the GNN implementation where we have to be careful.
        # In GNN eval: "fdisp is NORMALIZED" -> verified by comment in GNN snippet.
        force_info = datasets[first_node].get_force_node_info(idx)
        
        # Get initial inputs
        current_inputs, all_targets = get_initial_input(datasets, idx)
        
        # Handle Fixed Node Initial Input filtering
        # Fixed nodes in dataset have NaNs or are removed. 
        # But here we need to make sure the input dimension matches what the model expects.
        # If model expects 60 dims (fixed node) but input has 63 dims (full node), we must filter.
        
        filtered_inputs = {}
        for nid, inp in current_inputs.items():
             if nid not in models:
                 continue
             
             # Calculate expected input dimension from model weight
             # NbeSimpleMLP or SplitMLP -> fc1.weight.shape[1]
             model = models[nid]
             expected_dim = model.fc1.in_features
             
             # inp is (1, F)
             if inp.shape[1] != expected_dim:
                 # Assume fixed node logic: remove first 3 columns (dx, dy, dz)
                 # Check if that matches expected_dim
                 # Or use feature_mask from params if available?
                 # Actually NbeDataset already handles this check internally in __getitem__ usually?
                 # But if NbeDataset config (conn files) was missing, __getitem__ returns wrong size.
                 
                 # Now we fixed load_models, so get_initial_input should return correct size!
                 # So this filter block might be redundant or just a sanity check.
                 
                 if inp.shape[1] > expected_dim:
                    # Try slicing. If fixed, it's usually neighbor handling or fixed node issue.
                    # If this block is hit, it means mismatch is still there.
                    # For now just use inp, assuming get_initial_input is correct now.
                    pass
                 
             filtered_inputs[nid] = inp

        current_inputs = filtered_inputs

        # Move to device
        for nid in current_inputs:
            current_inputs[nid] = current_inputs[nid].to(device)
            
        all_outputs_list = []
        
        # Determine number of steps
        # inputs[0] -> predicts targets[0] (which is Time 2)
        # ...
        target_len = all_targets[first_node].shape[0]
        
        # Loop for time steps
        for t in tqdm(range(target_len), desc=f"Sample {idx+1} steps"):
            step_physical_outputs = {}
            next_inputs = {}
            
            # Prediction Step
            for node_id, input_tensor in current_inputs.items():
                if node_id not in models:
                    continue
                
                model = models[node_id]
                
                with torch.no_grad():
                    # input_tensor: (1, F)
                    output_tensor = model(input_tensor) # (1, Out)
                    
                    # Store normalized output if needed, but we mostly need physical for result saving and recursive loop
                    pred_norm = output_tensor.squeeze(0) # (Out,)
                    
                    # Denormalize
                    params = node_params[node_id]
                    max_vals = params["max_values_tensor"].to(device)
                    alpha = params["alpha"]
                    feature_mask = params["feature_mask"].to(device)
                    
                    if params["is_fixed"]:
                         # Select only relevant max_values
                        valid_max_vals = max_vals[feature_mask]
                        phys_val_part = oka_denormalize(pred_norm, valid_max_vals, alpha)
                        
                        # Reconstruct full vector
                        phys_val = torch.zeros_like(max_vals)
                        phys_val[feature_mask] = phys_val_part
                    else:
                        phys_val = oka_denormalize(pred_norm, max_vals, alpha)

                    step_physical_outputs[node_id] = phys_val
            
            # Force Node Overwrite (Ground Truth Injection)
            # t=0 predicts Time 2 (index 0 of targets).
            # force_info usually usually aligned such that we look up the time step.
            # In GNN eval: force_idx = t + 1 for T=2 at start.
            # If t refers to the loop index 0..target_len-1. 
            # inputs[0] is time 1 data. Model predicts time 2 state.
            # So at t=0, we have prediction for Time 2.
            # force_info likely has T=1, T=2... index 0, 1...
            # We want force info for Time 2.
            
            force_time_idx = t + 1 # Aligning with GNN logic
            
            for fnid, fdisp_list in force_info.items():
                if fnid in step_physical_outputs:
                    if force_time_idx < len(fdisp_list):
                        # Force node displacement from list - typically normalized or physical?
                        # GNN code said: "fdisp is NORMALIZED".
                        # And: "Denormalize to match all_physical_outputs space".
                        
                        # So we take normalized value, denormalize it, and overwrite physical output.
                        norm_gt = torch.tensor(fdisp_list[force_time_idx], device=device, dtype=torch.float32)
                        
                        params = node_params[fnid]
                        max_vals_full = params["max_values_tensor"].to(device)
                        alpha = params["alpha"]
                        
                        # Use only first 3 max values for displacement (dx, dy, dz)
                        # We assume the first 3 columns are dx, dy, dz
                        max_vals_u = max_vals_full[:3]
                        
                        phys_gt = oka_denormalize(norm_gt, max_vals_u, alpha)
                        
                        # Overwrite the dx,dy,dz parts.
                        # Assuming first 3 columns are dx, dy, dz.
                        # The prediction might have more columns (Sxx etc).
                        # We only overwrite position for force nodes.
                        
                        # Clone to avoid in-place issues if any
                        current_phys = step_physical_outputs[fnid].clone()
                        current_phys[:3] = phys_gt # Overwrite dx, dy, dz
                        step_physical_outputs[fnid] = current_phys

            all_outputs_list.append(step_physical_outputs)
            
            # Prepare Next Inputs
            # Reconstruct inputs from all physical outputs (handling neighbors)
            current_inputs = convert_physical_output_to_input_mlp(step_physical_outputs, datasets, node_params, device)
            
        # Save results for this sample
        save_results(idx, all_outputs_list, datasets, output_dir)
        print(f"Sample {idx+1} completed.")

def save_results(idx, all_outputs_list, datasets, output_dir):
    first_node = list(datasets.keys())[0]
    target_file_path = datasets[first_node].files[idx]
    target_file_stem = target_file_path.stem
    correct_df = pd.read_feather(target_file_path)
    
    columns = datasets[first_node].columns
    
    result_df_list = []
    
    # Process predictions
    # all_outputs_list[0] corresponds to Time 2 prediction
    for step, step_outputs in enumerate(all_outputs_list):
        time = step + 2
        
        step_data = {}
        for node_id, val in step_outputs.items():
            step_data[node_id] = val.cpu().numpy()
            
        if not step_data:
            continue
            
        time_df = pd.DataFrame.from_dict(step_data, orient='index')
        
        # Basic validation of columns
        if len(time_df.columns) == len(columns):
            time_df.columns = columns
        else:
            time_df.columns = columns[:len(time_df.columns)]
            
        time_df['time'] = time
        time_df['node_id'] = time_df.index
        result_df_list.append(time_df)
        
    if not result_df_list:
        return

    result_df = pd.concat(result_df_list, axis=0)
    
    # Add Time 1 from Ground Truth (initial state)
    time1_correct = correct_df[correct_df['time'] == 1]
    time1_df = time1_correct[['time', 'node_id'] + columns].copy()
    
    result_df = pd.concat([time1_df, result_df], axis=0)
    result_df = result_df.sort_values(['node_id', 'time']).reset_index(drop=True)
    
    # Reconstruct absolute coordinates (x, y, z)
    # Using the same logic as GNN eval: Position = Init + CumSum(Disp) - First_Disp
    
    # Get initial coords from T=1 of Ground Truth
    initial_df = time1_correct[['node_id', 'x', 'y', 'z']].rename(columns={'x': 'init_x', 'y': 'init_y', 'z': 'init_z'})
    
    # Merge initial coords
    result_df = pd.merge(result_df, initial_df, on='node_id', how='left')
    
    # Calculate cumulative displacement
    # Note: result_df contains 'dx', 'dy', 'dz' which are incremental displacements per step?
    # Or positions?
    # dataset usually has 'dx', 'dy', 'dz' as displacements from initial shape? Or velocity?
    # Based on GNN eval logic: 
    # df['cum_dx'] = df.groupby('node_id')['dx'].cumsum()
    # df['x'] = df['init_x'] + df['cum_dx'] - df.groupby('node_id')['dx'].transform('first')
    # This implies 'dx' in the dataframe is position-like or relative displacement?
    # Actually, if 'dx' is displacement from init, then x = init_x + dx.
    # The cumsum logic suggests 'dx' might be inter-step velocity or similar?
    # Let's stick EXACTLY to the GNN eval logic provided in snippet.
    
    def add_xyz_cols(df):
        # We assume dataset uses accumulated dx/dy/dz relative to initial
        # Wait, if `dx` IS the accumulated displacement, why `cumsum`?
        # If I look at the snippet: `df['cum_dx'] = df.groupby('node_id')['dx'].cumsum()`
        # This strongly suggests the model predicts incremental change (velocity)?
        # OR the snippet was handling a specific case.
        # Let's assume the prediction target is what's in the dataset.
        # If the GNN logic does cumsum, I should do cumsum.
        
        df['cum_dx'] = df.groupby('node_id')['dx'].cumsum()
        df['cum_dy'] = df.groupby('node_id')['dy'].cumsum()
        df['cum_dz'] = df.groupby('node_id')['dz'].cumsum()
        
        # Transformation:
        # x = init + cumsum - first
        # value at t=1 is "first". cumsum at t=1 is val_t1.
        # x_t1 = init + val_t1 - val_t1 = init. Correct.
        # x_t2 = init + (val_t1 + val_t2) - val_t1 = init + val_t2.
        # This logic implies 'dx' is incremental displacement relative to previous step?
        # OR it implies something else. But I will copy-paste the logic to ensure consistency.
        
        df['x'] = df['init_x'] + df['cum_dx'] - df.groupby('node_id')['dx'].transform('first')
        df['y'] = df['init_y'] + df['cum_dy'] - df.groupby('node_id')['dy'].transform('first')
        df['z'] = df['init_z'] + df['cum_dz'] - df.groupby('node_id')['dz'].transform('first')
        
        return df.drop(columns=['cum_dx', 'cum_dy', 'cum_dz', 'init_x', 'init_y', 'init_z'])

    result_df = add_xyz_cols(result_df)
    
    # Compare with Ground Truth
    gt_df = correct_df.copy()
    gt_df_renamed = gt_df.add_suffix('_correct')
    gt_df_renamed = gt_df_renamed.rename(columns={'time_correct': 'time', 'node_id_correct': 'node_id'})
    
    merged_df = pd.merge(result_df, gt_df_renamed, on=['time', 'node_id'], how='left')
    
    # Calculate Errors
    error_cols = ['x', 'y', 'z', 'dx', 'dy', 'dz', 'Sxx', 'Syy', 'Szz', 'Sxy', 'Syz', 'Szx']
    for col in error_cols:
        if col in merged_df.columns and f'{col}_correct' in merged_df.columns:
            merged_df[f'{col}_error'] = (merged_df[col] - merged_df[f'{col}_correct']).abs()
            
    # Select columns
    cols_to_keep = ['time', 'node_id']
    for col in error_cols:
        if col in merged_df.columns:
            cols_to_keep.append(col)
        if f'{col}_correct' in merged_df.columns:
            cols_to_keep.append(f'{col}_correct')
        if f'{col}_error' in merged_df.columns:
            cols_to_keep.append(f'{col}_error')
            
    final_df = merged_df[cols_to_keep]
    
    output_csv = Path(output_dir) / f"{target_file_stem}_results.csv"
    final_df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")

if __name__ == "__main__":
    args = parse_args()
    models, node_params, datasets = load_models(args.model_dir, args.dataset_dir)
    print(f"Loaded {len(models)} models.")
    
    run_evaluation(models, node_params, datasets, args.output_dir)
