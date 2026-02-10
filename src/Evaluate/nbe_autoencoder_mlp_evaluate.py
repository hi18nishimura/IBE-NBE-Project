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

from src.Networks.nbe_simple_mlp import NbeAutoEncoderMLP
from src.Dataloader.nbeAEDataset import NbeDataset
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
    parser = argparse.ArgumentParser(description="Evaluate NBE AutoEncoder MLP models")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing model subdirectories (e.g. outputs/autoencoder_mlp_train)")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing dataset files")
    parser.add_argument("--output_dir", type=str, default="outputs/autoencoder_mlp_eval", help="Directory to save results")
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
        except Exception as e:
            print(f"Error accessing config: {e}")
            continue
        
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
        
        model = NbeAutoEncoderMLP(input_size, output_size)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
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
        
        feature_mask = torch.ones(len(dataset.columns), dtype=torch.bool)
        if is_fixed:
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
                # Fallback or missing logic if necessary
                pass 
            
            if neighbor_id in physical_outputs_dict:
                phys_val = physical_outputs_dict[neighbor_id].to(device)
                
                target_params = node_params[target_node]
                max_vals = target_params["max_values_tensor"].to(device)
                alpha = target_params["alpha"]
                
                # Normalize neighbor's physical value
                norm_val = oka_normalize_tensor(phys_val, max_vals, alpha)
                
                neighbor_is_fixed = dataset.fixed_nodes.get(neighbor_id, False)
                if neighbor_is_fixed:
                    # Remove first 3 elements (dx, dy, dz)
                    norm_val = norm_val[3:]
                
                feature_parts.append(norm_val)
                
        if feature_parts:
            # Concatenate all parts to form single vector (Input_Size)
            cat_features = torch.cat(feature_parts, dim=0) # (Total_F,)
            # Add Batch and Time dims: (1, Total_F)
            inputs_dict[target_node] = cat_features.unsqueeze(0)
            
    return inputs_dict

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
        
        force_info = datasets[first_node].get_force_node_info(idx)
        
        # Get initial inputs
        current_inputs, all_targets = get_initial_input(datasets, idx)
        
        # Move to device
        for nid in current_inputs:
            current_inputs[nid] = current_inputs[nid].to(device)
            
        all_outputs_list = []
        
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
                    # NbeAutoEncoderMLP returns (output, rec_x)
                    output_tensor, _ = model(input_tensor) # (1, Out)
                    
                    # Store normalized output
                    pred_norm = output_tensor.squeeze(0) # (Out,)
                    
                    # Denormalize
                    params = node_params[node_id]
                    max_vals = params["max_values_tensor"].to(device)
                    alpha = params["alpha"]
                    feature_mask = params["feature_mask"].to(device)
                    
                    if params["is_fixed"]:
                        valid_max_vals = max_vals[feature_mask]
                        phys_val_part = oka_denormalize(pred_norm, valid_max_vals, alpha)
                        
                        phys_val = torch.zeros_like(max_vals)
                        phys_val[feature_mask] = phys_val_part
                    else:
                        phys_val = oka_denormalize(pred_norm, max_vals, alpha)

                    step_physical_outputs[node_id] = phys_val
            
            # Force Node Overwrite (Ground Truth Injection)
            force_time_idx = t + 1 
            
            for fnid, fdisp_list in force_info.items():
                if fnid in step_physical_outputs:
                    if force_time_idx < len(fdisp_list):
                        norm_gt = torch.tensor(fdisp_list[force_time_idx], device=device, dtype=torch.float32)
                        
                        params = node_params[fnid]
                        max_vals_full = params["max_values_tensor"].to(device)
                        alpha = params["alpha"]
                        
                        max_vals_u = max_vals_full[:3]
                        
                        phys_gt = oka_denormalize(norm_gt, max_vals_u, alpha)
                        
                        current_phys = step_physical_outputs[fnid].clone()
                        current_phys[:3] = phys_gt # Overwrite dx, dy, dz
                        step_physical_outputs[fnid] = current_phys

            all_outputs_list.append(step_physical_outputs)
            
            # Prepare Next Inputs
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
    # Get initial coords from T=1 of Ground Truth
    initial_df = time1_correct[['node_id', 'x', 'y', 'z']].rename(columns={'x': 'init_x', 'y': 'init_y', 'z': 'init_z'})
    
    # Merge initial coords
    result_df = pd.merge(result_df, initial_df, on='node_id', how='left')
    
    def add_xyz_cols(df):
        df['cum_dx'] = df.groupby('node_id')['dx'].cumsum()
        df['cum_dy'] = df.groupby('node_id')['dy'].cumsum()
        df['cum_dz'] = df.groupby('node_id')['dz'].cumsum()
        
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
