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
from src.Dataloader.nbeForceDataset import NbeForceDataset
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
    parser = argparse.ArgumentParser(description="Evaluate NBE Force Simple MLP models")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing model subdirectories (e.g. outputs/force_simple_mlp_train)")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing dataset files")
    parser.add_argument("--output_dir", type=str, default="outputs/force_simple_mlp_eval", help="Directory to save results")
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
    
    # Try to find auxiliary files in dataset_dir if needed
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
        
        # Initialize NbeForceDataset
        try:
            dataset = NbeForceDataset(
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
            
        # Get input output size
        # NbeForceDataset handles +6 input size augmentation internally
        input_size = dataset.input_feature_size
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

def convert_physical_output_to_node_features(physical_outputs_dict, datasets, node_params, device):
    """
    Constructs the NODE FEATURE part of the input for each model from the physical output of the previous step.
    Does NOT include Force or Distance Vector features.
    """
    node_features_dict = {}
    
    for target_node, dataset in datasets.items():
        node_order = dataset.node_order
        
        feature_parts = []
        for neighbor_id in node_order:
            if neighbor_id in physical_outputs_dict:
                phys_val = physical_outputs_dict[neighbor_id].to(device)
                
                # Use target node's normalization parameters (assuming global_normalize=True)
                target_params = node_params[target_node]
                max_vals = target_params["max_values_tensor"].to(device)
                alpha = target_params["alpha"]
                
                # Normalize
                norm_val = oka_normalize_tensor(phys_val, max_vals, alpha)
                
                neighbor_is_fixed = dataset.fixed_nodes.get(neighbor_id, False)
                if neighbor_is_fixed:
                    # Remove dx, dy, dz (first 3)
                    norm_val = norm_val[3:]
                
                feature_parts.append(norm_val)
            else:
                 # Missing neighbor data
                 pass

        if feature_parts:
            cat_features = torch.cat(feature_parts, dim=0) # (Total_Node_F,)
            node_features_dict[target_node] = cat_features
            
    return node_features_dict

def get_initial_input_and_force(datasets, idx):
    """
    Get the initial input (Time 1) and force info.
    """
    node_inputs = {}
    all_targets = {}
    dist_vecs = {}
    
    first_node = list(datasets.keys())[0]
    # Assuming force seq is same for all nodes
    common_force_seq = datasets[first_node][idx]["force"] # (T, 3)
    
    for node_id, dataset in datasets.items():
        data = dataset[idx]
        node_inputs[node_id] = data["inputs"][0].unsqueeze(0) # Shape (1, F+6)
        all_targets[node_id] = data["targets"]
        dist_vecs[node_id] = data["force_node_disp"] # (3,)
        
    return node_inputs, all_targets, common_force_seq, dist_vecs

def run_evaluation(models, node_params, datasets, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    for model in models.values():
        model.to(device)
        
    if not datasets:
        print("No datasets loaded.")
        return

    first_node = list(datasets.keys())[0]
    num_samples = len(datasets[first_node])
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for idx in range(num_samples):
        print(f"Evaluating sample {idx+1}/{num_samples}")
        
        # Get ground truth force info for result comparison/injection
        force_info_gt = datasets[first_node].get_force_node_info(idx)
        
        # Get initial inputs and force data for recursion
        current_inputs, all_targets, whole_force_seq, dist_vecs = get_initial_input_and_force(datasets, idx)
        
        whole_force_seq = whole_force_seq.to(device)
        for nid in dist_vecs:
             dist_vecs[nid] = dist_vecs[nid].to(device)

        # Filter initial inputs if necessary (size check)
        filtered_inputs = {}
        for nid, inp in current_inputs.items():
             if nid not in models:
                 continue
             filtered_inputs[nid] = inp.to(device)
        current_inputs = filtered_inputs

        all_outputs_list = []
        target_len = all_targets[first_node].shape[0]
        
        for t in tqdm(range(target_len), desc=f"Sample {idx+1} steps"):
            step_physical_outputs = {}
            
            # Predict
            for node_id, input_tensor in current_inputs.items():
                if node_id not in models:
                    continue
                
                model = models[node_id]
                with torch.no_grad():
                    output_tensor = model(input_tensor) # (1, Out)
                    pred_norm = output_tensor.squeeze(0) # (Out,)
                    
                    params = node_params[node_id]
                    max_vals = params["max_values_tensor"].to(device)
                    alpha = params["alpha"]
                    feature_mask = params["feature_mask"].to(device)
                    
                    if params["is_fixed"]:
                        valid_max_vals = max_vals[feature_mask]
                        print(f"pred_norm: {pred_norm}")
                        print(f"valid_max_vals: {valid_max_vals}")
                        exit()
                        phys_val_part = oka_denormalize(pred_norm, valid_max_vals, alpha)
                        phys_val = torch.zeros_like(max_vals)
                        phys_val[feature_mask] = phys_val_part
                    else:
                        phys_val = oka_denormalize(pred_norm, max_vals, alpha)

                    step_physical_outputs[node_id] = phys_val
            
            # Injection of GT for Force Node
            force_time_idx = t + 1 # Time 2 prediction corresponds to index 0. Force info needed at T=2.
            for fnid, fdisp_list in force_info_gt.items():
                if fnid in step_physical_outputs:
                    if force_time_idx < len(fdisp_list):
                        norm_gt = torch.tensor(fdisp_list[force_time_idx], device=device, dtype=torch.float32)
                        params = node_params[fnid]
                        max_vals_u = params["max_values_tensor"].to(device)[:3]
                        alpha = params["alpha"]
                        phys_gt = oka_denormalize(norm_gt, max_vals_u, alpha)
                        
                        current_phys = step_physical_outputs[fnid].clone()
                        current_phys[:3] = phys_gt
                        step_physical_outputs[fnid] = current_phys

            all_outputs_list.append(step_physical_outputs)
            
            # Prepare Next Inputs
            next_node_features = convert_physical_output_to_node_features(step_physical_outputs, datasets, node_params, device)
            
            next_inputs = {}
            next_step_idx = t + 1 # Input for predicting T+2 needs Force at T+1 ? 
            # Wait, inputs[0] corresponds to T=1 state predicting T=2.
            # Next input should be T=2 state predicting T=3.
            # So force at T=2 should be used?
            # force_seq index: 0->T=1, 1->T=2.
            # So next_step_idx should be 1?
            # t=0 -> predicting T=2. Next loop t=1 needs input representing T=2.
            # Input at T=2 should consist of NodeState(T=2) + Force(T=2)?
            # dataset inputs[i] is T(i+1). 
            # inputs[1] is T=2.
            # So we need Force at T=2. Indicated by index 1.
            # So `next_step_idx = t + 1` seems correct. t=0 -> use index 1 (T=2).
            
            if next_step_idx < whole_force_seq.shape[0]:
                force_vec = whole_force_seq[next_step_idx] # (3,)
            else:
                 # Out of bounds (last step?)
                 force_vec = torch.zeros(3, device=device)

            for nid, node_feats in next_node_features.items():
                if nid not in dist_vecs:
                    continue
                
                dist_vec = dist_vecs[nid] # (3,)
                
                # Concatenate: [NodeFeats, Force, Dist]
                # node_feats: (F,)
                cat_input = torch.cat([node_feats, force_vec, dist_vec], dim=0)
                next_inputs[nid] = cat_input.unsqueeze(0) # (1, F+6)
                
            current_inputs = next_inputs

        save_results(idx, all_outputs_list, datasets, output_dir)
        print(f"Sample {idx+1} completed.")

def save_results(idx, all_outputs_list, datasets, output_dir):
    first_node = list(datasets.keys())[0]
    target_file_path = datasets[first_node].files[idx]
    target_file_stem = target_file_path.stem
    correct_df = pd.read_feather(target_file_path)
    
    columns = datasets[first_node].columns
    
    result_df_list = []
    
    for step, step_outputs in enumerate(all_outputs_list):
        time = step + 2
        
        step_data = {}
        for node_id, val in step_outputs.items():
            step_data[node_id] = val.cpu().numpy()
            
        if not step_data:
            continue
            
        time_df = pd.DataFrame.from_dict(step_data, orient='index')
        if len(time_df.columns) == len(columns):
            time_df.columns = columns
        else:
            # Padding or truncating? Should match
            time_df.columns = columns[:len(time_df.columns)]
            
        time_df['time'] = time
        time_df['node_id'] = time_df.index
        result_df_list.append(time_df)
        
    if not result_df_list:
        return

    result_df = pd.concat(result_df_list, axis=0)
    
    time1_correct = correct_df[correct_df['time'] == 1]
    time1_df = time1_correct[['time', 'node_id'] + columns].copy()
    
    result_df = pd.concat([time1_df, result_df], axis=0)
    result_df = result_df.sort_values(['node_id', 'time']).reset_index(drop=True)
    
    # Reconstruct Coords
    initial_df = time1_correct[['node_id', 'x', 'y', 'z']].rename(columns={'x': 'init_x', 'y': 'init_y', 'z': 'init_z'})
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
    
    gt_df = correct_df.copy()
    gt_df_renamed = gt_df.add_suffix('_correct')
    gt_df_renamed = gt_df_renamed.rename(columns={'time_correct': 'time', 'node_id_correct': 'node_id'})
    
    merged_df = pd.merge(result_df, gt_df_renamed, on=['time', 'node_id'], how='left')
    
    error_cols = ['x', 'y', 'z', 'dx', 'dy', 'dz', 'Sxx', 'Syy', 'Szz', 'Sxy', 'Syz', 'Szx']
    for col in error_cols:
        if col in merged_df.columns and f'{col}_correct' in merged_df.columns:
            merged_df[f'{col}_error'] = (merged_df[col] - merged_df[f'{col}_correct']).abs()
            
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
