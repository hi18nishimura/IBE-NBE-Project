import argparse
import os
import sys
import glob
import torch
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
import pandas as pd

# Add workspace root to sys.path
sys.path.append("/workspace")

from src.Networks.nbe_gnn import NbeGNN
from src.Dataloader.nbeGNNDataset import NbeGNNDataset
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
    parser = argparse.ArgumentParser(description="Evaluate NBE GNN models")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing model subdirectories")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing dataset files")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save results")
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
    neighbor_map = {}
    datasets = {}
    
    # Iterate over subdirectories in model_dir
    subdirs = [d for d in model_dir_path.iterdir() if d.is_dir()]
    
    # Try to find auxiliary files in dataset_dir
    summary_overall_max = None
    summary_per_node_max = None
    
    possible_overall = list(dataset_dir_path.rglob("summary_overall_max_values.csv"))
    if possible_overall:
        summary_overall_max = possible_overall[0]
        
    possible_per_node = list(dataset_dir_path.rglob("summary_per_node_max_values.csv"))
    if possible_per_node:
        summary_per_node_max = possible_per_node[0]
        
    node_connection_file = None
    fixed_nodes_file = None
    
    possible_nc = list(dataset_dir_path.rglob("node_connections.csv"))
    if possible_nc:
        node_connection_file = possible_nc[0]
        
    possible_fn = list(dataset_dir_path.rglob("fixed_nodes.csv"))
    if possible_fn:
        fixed_nodes_file = possible_fn[0]

    for subdir in tqdm(subdirs, desc="Loading models"):
        try:
            node_id = int(subdir.name)
        except ValueError:
            continue 
            
        finetune_path = subdir / "best_finetune.pth"
        pretrain_path = subdir / "best_pretrain.pth"
        
        model_path = None
        if finetune_path.exists():
            model_path = finetune_path
        elif pretrain_path.exists():
            model_path = pretrain_path
            
        if model_path is None:
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
             print(f"Config is not a dict for {model_path}, type: {type(cfg)}")
             continue
             
        # Try to clean up potential OmegaConf objects inside the dict
        try:
            cfg = OmegaConf.create(cfg)
            cfg = OmegaConf.to_container(cfg, resolve=True)
        except Exception as e:
            # If conversion fails, proceed but warn
            # print(f"Warning: Failed to clean cfg with OmegaConf: {e}")
            pass

        try:
            alpha = cfg.get('alpha', 8.0)
            global_normalize = cfg.get('global_normalize', True)
            glob_pattern = cfg.get('glob', '*.feather')
        except Exception as e:
            print(f"Error accessing config: {e}, type: {type(cfg)}")
            continue
        
        summary_file = summary_overall_max if global_normalize else summary_per_node_max
        
        try:
            dataset = NbeGNNDataset(
                data_dir=dataset_dir_path,
                node_id=node_id,
                alpha=alpha,
                global_normalize=global_normalize,
                glob=glob_pattern,
                summary_overall_max=summary_file,
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
        input_size = sample['inputs'].shape[-1]
        output_size = dataset.target_feature_size
        
        hidden_size = cfg.get('hidden_size', 64)
        num_layers = cfg.get('num_layers', 2)
        dropout = cfg.get('dropout', 0.5)
        gnn_type = cfg.get('gnn_type', 'GAT')
        finetune = cfg.get('finetune', False)
        
        model = NbeGNN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            dropout=dropout,
            gnn_type=gnn_type,
            finetune=finetune
        )
        
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            
        model.eval()
        models[node_id] = model
        
        # Store params
        max_vals_list = [dataset.max_map.get(col, 1.0) for col in dataset.columns]
        max_vals_tensor = torch.tensor(max_vals_list, dtype=torch.float32)
        
        node_params[node_id] = {
            "input_size": input_size,
            "output_size": output_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "gnn_type": gnn_type,
            "max_values": dataset.max_map,
            "max_values_tensor": max_vals_tensor,
            "alpha": alpha
        }
        neighbor_map[node_id] = dataset.node_order
        datasets[node_id] = dataset
        
    return models, node_params, neighbor_map, datasets

def get_nbe_gnn_dataset_time1_data(nbe_dataset_dict, idx):
    node_time1_data = {}
    node_edge_index = {}
    all_targets = {}
    
    for node_id, dataset in nbe_dataset_dict.items():
        data = dataset[idx]
        inputs, targets, edge_index = data["inputs"], data["targets"], data["edge_index"]
        
        # inputs: (Time, N, F) -> time1: (1, 1, N, F)
        # Assuming inputs is (Time, N, F)
        # We want the first time step.
        time1_data = inputs[0, :].unsqueeze(0).unsqueeze(0)
        
        node_time1_data[node_id] = time1_data
        node_edge_index[node_id] = edge_index
        all_targets[node_id] = targets
        
    return node_time1_data, node_edge_index, all_targets

def get_nbe_gnn_dataset_all_data(nbe_dataset_dict, idx):
    node_all_inputs = {}
    node_edge_index = {}
    all_targets = {}
    
    for node_id, dataset in nbe_dataset_dict.items():
        data = dataset[idx]
        inputs, targets, edge_index = data["inputs"], data["targets"], data["edge_index"]
        
        # inputs: (Time, N, F)
        node_all_inputs[node_id] = inputs
        node_edge_index[node_id] = edge_index
        all_targets[node_id] = targets
        
    return node_all_inputs, node_edge_index, all_targets

def convert_physical_output_to_input_gnn(physical_outputs_dict, neighbor_map, node_params, device):
    inputs_dict = {}
    for target_node, neighbors in neighbor_map.items():
        params = node_params[target_node]
        max_vals = params["max_values_tensor"].to(device)
        alpha = params["alpha"]
        
        input_list = []
        # neighbors includes central node at index 0
        for neighbor_id in neighbors:
            if neighbor_id in physical_outputs_dict:
                phys_val = physical_outputs_dict[neighbor_id].to(device)
                # Normalize using target node's max values
                norm_val = oka_normalize_tensor(phys_val, max_vals, alpha)
                input_list.append(norm_val)
            else:
                # Handle missing node (should not happen if all nodes are predicted)
                pass
        
        if input_list:
            # Stack neighbors: (N_neighbors, F)
            stacked_inputs = torch.stack(input_list, dim=0)
            # Add Batch and Time dimensions: (1, 1, N_neighbors, F)
            inputs_dict[target_node] = stacked_inputs.unsqueeze(0).unsqueeze(0)
        else:
             inputs_dict[target_node] = torch.empty(0)

    return inputs_dict

def run_evaluation(models, node_params, neighbor_map, datasets, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move models to device
    for model in models.values():
        model.to(device)
        
    if not datasets:
        print("No datasets loaded.")
        return

    # Assume all datasets have same length
    first_node = list(datasets.keys())[0]
    num_samples = len(datasets[first_node])
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for idx in range(num_samples):
        print(f"Evaluating sample {idx+1}/{num_samples}")
        
        # Get all data for current sample
        all_inputs_dict, all_edge_indices, all_targets = get_nbe_gnn_dataset_all_data(datasets, idx)
        
        # Move edge indices to device
        for k, v in all_edge_indices.items():
            all_edge_indices[k] = v.to(device)
            
        all_outputs_list = []
        
        # Determine number of steps
        # targets usually has shape (Time, OutputSize)
        # inputs usually has shape (Time, N, F)
        # We want to predict for each time step where we have a target.
        target_len = all_targets[first_node].shape[0]
        
        zero_disp = torch.tensor([0.5, 0.5, 0.5], device=device)
        
        # Predict loop
        # We iterate through time steps.
        # If inputs has length T, and targets has length T.
        # inputs[t] predicts targets[t] (which corresponds to state at t+1 relative to input t?)
        # Actually, usually inputs[t] is state at time t, and targets[t] is state at time t+1.
        
        for t in tqdm(range(target_len), desc=f"Sample {idx+1} steps"):
            all_nbe_outputs = {}
            all_physical_outputs = {}
            
            for node_id in models:
                if node_id not in all_inputs_dict:
                    continue
                    
                model = models[node_id]
                edge_index = all_edge_indices[node_id]
                
                # Get input for current time step t
                # inputs tensor is (Time, N, F)
                # We need (1, 1, N, F) for the model
                if t < all_inputs_dict[node_id].shape[0]:
                    inputs = all_inputs_dict[node_id][t].unsqueeze(0).unsqueeze(0).to(device)
                else:
                    # Should not happen if target_len matches inputs len
                    continue
                
                with torch.no_grad():
                    outputs = model(inputs, edge_index)
                    
                    if outputs.dim() == 4:
                        # (B, S, N, Out) -> take central node (index 0)
                        pred = outputs[0, 0, 0, :]
                    elif outputs.dim() == 3:
                        # (B, S, Out) -> (1, 1, Out)
                        pred = outputs[0, 0, :]
                    else:
                        pred = outputs
                        
                    if pred.shape[0] == 6:
                        pred = torch.cat([zero_disp, pred], dim=0)
                        
                    all_nbe_outputs[node_id] = pred
                    
                    # Denormalize
                    params = node_params[node_id]
                    max_vals = params["max_values_tensor"].to(device)
                    alpha = params["alpha"]
                    
                    phys_val = oka_denormalize(pred, max_vals, alpha)
                    all_physical_outputs[node_id] = phys_val
            
            all_outputs_list.append(all_physical_outputs)
            
        # Save results
        save_results(idx, all_outputs_list, datasets, output_dir)
        
        # For now, just print shape of last output
        print(f"Sample {idx+1} completed. Steps: {len(all_outputs_list)}")

def save_results(idx, all_outputs_list, datasets, output_dir):
    first_node = list(datasets.keys())[0]
    target_file_path = datasets[first_node].files[idx]
    target_file_stem = target_file_path.stem
    correct_df = pd.read_feather(target_file_path)
    
    columns = datasets[first_node].columns
    
    result_df_list = []
    
    # Time 2 to T
    for step, step_outputs in enumerate(all_outputs_list):
        time = step + 2
        # step_outputs is {node_id: tensor}
        
        # Convert to dict of numpy arrays
        step_data = {}
        for node_id, val in step_outputs.items():
            step_data[node_id] = val.cpu().numpy()
            
        if not step_data:
            continue
            
        time_df = pd.DataFrame.from_dict(step_data, orient='index')
        
        # Ensure columns match
        if len(time_df.columns) == len(columns):
            time_df.columns = columns
        else:
            # Fallback if mismatch (e.g. extra columns or missing)
            # Assuming order is correct
            time_df.columns = columns[:len(time_df.columns)]
            
        time_df['time'] = time
        time_df['node_id'] = time_df.index
        result_df_list.append(time_df)
        
    if not result_df_list:
        return

    result_df = pd.concat(result_df_list, axis=0)
    
    # Add time 1 from correct_df
    time1_correct_df = correct_df[correct_df['time']==1]
    
    # Create time 1 dataframe with same columns
    time1_df = time1_correct_df[['time', 'node_id'] + columns].copy()
    
    # Concatenate
    result_df = pd.concat([time1_df, result_df], axis=0)
    result_df = result_df.sort_values(['node_id', 'time']).reset_index(drop=True)
    
    # Reconstruct x, y, z
    # Initialize with 0
    result_df['x'] = 0.0
    result_df['y'] = 0.0
    result_df['z'] = 0.0
    
    # Set initial positions (time 1)
    # We map from time1_correct_df
    initial_pos_map = time1_correct_df.set_index('node_id')[['x', 'y', 'z']].to_dict('index')
    
    # Function to apply initial pos
    def set_initial(row):
        if row['time'] == 1:
            if row['node_id'] in initial_pos_map:
                return pd.Series(initial_pos_map[row['node_id']])
        return pd.Series([0.0, 0.0, 0.0], index=['x', 'y', 'z'])

    # Optimization: Set time 1 values directly
    # result_df is sorted by node_id, time.
    # Time 1 rows are the first for each node.
    
    # Let's use the cumulative sum approach which is faster
    result_df['cum_dx'] = result_df.groupby('node_id')['dx'].cumsum()
    result_df['cum_dy'] = result_df.groupby('node_id')['dy'].cumsum()
    result_df['cum_dz'] = result_df.groupby('node_id')['dz'].cumsum()

    # Get initial x,y,z for each node
    # We can merge initial positions to the dataframe
    initial_df = time1_correct_df[['node_id', 'x', 'y', 'z']].rename(columns={'x':'init_x', 'y':'init_y', 'z':'init_z'})
    result_df = pd.merge(result_df, initial_df, on='node_id', how='left')

    result_df['x'] = result_df['init_x'] + result_df['cum_dx'] - result_df.groupby('node_id')['dx'].transform('first')
    result_df['y'] = result_df['init_y'] + result_df['cum_dy'] - result_df.groupby('node_id')['dy'].transform('first')
    result_df['z'] = result_df['init_z'] + result_df['cum_dz'] - result_df.groupby('node_id')['dz'].transform('first')
    
    # Drop temp columns
    result_df = result_df.drop(columns=['cum_dx', 'cum_dy', 'cum_dz', 'init_x', 'init_y', 'init_z'])
    
    # Merge with correct_df for errors
    merged_df = pd.merge(result_df, correct_df, on=['time', 'node_id'], suffixes=('', '_correct'), how='left')
    
    # Calculate errors
    error_cols = ['x', 'y', 'z', 'dx', 'dy', 'dz', 'Sxx', 'Syy', 'Szz', 'Sxy', 'Syz', 'Szx']
    for col in error_cols:
        if col in merged_df.columns and f'{col}_correct' in merged_df.columns:
            merged_df[f'{col}_error'] = (merged_df[col] - merged_df[f'{col}_correct']).abs()
            
    # Select columns to save
    cols_to_keep = ['time', 'node_id']
    for col in error_cols:
        if col in merged_df.columns:
            cols_to_keep.append(col)
        if f'{col}_error' in merged_df.columns:
            cols_to_keep.append(f'{col}_error')
            
    final_df = merged_df[cols_to_keep]
    
    output_csv = Path(output_dir) / f"{target_file_stem}_results.csv"
    final_df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")

if __name__ == "__main__":
    args = parse_args()
    models, node_params, neighbor_map, datasets = load_models(args.model_dir, args.dataset_dir)
    print(f"Loaded {len(models)} models.")
    
    run_evaluation(models, node_params, neighbor_map, datasets, args.output_dir)
