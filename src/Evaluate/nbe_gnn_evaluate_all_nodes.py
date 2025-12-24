import sys
import os
import argparse
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append("/workspace/src")

from Networks.nbe_gnn import NbeGNN
from Dataloader.nbeGNNDataset import NbeGNNDataset
from Evaluate.nbe_normalize_inverse import oka_denormalize

def infer_model_config(state_dict):
    config = {}
    
    # 1. Detect GNN Type
    # GAT has 'att_src', 'att_dst' in keys or 'lin_src'
    # Based on error message, GAT has 'att_src'
    is_gat = any('att_src' in k for k in state_dict.keys())
    config['gnn_type'] = 'GAT' if is_gat else 'GCN'
    
    # 2. Detect num_layers
    # Find max index of convs.X
    conv_indices = [int(k.split('.')[1]) for k in state_dict.keys() if k.startswith('convs.') and k.split('.')[1].isdigit()]
    if conv_indices:
        config['num_layers'] = max(conv_indices) + 1
    
    # 3. Detect hidden_size and input_size
    # Look at convs.0
    if 'convs.0.lin.weight' in state_dict:
        w = state_dict['convs.0.lin.weight']
        config['hidden_size'] = w.shape[0]
        config['input_size'] = w.shape[1]
    elif 'convs.0.weight' in state_dict: # Standard GCN sometimes
        w = state_dict['convs.0.weight']
        config['hidden_size'] = w.shape[0]
        config['input_size'] = w.shape[1]
        
    # 4. Detect multi_fully_layer and output_size
    # Check readout keys
    readout_keys = [k for k in state_dict.keys() if k.startswith('readout')]
    
    if 'readout.weight' in state_dict:
        # Single layer
        config['multi_fully_layer'] = 1
        w = state_dict['readout.weight']
        config['output_size'] = w.shape[0]
    else:
        # Multi layer
        # Find max index in readout.X
        indices = [int(k.split('.')[1]) for k in readout_keys if k.split('.')[1].isdigit()]
        if indices:
            max_idx = max(indices)
            # Assuming structure Linear -> ReLU -> Dropout
            # 0, 3, 6 ...
            config['multi_fully_layer'] = (max_idx // 3) + 1
            
            # Output size is from the last layer
            last_w_key = f'readout.{max_idx}.weight'
            if last_w_key in state_dict:
                w = state_dict[last_w_key]
                config['output_size'] = w.shape[0]

    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing model checkpoints")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing test feather files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output CSVs")
    parser.add_argument("--liver_coord_file", type=str, default="/workspace/dataset/liver_model_info/liver_coordinates.csv")
    parser.add_argument("--input_size", type=int, default=9)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--output_size", type=int, default=9, help="Output size of the network (9 or 6)")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--gnn_type", type=str, default="GCN")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Get all node IDs
    if os.path.exists(args.liver_coord_file):
        df_coords = pd.read_csv(args.liver_coord_file)
        all_node_ids = sorted(df_coords['node_id'].unique())
    else:
        print(f"Error: Liver coordinate file not found at {args.liver_coord_file}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Columns for output CSV
    feature_cols = ["dx", "dy", "dz", "Sxx", "Syy", "Szz", "Sxy", "Syz", "Szx"]
    out_cols = ["time", "node_id"] + feature_cols + [f"{c}_abs_err" for c in feature_cols] + [f"{c}_rel_err" for c in feature_cols]

    print(f"Starting evaluation for {len(all_node_ids)} nodes...")

    for node_id in tqdm(all_node_ids, desc="Evaluating Nodes"):
        # Load model for this node
        model_path = Path(args.model_dir) / f"{node_id}" / "best_pretrain.pth"
        if not model_path.exists():
            # print(f"Model not found for node {node_id}, skipping...")
            continue

        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            state_dict = None
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'model_state' in checkpoint:
                    state_dict = checkpoint['model_state']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint # Assuming it's the dict itself
            
            # Infer parameters
            inferred_config = infer_model_config(state_dict)
            
            input_size = inferred_config.get('input_size', args.input_size)
            hidden_size = inferred_config.get('hidden_size', args.hidden_size)
            output_size = inferred_config.get('output_size', args.output_size)
            num_layers = inferred_config.get('num_layers', args.num_layers)
            gnn_type = inferred_config.get('gnn_type', args.gnn_type)
            multi_fully_layer = inferred_config.get('multi_fully_layer', 1)
            
            # print(f"Node {node_id}: Inferred config: {inferred_config}")

            model = NbeGNN(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                num_layers=num_layers,
                gnn_type=gnn_type,
                multi_fully_layer=multi_fully_layer,
                return_only_central=True
            ).to(device)
            
            model.load_state_dict(state_dict)
            model.eval()
        except Exception as e:
            print(f"Failed to load model for node {node_id}: {e}")
            continue

        try:
            # Initialize dataset for this node
            # Note: This will filter files based on proximity to the node
            dataset = NbeGNNDataset(
                data_dir=args.data_dir,
                node_id=node_id,
                liver_coord_file=None,
                # Assuming default alpha=8.0 and other paths are resolved automatically or via defaults
            )
        except Exception as e:
            # print(f"Skipping node {node_id}: {e}")
            continue

        if len(dataset) == 0:
            continue

        # Prepare max values for denormalization
        # dataset.max_map contains max values for each feature
        try:
            max_vals_list = [dataset.max_map[c] for c in feature_cols]
            max_vals_tensor = torch.tensor(max_vals_list, device=device)
        except KeyError as e:
            print(f"Error: Missing key in max_map for node {node_id}: {e}")
            continue
        
        node_results = []
        is_fixed = dataset.fixed_nodes.get(node_id, False)

        for i in range(len(dataset)):
            try:
                data = dataset[i]
                inputs = data['inputs'].to(device) # (19, N, F)
                targets = data['targets'].to(device) # (19, F_out)
                edge_index = data['edge_index'].to(device)
                
                # Add batch dim for model
                inputs_batch = inputs.unsqueeze(0) # (1, 19, N, F)
                
                with torch.no_grad():
                    preds = model(inputs_batch, edge_index)
                    preds = preds.squeeze(0) # (19, Out)

                # --- Denormalization & Error Calculation ---
                
                # 1. Prepare Denormalized Predictions (preds_denorm)
                if output_size == 9:
                    preds_denorm = oka_denormalize(preds, max_vals_tensor)
                elif output_size == 6:
                    # Denormalize stress (last 6 components)
                    max_vals_stress = max_vals_tensor[3:]
                    preds_stress_denorm = oka_denormalize(preds, max_vals_stress)
                    
                    # Combine with 0s for displacement
                    preds_denorm = torch.zeros((19, 9), device=device)
                    preds_denorm[:, 3:] = preds_stress_denorm
                    # dx, dy, dz remain 0
                else:
                    print(f"Unsupported output size: {output_size}")
                    continue

                # 2. Prepare Denormalized Targets (targets_denorm)
                # If node is fixed, NbeGNNDataset returns only 6 columns (stresses)
                if is_fixed:
                    # targets shape is (19, 6)
                    max_vals_stress = max_vals_tensor[3:]
                    targets_stress_denorm = oka_denormalize(targets, max_vals_stress)
                    
                    targets_denorm = torch.zeros((19, 9), device=device)
                    targets_denorm[:, 3:] = targets_stress_denorm
                    # dx, dy, dz are 0 for fixed nodes
                else:
                    # targets shape is (19, 9)
                    targets_denorm = oka_denormalize(targets, max_vals_tensor)

                # 3. Calculate Errors
                abs_err = torch.abs(preds_denorm - targets_denorm)
                
                epsilon = 1e-8
                rel_err = abs_err / (torch.abs(targets_denorm) + epsilon)
                
                # 4. Save Results
                preds_np = preds_denorm.cpu().numpy()
                abs_err_np = abs_err.cpu().numpy()
                rel_err_np = rel_err.cpu().numpy()
                
                # Time steps: 2 to 20
                times = np.arange(2, 21)
                
                df_res = pd.DataFrame(preds_np, columns=feature_cols)
                df_res['time'] = times
                df_res['node_id'] = node_id
                
                for idx, col in enumerate(feature_cols):
                    df_res[f"{col}_abs_err"] = abs_err_np[:, idx]
                    df_res[f"{col}_rel_err"] = rel_err_np[:, idx]
                
                # Reorder columns
                df_res = df_res[out_cols]
                node_results.append(df_res)
            
            except Exception as e:
                print(f"Error processing file {i} for node {node_id}: {e}")
                continue
            
        if node_results:
            final_df = pd.concat(node_results, ignore_index=True)
            output_csv_path = output_dir / f"node{node_id}_eval.csv"
            final_df.to_csv(output_csv_path, index=False)

    print("Evaluation complete.")

if __name__ == "__main__":
    main()
