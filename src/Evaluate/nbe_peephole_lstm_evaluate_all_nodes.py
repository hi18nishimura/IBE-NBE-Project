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

from Networks.nbe_peephole_lstm import NbePeepholeLSTM
from Dataloader.nbeDataset import NbeDataset
from Evaluate.nbe_normalize_inverse import oka_denormalize

def infer_lstm_config(state_dict):
    config = {}
    
    # 1. Detect num_layers
    # Count cells.X
    cell_indices = [int(k.split('.')[1]) for k in state_dict.keys() if k.startswith('cells.') and k.split('.')[1].isdigit()]
    if cell_indices:
        config['num_layers'] = max(cell_indices) + 1
    else:
        config['num_layers'] = 1

    # 2. Detect hidden_size and input_size
    # Look at cells.0.wx.weight (shape: 4*hidden, input)
    if 'cells.0.wx.weight' in state_dict:
        w = state_dict['cells.0.wx.weight']
        config['hidden_size'] = w.shape[0] // 4
        config['input_size'] = w.shape[1]
        
    # 3. Detect output_size
    # Look at out_proj.weight (shape: output, hidden)
    if 'out_proj.weight' in state_dict:
        w = state_dict['out_proj.weight']
        config['output_size'] = w.shape[0]

    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing model checkpoints")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing test feather files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output CSVs")
    parser.add_argument("--liver_coord_file", type=str, default="/workspace/dataset/liver_model_info/liver_coordinates.csv")
    # Default params (will be overwritten by inference)
    parser.add_argument("--input_size", type=int, default=9)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--output_size", type=int, default=9)
    parser.add_argument("--num_layers", type=int, default=1)
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
        # Assuming structure: model_dir/target_{node_id}/best_model.pth or similar
        # The user used "target_{node_id}" in previous request, but then "node_id" in the file path check.
        # I will try both or stick to what was working.
        # In the GNN script, I changed it to `args.model_dir / f"{node_id}" / "best_pretrain.pth"` based on user feedback/context.
        # But for LSTM, the folder structure might be different.
        # Let's assume `target_{node_id}` as per common practice in this project, or check if user specified.
        # The user said "target_1をnode_idに変更して" for GNN.
        # Let's try `target_{node_id}` first, if not found try `{node_id}`.
        
        model_path = Path(args.model_dir) / f"{node_id}" / "best.pth"
        if not model_path.exists():
             model_path = Path(args.model_dir) / f"{node_id}" / "best.pth"
        
        if not model_path.exists():
            # print(model_path)
            print(f"Model not found for node {node_id}, skipping...")
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
                state_dict = checkpoint
            
            # Infer parameters
            inferred_config = infer_lstm_config(state_dict)
            
            input_size = inferred_config.get('input_size', args.input_size)
            hidden_size = inferred_config.get('hidden_size', args.hidden_size)
            output_size = inferred_config.get('output_size', args.output_size)
            num_layers = inferred_config.get('num_layers', args.num_layers)
            
            model = NbePeepholeLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                num_layers=num_layers,
            ).to(device)
            
            model.load_state_dict(state_dict)
            model.eval()
        except Exception as e:
            print(f"Failed to load model for node {node_id}: {e}")
            continue

        try:
            # Initialize dataset for this node
            dataset = NbeDataset(
                data_dir=args.data_dir,
                node_id=node_id
            )
        except Exception as e:
            print(f"Error initializing dataset for node {node_id}: {e}")
            continue

        if len(dataset) == 0:
            continue

        # Prepare max values for denormalization
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
                inputs = data['inputs'].to(device) # (19, Input_F)
                targets = data['targets'].to(device) # (19, Output_F)
                
                # Add batch dim for model
                inputs_batch = inputs.unsqueeze(0) # (1, 19, Input_F)
                
                with torch.no_grad():
                    preds, _ = model(inputs_batch)
                    preds = preds.squeeze(0)

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
                if is_fixed:
                    # targets shape is (19, 6)
                    max_vals_stress = max_vals_tensor[3:]
                    targets_stress_denorm = oka_denormalize(targets, max_vals_stress)
                    
                    targets_denorm = torch.zeros((19, 9), device=device)
                    targets_denorm[:, 3:] = targets_stress_denorm
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
