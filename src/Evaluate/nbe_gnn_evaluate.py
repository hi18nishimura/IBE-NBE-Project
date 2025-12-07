from src.Dataloader.nbeGNNDataset import NbeGNNDataset
from src.Evaluate.nbe_gnn_load_weights import load_model_from_dir
from src.Evaluate.nbe_normalize_inverse import denormalize_nbe_outputs, oka_denormalize
from src.Networks.nbe_gnn import NbeGNN
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path

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

# NbeGNNDatasetの初期化
def all_node_nbe_dataset_init(data_dir, max_node, global_normalize: bool, alpha: float) -> dict[int, NbeGNNDataset]:
    node_datasets = {}
    for node_id in tqdm(range(1, max_node+1), desc="Dataset Init"):
        node_datasets[node_id] = NbeGNNDataset(data_dir=data_dir, node_id=node_id, global_normalize=global_normalize, alpha=alpha)
    return node_datasets

# Nbeパラメータの取得
def all_node_nbe_params(node_datasets: dict[int, NbeGNNDataset], hidden_size: int, num_layers: int, drop_out: float, gnn_type: str) -> dict[int, dict[str, int]]:
    node_params = {}
    all_neighbor_node = {}
    for node_id, dataset in tqdm(node_datasets.items(), desc="Params Init"):
        # Get input size from a sample
        sample_item = dataset[0]
        input_size = sample_item["inputs"].shape[-1]
        
        # Create max_values tensor ordered by columns
        max_vals_list = [dataset.max_map.get(col, 1.0) for col in dataset.columns]
        max_vals_tensor = torch.tensor(max_vals_list, dtype=torch.float32)
        
        node_params[node_id] = {
            "input_size": input_size,
            "output_size": dataset.target_feature_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "drop_out": drop_out,
            "gnn_type": gnn_type,
            "max_values": dataset.max_map,
            "max_values_tensor": max_vals_tensor,
            "alpha": dataset.alpha
        }
        all_neighbor_node[node_id] = dataset.node_order
    return node_params, all_neighbor_node

# NBEの重みを取得する
def all_nbe_weights_load(node_params: dict[int, dict[str, int]], weight_dir: str, map_location: torch.device | str, model_name: str = "best.pth") -> dict[int, NbeGNN]:
    nbe_models = {}
    for node_id, params in tqdm(node_params.items(), desc="Weights Load"):
        model, ckpt, path = load_model_from_dir(
            root=f"{weight_dir}/{node_id}",
            model_ctor=NbeGNN,
            model_ctor_kwargs={
                "input_size": params["input_size"],
                "hidden_size": params["hidden_size"],
                "output_size": params["output_size"],
                "num_layers": params["num_layers"],
                "dropout": params["drop_out"],
                "gnn_type": params["gnn_type"],
            },
            map_location=map_location,
            model_name=model_name,
        )
        if model is None:
             print(f"Warning: No model found for node {node_id} in {weight_dir}/{node_id}")
        nbe_models[node_id] = model
    return nbe_models

# 時刻1のデータを取得する
def get_nbe_gnn_dataset_time1_data(nbe_dataset_dict, max_node, idx):
    node_time1_data = {}
    node_edge_index = {}
    all_targets = {}
    for node_id in range(1, max_node+1):
        data = nbe_dataset_dict[node_id].__getitem__(idx)
        inputs, targets, edge_index = data["inputs"], data["targets"], data["edge_index"]
        
        # inputs: (19, N, F) -> time1: (N, F)
        time1_data = inputs[0, :]
        
        node_time1_data[node_id] = time1_data
        node_edge_index[node_id] = edge_index
        all_targets[node_id] = targets
    return node_time1_data, node_edge_index, all_targets

# GNNの出力を次の入力に変換する (物理量を経由して再正規化)
def convert_physical_output_to_input_gnn(physical_outputs_dict, neighborhood_dict, node_params, device):
    inputs_dict = {}
    for target_node, neighbors in neighborhood_dict.items():
        # target_node is the node whose model we are preparing input for.
        # We need to normalize neighbors' physical values using target_node's parameters.
        
        params = node_params[target_node]
        max_vals = params["max_values_tensor"].to(device)
        alpha = params["alpha"]
        
        input_list = []
        for neighbor_id in neighbors:
            if neighbor_id in physical_outputs_dict:
                phys_val = physical_outputs_dict[neighbor_id].to(device)
                # Normalize using target node's max values
                norm_val = oka_normalize_tensor(phys_val, max_vals, alpha)
                input_list.append(norm_val)
            else:
                # Handle missing node
                # If a neighbor is not in physical_outputs_dict (e.g. it wasn't predicted),
                # we might have a problem. 
                # Assuming all nodes in neighborhood are in range(1, max_node+1) and were predicted.
                pass
        
        if input_list:
            inputs_dict[target_node] = torch.stack(input_list, dim=0)
        else:
             inputs_dict[target_node] = torch.empty(0)

    return inputs_dict

def run_single_evaluation(target_idx, all_nbe_dataset, all_nbe_params, all_neighbor_node, all_nbe, device, output_dir):
    target_file_path = all_nbe_dataset[1].files[target_idx]
    target_file_stem = target_file_path.stem
    correct_df = pd.read_feather(target_file_path)
    
    print(f"Processing: {target_file_stem}")

    # 時刻1→2の推定
    all_outputs_list = []
    
    # max_nodeを取得 (datasetのキーの最大値)
    max_node = max(all_nbe_dataset.keys())

    # 時刻1の情報を取得する
    all_next_inputs, all_edge_indices, all_targets = get_nbe_gnn_dataset_time1_data(all_nbe_dataset, max_node, target_idx)
    
    all_nbe_outputs = {}
    
    # Move edge indices to device once
    for k, v in all_edge_indices.items():
        all_edge_indices[k] = v.to(device)

    # Prediction Loop
    zero_disp = torch.tensor([0.5, 0.5, 0.5])
    all_physical_outputs = {}

    # Time 2 prediction
    for key, inputs in all_next_inputs.items():
        if key not in all_nbe or all_nbe[key] is None:
            continue
            
        with torch.no_grad():
            inputs = inputs.to(device)
            edge_index = all_edge_indices[key]
            
            outputs = all_nbe[key](inputs, edge_index)
            pred = outputs[0]
            
            if pred.shape[0] == 6:
                pred = torch.cat([zero_disp.to(pred.device), pred], dim=0)
            
            all_nbe_outputs[key] = pred
            
            params = all_nbe_params[key]
            max_vals = params["max_values_tensor"].to(device)
            alpha = params["alpha"]
        
            phys_val = oka_denormalize(pred, max_vals, alpha)
            all_physical_outputs[key] = phys_val

    all_outputs_list.append(all_nbe_outputs.copy())
    all_next_inputs = convert_physical_output_to_input_gnn(all_physical_outputs, all_neighbor_node, all_nbe_params, device)

    # Time 3~20 prediction
    for time in tqdm(range(3, 21), desc=f"Time 3~20 ({target_file_stem})", leave=False):
        current_step_outputs = {}
        current_step_physical_outputs = {}
        
        for key, inputs in all_next_inputs.items():
            if key not in all_nbe or all_nbe[key] is None:
                continue

            with torch.no_grad():
                inputs = inputs.to(device)
                edge_index = all_edge_indices[key]
                
                outputs = all_nbe[key](inputs, edge_index)
                pred = outputs[0]
                
                if pred.shape[0] == 6:
                    pred = torch.cat([zero_disp.to(pred.device), pred], dim=0)
                
                current_step_outputs[key] = pred
                
                params = all_nbe_params[key]
                max_vals = params["max_values_tensor"].to(device)
                alpha = params["alpha"]
                phys_val = oka_denormalize(pred, max_vals, alpha)
                current_step_physical_outputs[key] = phys_val
        
        all_nbe_outputs = current_step_outputs
        all_physical_outputs = current_step_physical_outputs
        
        all_outputs_list.append(all_nbe_outputs.copy())
        all_next_inputs = convert_physical_output_to_input_gnn(all_physical_outputs, all_neighbor_node, all_nbe_params, device)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Denormalize all
    all_denormalized_outputs = denormalize_nbe_outputs(all_outputs_list, all_nbe_params, all_nbe_dataset)

    # Result DataFrame
    result_df_list = []
    for time in range(2, 21):
        if time not in all_denormalized_outputs:
            continue
        time_df = pd.DataFrame.from_dict(all_denormalized_outputs[time], orient='index')
        time_df.columns = all_nbe_dataset[1].columns
        time_df['time'] = time
        time_df['node_id'] = time_df.index
        result_df_list.append(time_df)
    
    if result_df_list:
        result_df = pd.concat(result_df_list, axis=0)
        
        time1_correct_df = correct_df[correct_df['time']==1]
        result_df = pd.concat([time1_correct_df[['time','node_id']+list(all_nbe_dataset[1].columns)], result_df], axis=0)
        result_df[['x','y','z']] = 0.0
        result_df.loc[result_df['time']==1, ['x','y','z']] = time1_correct_df[['x','y','z']].values

        result_df['cum_dx'] = result_df.groupby('node_id')['dx'].cumsum()
        result_df['cum_dy'] = result_df.groupby('node_id')['dy'].cumsum()
        result_df['cum_dz'] = result_df.groupby('node_id')['dz'].cumsum()

        initial_x = result_df.groupby('node_id')['x'].transform('first')
        initial_y = result_df.groupby('node_id')['y'].transform('first')
        initial_z = result_df.groupby('node_id')['z'].transform('first')

        result_df['x'] = initial_x + result_df['cum_dx']
        result_df['y'] = initial_y + result_df['cum_dy']
        result_df['z'] = initial_z + result_df['cum_dz']

        result_df = result_df.drop(columns=['cum_dx', 'cum_dy', 'cum_dz'])
        result_df = result_df.reset_index(drop=True)

        merged_df = pd.merge(result_df, correct_df[['time', 'node_id', 'x', 'y', 'z', 'Sxx', 'Syy', 'Szz', 'Sxy', 'Syz', 'Szx']], 
                             on=['time', 'node_id'], suffixes=('', '_correct'))
        
        for col in ['x', 'y', 'z', 'Sxx', 'Syy', 'Szz', 'Sxy', 'Syz', 'Szx']:
            merged_df[f'{col}_error'] = abs(merged_df[col] - merged_df[f'{col}_correct'])

        cols_to_keep = ['time', 'node_id'] + \
                       ['x', 'y', 'z', 'Sxx', 'Syy', 'Szz', 'Sxy', 'Syz', 'Szx'] + \
                       [f'{col}_error' for col in ['x', 'y', 'z', 'Sxx', 'Syy', 'Szz', 'Sxy', 'Syz', 'Szx']]
        
        final_df = merged_df[cols_to_keep]

        output_csv = output_dir / f"{target_file_stem}_results.csv"
        final_df.to_csv(output_csv, index=False)
        print(f"Saved results to {output_csv}")
    else:
        print(f"No results generated for {target_file_stem}.")


if __name__ == "__main__":
    # コマンドライン引数の解析
    import argparse

    parser = argparse.ArgumentParser(description="NbeGNNのモデル可視化プログラムです。")
    parser.add_argument("weight_dir", help="Directory to search for best.pth")
    parser.add_argument("dataset_dir", help="テスト用データの保存ディレクトリを指定してください。")
    parser.add_argument("--all_node_id", type=int, default=140)
    parser.add_argument("--target_idx", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--gnn_type", type=str, default='GAT')
    parser.add_argument("--device", default=None, help="torch device string, e.g. cpu or cuda:0")
    parser.add_argument("--global_normalize", action="store_true", help="Whether to apply global normalization to the dataset")
    parser.add_argument("--output_dir", default="/workspace/results", help="Directory to save evaluation results")
    parser.add_argument("--alpha", type=float, default=8.0, help="Oka normalization alpha parameter")
    parser.add_argument("--files_num", type=int, default=None, help="Number of files to process from index 0")
    parser.add_argument("--model_name", type=str, default="best_pretrain.pth", help="Name of the model file to load (default: best_pretrain.pth)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # NbeDatasetの読み込み
    print("NbeDatasetの初期化")
    all_nbe_dataset = all_node_nbe_dataset_init(args.dataset_dir,
                                                args.all_node_id, 
                                                global_normalize=args.global_normalize,
                                                alpha=args.alpha)
    
    # Check if dataset is empty or failed
    if not all_nbe_dataset:
        print("Failed to initialize datasets.")
        exit(1)

    target_file_path = all_nbe_dataset[1].files[args.target_idx]
    target_file_stem = target_file_path.stem
    correct_df = pd.read_feather(target_file_path)
    
    # Nbeパラメータの取得
    print("Nbeパラメータの取得")
    all_nbe_params, all_neighbor_node = all_node_nbe_params(all_nbe_dataset, 
                                                            args.hidden_size, 
                                                            args.num_layers, 
                                                            args.dropout,
                                                            args.gnn_type)
    
    print(all_nbe_dataset[1].files[args.target_idx])
    # NBEモデルの取得
    print("NBEモデルの取得")
    all_nbe = all_nbe_weights_load(all_nbe_params, args.weight_dir, map_location=device, model_name=args.model_name)

    # Determine indices to process
    if args.files_num is not None:
        target_indices = range(args.files_num)
    else:
        target_indices = [args.target_idx]

    for target_idx in target_indices:
        run_single_evaluation(target_idx, all_nbe_dataset, all_nbe_params, all_neighbor_node, all_nbe, device, args.output_dir)
