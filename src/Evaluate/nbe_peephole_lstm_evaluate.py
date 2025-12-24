from src.Dataloader.nbeDataset import NbeDataset
from src.Evaluate.nbe_peephole_lstm_load_weights import load_model_from_dir
from src.Evaluate.nbe_output_parse import convert_nbe_output_to_input
from src.Evaluate.nbe_normalize_inverse import denormalize_nbe_outputs
from src.Evaluate.nbe_dataset_time1_data import get_nbe_dataset_time1_data
from src.Networks.nbe_peephole_lstm import NbePeepholeLSTM
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path

# NbeDatasetの初期化
def all_node_nbe_dataset_init(data_dir, max_node, global_normalize: bool, alpha: float) -> dict[int, NbeDataset]:
    node_datasets = {}
    for node_id in tqdm(range(1, max_node+1)):
        node_datasets[node_id] = NbeDataset(data_dir=data_dir, node_id=node_id, global_normalize=global_normalize, alpha=alpha)
    return node_datasets

# Nbeパラメータの取得
def all_node_nbe_params(node_datasets: dict[int, NbeDataset],hidden_size: int,num_layers: int,drop_out: float) -> dict[int, dict[str, int]]:
    node_params = {}
    all_neighbor_node = {}
    for node_id, dataset in tqdm(node_datasets.items()):
        node_params[node_id] = {
            "input_size": dataset.input_feature_size,
            "output_size": dataset.target_feature_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "drop_out": drop_out,
            "max_values": dataset.max_map
        }
        all_neighbor_node[node_id] = dataset.node_order
    return node_params,all_neighbor_node

# NBEの重みを取得する
def all_nbe_weights_load(node_params: dict[int, dict[str, int]], weight_dir: str, map_location: torch.device | str, model_name: str = "best.pth") -> dict[int, NbePeepholeLSTM]:
    nbe_models = {}
    for node_id, params in tqdm(node_params.items()):
        model, ckpt, path = load_model_from_dir(
            root=f"{weight_dir}/{node_id}",
            model_ctor=NbePeepholeLSTM,
            model_ctor_kwargs={
                "input_size": params["input_size"],
                "hidden_size": params["hidden_size"],
                "output_size": params["output_size"],
                "num_layers": params["num_layers"],
                "dropout": params["drop_out"],
            },
            map_location=map_location,
            model_name=model_name,
        )
        if model is None:
            print(f"Warning: Model for node {node_id} not found in {weight_dir}/{node_id} with name {model_name}")
        nbe_models[node_id] = model
    return nbe_models

def run_single_evaluation(target_idx, all_nbe_dataset, all_nbe_params, all_neighbor_node, all_nbe, device, output_dir, all_node_id):
    target_file_path = all_nbe_dataset[1].files[target_idx]
    target_file_stem = target_file_path.stem
    correct_df = pd.read_feather(target_file_path)
    
    print(f"Processing: {target_file_stem}")

    # 時刻1→2の推定
    all_outputs_list = []
    # 時刻1の情報を取得する
    # print("時刻1の情報を取得する")
    all_next_inputs,all_targets = get_nbe_dataset_time1_data(all_nbe_dataset, all_node_id, target_idx)
    all_lstm_gates = {}
    all_nbe_outputs = {}
    for key,values in tqdm(all_next_inputs.items(),desc="時刻2の推定", leave=False):
        inputs = values.unsqueeze(0)  # バッチ次元の追加
        with torch.no_grad():
            if all_nbe[key] is None:
                raise RuntimeError(f"Model for node {key} is not loaded. Cannot proceed.")
            inputs = inputs.to(next(all_nbe[key].parameters()).device)
            outputs, (h,c) = all_nbe[key].predict_next(inputs)
            all_lstm_gates[key] = (h, c)
            all_nbe_outputs[key] = outputs.squeeze(0)  # バッチ次元の削除
    
    # 全時刻の情報を保存する
    all_outputs_list.append(all_nbe_outputs.copy())
    # NBEの出力を次の出力にする
    all_next_inputs = convert_nbe_output_to_input(all_nbe_outputs, all_neighbor_node)

    # # 時刻3~20の推定
    for time in tqdm(range(3,21),desc="時刻3~20の推定", leave=False):
        for key,values in all_next_inputs.items():
            inputs = values.unsqueeze(0)  # バッチ次元の追加
            with torch.no_grad():
                if all_nbe[key] is None:
                    raise RuntimeError(f"Model for node {key} is not loaded. Cannot proceed.")
                inputs = inputs.to(next(all_nbe[key].parameters()).device)
                outputs, (h,c) = all_nbe[key].predict_next(inputs, all_lstm_gates[key])
                all_lstm_gates[key] = (h, c)
                all_nbe_outputs[key] = outputs.squeeze(0)  # バッチ次元の削除
        # 全時刻の情報を保存する
        all_outputs_list.append(all_nbe_outputs.copy())
        # NBEの出力を次の出力にする
        all_next_inputs = convert_nbe_output_to_input(all_nbe_outputs, all_neighbor_node)
    
    # 全時刻の正規化を元に戻す
    all_denormalized_outputs = denormalize_nbe_outputs(all_outputs_list, all_nbe_params,all_nbe_dataset)

    # 全時刻の結果をDataFrameに変換する
    result_df_list = []
    raw_df_list = []
    for time in range(2,21):
        time_df = pd.DataFrame.from_dict(all_denormalized_outputs[time], orient='index')
        time_df.columns = all_nbe_dataset[1].columns
        time_df['time'] = time
        time_df['node_id'] = time_df.index
        result_df_list.append(time_df)

        # Raw outputs
        list_idx = time - 2
        if list_idx < len(all_outputs_list):
            step_outputs = all_outputs_list[list_idx]
            if step_outputs:
                # Convert tensors to numpy
                step_data = {k: v.cpu().numpy() for k, v in step_outputs.items()}
                raw_step_df = pd.DataFrame.from_dict(step_data, orient='index')
                raw_step_df.columns = [f"{col}_raw" for col in all_nbe_dataset[1].columns]
                raw_step_df['time'] = time
                raw_step_df['node_id'] = raw_step_df.index
                raw_df_list.append(raw_step_df)
    
    if result_df_list:
        result_df = pd.concat(result_df_list, axis=0)
        
        if raw_df_list:
            raw_df = pd.concat(raw_df_list, axis=0)
            result_df = pd.merge(result_df, raw_df, on=['time', 'node_id'], how='left')
        
        # 結果の保存
        # correct_df = pd.read_feather(all_nbe_dataset[1].files[target_idx])
        # 時刻１の情報を結合する（現在は時刻１の情報に正解データを使っているので誤差はない）
        time1_correct_df = correct_df[correct_df['time']==1]
        result_df = pd.concat([time1_correct_df[['time','node_id']+list(all_nbe_dataset[1].columns)], result_df], axis=0)
        result_df[['x','y','z']] = 0.0
        result_df.loc[result_df['time']==1, ['x','y','z']] = time1_correct_df[['x','y','z']].values

        # node_id ごとにグループ化して累積和を計算
        result_df['cum_dx'] = result_df.groupby('node_id')['dx'].cumsum()
        result_df['cum_dy'] = result_df.groupby('node_id')['dy'].cumsum()
        result_df['cum_dz'] = result_df.groupby('node_id')['dz'].cumsum()

        # 2. 初期座標の取得と適用
        # 時刻 1 の 'x', 'y', 'z' の値を、node_id ごとに全行に適用できるように抽出
        initial_x = result_df.groupby('node_id')['x'].transform('first')
        initial_y = result_df.groupby('node_id')['y'].transform('first')
        initial_z = result_df.groupby('node_id')['z'].transform('first')

        # 3. 最終座標の更新
        # 初期座標 + 累積変位
        result_df['x'] = initial_x + result_df['cum_dx']
        result_df['y'] = initial_y + result_df['cum_dy']
        result_df['z'] = initial_z + result_df['cum_dz']

        # 不要な列の削除
        result_df = result_df.drop(columns=['cum_dx', 'cum_dy', 'cum_dz'])
        result_df = result_df.reset_index(drop=True)

        # x,y,z,Sxx,Syy,Szz,Sxy,Syz,Szxの誤差の計算
        merged_df = pd.merge(result_df, correct_df[['time', 'node_id', 'x', 'y', 'z', 'dx', 'dy', 'dz', 'Sxx', 'Syy', 'Szz', 'Sxy', 'Syz', 'Szx']], 
                             on=['time', 'node_id'], suffixes=('', '_correct'))
        
        for col in ['x', 'y', 'z', 'dx', 'dy', 'dz', 'Sxx', 'Syy', 'Szz', 'Sxy', 'Syz', 'Szx']:
            merged_df[f'{col}_error'] = abs(merged_df[col] - merged_df[f'{col}_correct'])

        raw_cols = [f"{col}_raw" for col in all_nbe_dataset[1].columns]
        cols_to_keep = ['time', 'node_id'] + \
                       ['x', 'y', 'z', 'dx', 'dy', 'dz', 'Sxx', 'Syy', 'Szz', 'Sxy', 'Syz', 'Szx'] + \
                       [f'{col}_error' for col in ['x', 'y', 'z', 'dx', 'dy', 'dz', 'Sxx', 'Syy', 'Szz', 'Sxy', 'Syz', 'Szx']] + \
                       raw_cols
        
        final_df = merged_df[cols_to_keep]

        # 結果の保存
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_csv = output_dir / f"{target_file_stem}_results.csv"
        final_df.to_csv(output_csv, index=False)
        print(f"Saved results to {output_csv}")
    else:
        print(f"No results generated for {target_file_stem}.")

if __name__ == "__main__":
    # コマンドライン引数の解析
    import argparse

    parser = argparse.ArgumentParser(description="NbePeepholeLSTMのモデル可視化プログラムです。")
    parser.add_argument("weight_dir", help="Directory to search for best.pth")
    parser.add_argument("dataset_dir", help="テスト用データの保存ディレクトリを指定してください。")
    parser.add_argument("--all_node_id", type=int, default=140)
    parser.add_argument("--target_idx", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--device", default=None, help="torch device string, e.g. cpu or cuda:0")
    parser.add_argument("--global_normalize", action="store_true", help="Whether to apply global normalization to the dataset")
    parser.add_argument("--output_dir", default="/workspace/results", help="Directory to save evaluation results")
    parser.add_argument("--alpha", type=float, default=8.0, help="Oka normalization alpha parameter")
    parser.add_argument("--files_num", type=str, default=None, help="Number of files to process from index 0 or 'all'")
    parser.add_argument("--model_name", type=str, default="best.pth", help="Name of the model file to load (default: best_pretrain.pth)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for file selection")
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

    # Nbeパラメータの取得
    print("Nbeパラメータの取得")
    all_nbe_params, all_neighbor_node = all_node_nbe_params(all_nbe_dataset, 
                                                            args.hidden_size, 
                                                            args.num_layers, 
                                                            args.dropout)
    
    # NBEモデルの取得
    print("NBEモデルの取得")
    all_nbe = all_nbe_weights_load(all_nbe_params, args.weight_dir, map_location=device, model_name=args.model_name)

    # Determine indices to process
    if args.files_num is not None:
        total_files = len(all_nbe_dataset[1].files)
        if args.files_num.lower() == 'all':
            target_indices = range(total_files)
        else:
            try:
                files_num = int(args.files_num)
                np.random.seed(args.seed)
                if files_num > total_files:
                    target_indices = range(total_files)
                else:
                    target_indices = np.random.choice(total_files, files_num, replace=False)
                    target_indices.sort()
            except ValueError:
                print(f"Error: files_num must be an integer or 'all'. Got {args.files_num}")
                exit(1)
        print(f"Selected indices: {target_indices}")
    else:
        target_indices = [args.target_idx]

    for target_idx in target_indices:
        run_single_evaluation(target_idx, all_nbe_dataset, all_nbe_params, all_neighbor_node, all_nbe, device, args.output_dir, args.all_node_id)
    