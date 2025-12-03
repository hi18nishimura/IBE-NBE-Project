from src.Dataloader.nbeDataset import NbeDataset
from src.Evaluate.nbe_peephole_lstm_load_weights import load_model_from_dir
from src.Evaluate.nbe_output_parse import convert_nbe_output_to_input
from src.Evaluate.nbe_normalize_inverse import denormalize_nbe_outputs
from src.Evaluate.nbe_dataset_time1_data import get_nbe_dataset_time1_data
from src.Networks.nbe_peephole_lstm import NbePeepholeLSTM
import torch
from tqdm import tqdm
import pandas as pd
# NbeDatasetの初期化
def all_node_nbe_dataset_init(data_dir, max_node, global_normalize: bool) -> dict[int, NbeDataset]:
    node_datasets = {}
    for node_id in tqdm(range(1, max_node+1)):
        node_datasets[node_id] = NbeDataset(data_dir=data_dir, node_id=node_id, global_normalize=global_normalize)
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
def all_nbe_weights_load(node_params: dict[int, dict[str, int]], weight_dir: str, map_location: torch.device | str) -> dict[int, NbePeepholeLSTM]:
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
        )
        nbe_models[node_id] = model
    return nbe_models

if __name__ == "__main__":
    # コマンドライン引数の解析
    import argparse

    parser = argparse.ArgumentParser(description="NbePeepholeLSTMのモデル可視化プログラムです。")
    parser.add_argument("weight_dir", help="Directory to search for best.pth")
    parser.add_argument("dataset_dir", help="テスト用データの保存ディレクトリを指定してください。")
    parser.add_argument("--all_node_id", type=int, default=140)
    parser.add_argument("--target_idx", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--device", default=None, help="torch device string, e.g. cpu or cuda:0")
    parser.add_argument("--global_normalize", action="store_true", help="Whether to apply global normalization to the dataset")
    args = parser.parse_args()

    # NbeDatasetの読み込み
    print("NbeDatasetの初期化")
    all_nbe_dataset = all_node_nbe_dataset_init(args.dataset_dir,
                                                args.all_node_id, 
                                                global_normalize=args.global_normalize)
    correct_df = pd.read_feather(all_nbe_dataset[1].files[0])
    # Nbeパラメータの取得
    print("Nbeパラメータの取得")
    all_nbe_params, all_neighbor_node = all_node_nbe_params(all_nbe_dataset, 
                                                            args.hidden_size, 
                                                            args.num_layers, 
                                                            args.dropout)
    
    # NBEモデルの取得
    print("NBEモデルの取得")
    all_nbe = all_nbe_weights_load(all_nbe_params, args.weight_dir, map_location=args.device)

    # 時刻1→2の推定
    all_outputs_list = []
    # 時刻1の情報を取得する
    print("時刻1の情報を取得する")
    all_next_inputs,all_targets = get_nbe_dataset_time1_data(all_nbe_dataset, args.all_node_id, args.target_idx)
    all_lstm_gates = {}
    all_nbe_outputs = {}
    for key,values in tqdm(all_next_inputs.items(),desc="時刻2の推定"):
        inputs = values.unsqueeze(0)  # バッチ次元の追加
        with torch.no_grad():
            inputs = inputs.to(next(all_nbe[key].parameters()).device)
            outputs, (h,c) = all_nbe[key].predict_next(inputs)
            all_lstm_gates[key] = (h, c)
            all_nbe_outputs[key] = outputs.squeeze(0)  # バッチ次元の削除
    
    # 全時刻の情報を保存する
    all_outputs_list.append(all_nbe_outputs.copy())
    # NBEの出力を次の出力にする
    all_next_inputs = convert_nbe_output_to_input(all_nbe_outputs, all_neighbor_node)

    # # 時刻3~20の推定
    for time in tqdm(range(3,21),desc="時刻3~20の推定"):
        for key,values in all_next_inputs.items():
            inputs = values.unsqueeze(0)  # バッチ次元の追加
            with torch.no_grad():
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
    for time in range(2,21):
        time_df = pd.DataFrame.from_dict(all_denormalized_outputs[time], orient='index')
        time_df.columns = all_nbe_dataset[1].columns
        time_df['time'] = time
        time_df['node_id'] = time_df.index
        result_df_list.append(time_df)
    result_df = pd.concat(result_df_list, axis=0)
    
    # 結果の保存
    correct_df = pd.read_feather(all_nbe_dataset[1].files[args.target_idx])
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
    result_df['x_error'] = abs(result_df['x'] - correct_df['x'])
    result_df['y_error'] = abs(result_df['y'] - correct_df['y'])
    result_df['z_error'] = abs(result_df['z'] - correct_df['z'])
    result_df['Sxx_error'] = abs(result_df['Sxx'] - correct_df['Sxx'])
    result_df['Syy_error'] = abs(result_df['Syy'] - correct_df['Syy'])
    result_df['Szz_error'] = abs(result_df['Szz'] - correct_df['Szz'])
    result_df['Sxy_error'] = abs(result_df['Sxy'] - correct_df['Sxy'])
    result_df['Syz_error'] = abs(result_df['Syz'] - correct_df['Syz'])
    result_df['Szx_error'] = abs(result_df['Szx'] - correct_df['Szx'])

    # 結果の保存
    result_df.to_csv(f"nbe_peephole_lstm_evaluation_results.csv", index=False)
    