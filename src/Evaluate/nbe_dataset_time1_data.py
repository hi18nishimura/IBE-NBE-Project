# NBEDatasetから全節点の情報を取得する
# 全節点の情報から、時刻１の情報だけを取り出す
# 取り出した情報からnode_idをキーとした辞書を作成する

def get_nbe_dataset_time1_data(nbe_dataset_dict, max_node, idx):
    node_time1_data = {}
    all_targets = {}
    for node_id in max_node:
        data = nbe_dataset_dict[node_id].__getitem__(idx)
        inputs,targets = data["inputs"], data["targets"]
        time1_data = inputs[0,:]

        # 時刻1のデータを抽出

        node_time1_data[node_id] = time1_data
        all_targets[node_id] = targets
    return node_time1_data, all_targets