# 各節点のNBEの出力を入力できる形状にする関数
# 節点数、各節点の隣接節点の情報を引数に受け取る

import torch

def convert_nbe_output_to_input(nbe_outputs_dict,neighborhood_dict):
    inputs_dict = {}
    for key,value in nbe_outputs_dict.items():
        input_list = []
        for node_id in neighborhood_dict[key]:
            input_list.append(nbe_outputs_dict[node_id])
        inputs_dict[key] = torch.stack(input_list,dim=1)
    return inputs_dict