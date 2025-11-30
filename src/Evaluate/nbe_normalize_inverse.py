# 各節点の出力を元のスケールに戻す処理
# とりあえず、全特徴量をパワー正規化によって元のスケールに戻す実装にする
# 今後は変数によっては通常のmin-max正規化に戻す等の選択肢も検討する
import torch

def oka_denormalize(x, max_value, alpha=8.0):
    deviation = x - 0.5
    s = torch.sign(deviation)
    magnitude = (torch.abs(deviation) / 0.4) ** alpha
    p_in = s * max_value * magnitude
    return p_in

def denormalize_nbe_outputs(all_outputs_list: list[dict[int, torch.Tensor]], all_nbe_params: dict[int, dict[str, int]],all_node_dataset: dict[int, any]) -> list[dict[int, torch.Tensor]]:
    denormalized_outputs_dict = {}
    zero_disp_tensor = torch.tensor([0.5, 0.5, 0.5])
    for time, time_outputs in enumerate(all_outputs_list):
        denormalized_time_outputs = {}
        for node_id, output in time_outputs.items():
            max_value_list = []
            for col in all_node_dataset[node_id].columns:
                max_value_list.append(all_node_dataset[node_id].max_map[col])
            max_values = torch.tensor(max_value_list, device=output.device)
            if output.shape[0] == 6:
                output_reshape = torch.cat([zero_disp_tensor.to(output.device),output],dim=0)
                denormalized_output = oka_denormalize(output_reshape, max_values)  # パワー正規化の逆変換
            else:
                denormalized_output = oka_denormalize(output, max_values)  # パワー正規化の逆変換
            denormalized_time_outputs[node_id] = denormalized_output.detach().cpu().numpy()
        denormalized_outputs_dict[time+2]=denormalized_time_outputs
    return denormalized_outputs_dict