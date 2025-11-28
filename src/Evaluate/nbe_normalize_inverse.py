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

def denormalize_nbe_outputs(all_outputs_list: list[dict[int, torch.Tensor]], all_nbe_params: dict[int, dict[str, int]]):
    denormalized_outputs_list = []
    for time_outputs in all_outputs_list:
        denormalized_time_outputs = {}
        for node_id, output in time_outputs.items():
            max_value_list = []
            for max_value in all_nbe_params[node_id]["max_values"].values():
                max_value_list.append(max_value)
            max_values = torch.tensor(max_value_list, device=output.device)
            denormalized_output = oka_denormalize(output, max_values)  # パワー正規化の逆変換
            denormalized_time_outputs[node_id] = denormalized_output.detach().cpu().numpy()
        denormalized_outputs_list.append(denormalized_time_outputs)
    return denormalized_outputs_list