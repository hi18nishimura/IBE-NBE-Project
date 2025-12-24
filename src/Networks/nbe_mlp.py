import torch
import torch.nn as nn
import torch.optim as optim

class NbeMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128,num_hidden_layers=9):
        super().__init__()
        
        # 活性化関数: Leaky ReLU (勾配消失対策)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        
        # --- 9層の隠れ層を定義 ---
        
        # 1層目: 入力 -> 隠れ層
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), # Batch Normalization (勾配の安定化)
            self.activation
        ]
        
        # 2層目から9層目: 隠れ層 -> 隠れ層 (繰り返し処理)
        for _ in range(num_hidden_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation
            ])

        # 隠れ層全体をSequentialにまとめる
        self.hidden_stack = nn.Sequential(*layers)
            
        # --- 10層目: 出力層を定義 ---
        # 線形変換
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        # 最終活性化関数: Sigmoid
        self.sigmoid = nn.Sigmoid()

        # 重み初期化 (He Initializationを明示的に適用)
        self._initialize_weights()

    def _initialize_weights(self):
        """Heの初期化とBatch Normの標準的な初期化を適用"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming Uniform (He Initialization)を適用
                nn.init.kaiming_uniform_(m.weight, a=0.01, nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm1d):
                # BNの重み(gamma)は1、バイアス(beta)は0に初期化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x, finetune: bool = False):
        # 9層の隠れ層の順伝播
        x = self.hidden_stack(x)
        
        # 10層目（出力層）の線形変換
        x = self.fc_out(x)
        
        # 1. Sigmoidを適用し、出力を (0, 1) に制限
        y_sig = self.sigmoid(x)
        
        # 2. スケーリングを適用し、出力を [0.1, 0.9] に制限
        # 変換式: 0.1 + (0.9 - 0.1) * y_sig
        y_out = 0.1 + 0.8 * y_sig
        
        return y_out
    
class NbeMLPFinetune(nn.Module):
    def ___init__(self, base_model_list: list[NbeMLP], connect_model: list[list[int]]):
        super().__init__()

        self.base_models = nn.ModuleList()
        for model in base_model_list:
            self.base_models.append(model)
        # 各ネットワークの接続情報
        self.connect_model = connect_model

    def forward(self, x_list: list[torch.Tensor]) -> torch.Tensor:
        # 各節点で2時刻目の出力する
        outputs = []
        for i, model in enumerate(self.base_models):
            out = model(x_list[i], finetune=True)
            outputs.append(out)
        
        # 予測を平均化して最終出力を得る
        final_output = torch.mean(torch.stack(outputs, dim=0), dim=0)
        return final_output
