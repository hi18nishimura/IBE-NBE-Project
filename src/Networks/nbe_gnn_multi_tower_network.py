import torch
import torch.nn as nn
from typing import Dict, Any
from src.Networks.nbe_gnn import NbeGNN

class MultiTowerGNN(nn.Module):
    """
    複数のNbeGNNを統合し、異なる入力・構造を並列に処理するモデル
    """
    def __init__(self, configs: Dict[str, Dict[str, Any]]):
        super().__init__()
        self.towers = nn.ModuleDict()
        
        # 各ネットワークを辞書の設定に基づいて初期化
        for name, cfg in configs.items():
            self.towers[name] = NbeGNN(
                input_size=cfg['input_size'],
                hidden_size=cfg['hidden_size'],
                output_size=cfg['output_size'],
                num_layers=cfg.get('num_layers', 2),
                dropout=cfg.get('dropout', 0.5),
                gnn_type=cfg.get('gnn_type', 'GAT'),
                return_only_central=cfg.get('return_only_central', False),
                multi_fully_layer=cfg.get('multi_fully_layer', 5)
            )

    def forward(self, inputs: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        inputs: {
            'net_a': {'x': tensor, 'edge_index': tensor},
            'net_b': ...
        }
        """
        results = {}
        for name, net in self.towers.items():
            # 各ネットワークに対応するデータを取得して推論
            data = inputs[name]
            results[name] = net(
                x=data['x'], 
                edge_index=data['edge_index']
            )
        return results
    