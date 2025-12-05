from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class NbeGNN(nn.Module):
    """
    A Graph Neural Network (GNN) for learning on graphs with N nodes and K undirected edges.
    """
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        output_size: int, 
        num_layers: int = 2, 
        dropout: float = 0.5,
        gnn_type: str = 'GCN',
        finetune: bool = False
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        self.finetune = finetune

        self.convs = nn.ModuleList()
        
        # Factory for GNN layers
        if gnn_type == 'GCN':
            ConvLayer = GCNConv
        elif gnn_type == 'GAT':
            ConvLayer = GATConv
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")

        # Input layer
        if num_layers > 1:
            self.convs.append(ConvLayer(input_size, hidden_size))
            
            # Hidden layers
            for _ in range(num_layers - 2):
                self.convs.append(ConvLayer(hidden_size, hidden_size))
                
            # Output layer
            self.convs.append(ConvLayer(hidden_size, output_size))
        else:
            # Single layer
            self.convs.append(ConvLayer(input_size, output_size))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features of shape (num_nodes, input_size)
               OR (seq_len, num_nodes, input_size)
            edge_index: Graph connectivity of shape (2, num_edges)
        
        Returns:
            out: Node features of shape (num_nodes, output_size)
                 OR (seq_len, num_nodes, output_size)
        """
        if x.dim() == 3:
            # Case: (seq_len, num_nodes, input_size)
            seq_len, num_nodes, _ = x.size()
            outputs = []
            for t in range(seq_len):
                # Process each time step
                if self.finetune and t != 0:
                    # x[t] をコピーして新しいテンソルを作成
                    current_input = x[t].clone()
                    # 予測値で上書き
                    current_input[:, :self.output_size] = out_t
                    # 推論
                    out_t = self._forward_single(current_input, edge_index)
                else:
                    out_t = self._forward_single(x[t], edge_index)
                outputs.append(out_t)
            return torch.stack(outputs, dim=0)
        else:
            # Case: (num_nodes, input_size)
            return self._forward_single(x, edge_index)

    def _forward_single(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Last layer
        x = self.convs[-1](x, edge_index)
        
        # Apply sigmoid and scale to [0.1, 0.9]
        x = torch.sigmoid(x) * 0.8 + 0.1
        return x

    def set_finetune(self, finetune: bool) -> None:
        self.finetune = finetune

if __name__ == "__main__":
    # Test code
    N = 100  # Nodes
    K = 200  # Edges
    input_dim = 16
    hidden_dim = 32
    output_dim = 8
    
    print("Testing GCN...")
    model = NbeGNN(input_dim, hidden_dim, output_dim, num_layers=3, gnn_type='GCN')
    x = torch.randn(N, input_dim)
    edge_index = torch.randint(0, N, (2, K))
    out = model(x, edge_index)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")

    print("\nTesting GAT...")
    model_gat = NbeGNN(input_dim, hidden_dim, output_dim, num_layers=2, gnn_type='GAT')
    out_gat = model_gat(x, edge_index)
    print(f"Output shape (GAT): {out_gat.shape}")
    print(f"Output range (GAT): [{out_gat.min().item():.4f}, {out_gat.max().item():.4f}]")
