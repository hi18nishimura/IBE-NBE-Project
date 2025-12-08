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
        finetune: bool = False,
        return_only_central: bool = False
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        self.finetune = finetune
        self.return_only_central = return_only_central

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
                
            # Last GNN layer (outputs hidden_size)
            self.convs.append(ConvLayer(hidden_size, hidden_size))
        else:
            # Single layer (outputs hidden_size)
            self.convs.append(ConvLayer(input_size, hidden_size))
            
        # Readout layer (Linear)
        self.readout = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Case 1: (Batch, Seq, Nodes, Input) -> 4 dims
        if x.dim() == 4:
            B, S, N, F = x.size()
            # Reshape to (Seq, Batch*Nodes, Input) for sequential processing
            # Note: edge_index must be batched accordingly (Batch*Nodes)
            x_reshaped = x.permute(1, 0, 2, 3).reshape(S, B*N, F)
            
            out = self._forward_seq(x_reshaped, edge_index, num_nodes=N) # (S, B, Out) if central else (S, B*N, Out)
            
            if self.return_only_central:
                # out: (S, B, Out) -> (B, S, Out)
                return out.permute(1, 0, 2)
            else:
                # out: (S, B*N, Out) -> (S, B, N, Out) -> (B, S, N, Out)
                return out.view(S, B, N, -1).permute(1, 0, 2, 3)

        # Case 2: (Seq, Nodes, Input) -> 3 dims
        elif x.dim() == 3:
            S, N, F = x.size()
            out = self._forward_seq(x, edge_index, num_nodes=N) # (S, 1, Out) if central else (S, N, Out)
            
            if self.return_only_central:
                # out: (S, 1, Out) -> (S, Out)
                return out.squeeze(1)
            return out

        # Case 3: (Nodes, Input) -> 2 dims
        else:
            N, F = x.size()
            out = self._forward_single(x, edge_index, num_nodes=N)
            # out: (1, Out) if central else (N, Out)
            if self.return_only_central:
                return out.squeeze(0)
            return out

    def _forward_seq(self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Process a sequence of graph states.
        x: (Seq, TotalNodes, Input)
        """
        seq_len, total_nodes, input_size = x.size()
        batch_size = total_nodes // num_nodes
        
        if self.finetune:
            # 再帰的な依存があるため、ループ処理が必要
            outputs = []
            out_t = None
            for t in range(seq_len):
                if t != 0:
                    current_input = x[t].clone()
                    
                    if self.return_only_central:
                        # out_t: (Batch, Output)
                        # current_input: (TotalNodes, Input) -> (Batch, NumNodes, Input)
                        # Update only central nodes (index 0)
                        current_input_view = current_input.view(batch_size, num_nodes, -1)
                        current_input_view[:, 0, :self.output_size] = out_t.detach()
                    else:
                        current_input[:, :self.output_size] = out_t.detach() # out_tをdetachして勾配計算のグラフを分離を推奨
                        
                    out_t = self._forward_single(current_input, edge_index, num_nodes)
                else:
                    out_t = self._forward_single(x[t], edge_index, num_nodes)
                outputs.append(out_t)
            return torch.stack(outputs, dim=0)
        else:
            # Case: (seq_len, num_nodes, input_size) -> (seq_len * num_nodes, input_size)
            x_reshaped = x.view(-1, input_size) 
            
            # GNN計算
            out_reshaped = self._forward_single(x_reshaped, edge_index, num_nodes)
            
            if self.return_only_central:
                # out_reshaped: (Seq * Batch, Output)
                # -> (Seq, Batch, Output)
                return out_reshaped.view(seq_len, batch_size, -1)
            else:
                # (seq_len * num_nodes, output_size) -> (seq_len, num_nodes, output_size)
                output_size = out_reshaped.size(-1)
                return out_reshaped.view(seq_len, total_nodes, output_size)

    def _forward_single(self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: Optional[int] = None) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # x: (TotalNodes, Hidden)
        
        if self.return_only_central and num_nodes is not None:
            total_nodes = x.size(0)
            batch_size = total_nodes // num_nodes
            # Extract central nodes: (Batch, NumNodes, Hidden) -> (Batch, Hidden)
            # Central node is at index 0 for each graph in the batch
            x = x.view(batch_size, num_nodes, -1)[:, 0, :]
        
        # Readout layer
        x = self.readout(x)
        
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
