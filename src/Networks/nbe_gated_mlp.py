import torch
import torch.nn as nn

class SpatialGatingUnit(nn.Module):
    """論文の肝となるSGU (Spatial Gating Unit)"""
    def __init__(self, dim, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(dim // 2)
        # 空間方向の情報の混ざり合いを決定する重み W
        self.proj = nn.Linear(seq_len, seq_len)
       
        # 論文の初期化手法: Wをほぼ0、biasを1にして、最初は「何もしない」状態にする
        nn.init.zeros_(self.proj.weight)
        nn.init.ones_(self.proj.bias)

    def forward(self, x):
        # x: [Batch, Seq_Len, Dim]
        res, gate = x.chunk(2, dim=-1) # チャネル方向に分割
       
        gate = self.norm(gate)
        gate = gate.transpose(-1, -2)   # [Batch, Dim/2, Seq_Len]
        gate = self.proj(gate)         # 空間（シーケンス）方向の重み付け
        gate = gate.transpose(-1, -2)   # [Batch, Seq_Len, Dim/2]
       
        return res * gate

class gMLPBlock(nn.Module):
    """gMLPの基本1ブロック"""
    def __init__(self, d_model, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.Tanh()
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)
        self.channel_proj2 = nn.Linear(d_ffn // 2, d_model)

    def forward(self, x):
        # 残差接続 (Residual Connection)
        shortcut = x
        x = self.norm(x)
        x = self.channel_proj1(x)
        x = self.activation(x)
        x = self.sgu(x)
        x = self.channel_proj2(x)
        return x + shortcut

class NbeGatedMLP(nn.Module):
    """論文の構成に基づいた回帰用gMLP"""
    def __init__(self, input_dim, output_dim,d_model=64, d_ffn=128, seq_len=1, num_layers=6):
        super().__init__()
        # 入力をモデル次元に合わせる
        self.embed = nn.Linear(input_dim, d_model)
       
        # ブロックを重ねる
        self.blocks = nn.ModuleList([
            gMLPBlock(d_model, d_ffn, seq_len) for _ in range(num_layers)
        ])
       
        # 回帰ヘッド
        self.norm = nn.LayerNorm(d_model)
        #self.head = nn.Linear(d_model, output_dim)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.Tanh(),
            nn.Linear(d_model//2, output_dim)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [Batch, Input_Dim]
        # 回帰タスクの場合、seq_len=1として扱う
        if x.dim() == 2:
            x = x.unsqueeze(1) # [Batch, 1, Input_Dim]
           
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
           
        x = self.norm(x)
        x = x.mean(dim=1) # 空間方向の平均（Pooling）
        x = 0.1 + 0.8 * self.sigmoid(self.head(x))  # Scale sigmoid to 0.1 - 0.9 range
        return x