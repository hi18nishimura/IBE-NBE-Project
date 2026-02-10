import torch
import torch.nn as nn
import torch.nn.functional as F

class NbeAttentionMLP(nn.Module):
    def __init__(self, 
                 nbe_fixed=False,
                 dim_u=3, 
                 dim_sigma=3, 
                 dim_tau=3, 
                 feature_dim=32, 
                 hidden_dim=128, 
                 latent_dim=32,
                 output_dim=9, 
                 num_heads=4, 
                 num_layers_mlp=3,
                 num_nodes_u=None,
                 num_nodes_sigma=None):
        """
        提供されたアーキテクチャ図に基づくNbeAttentionMLPの実装。
        Cross Attentionの特徴量をフラット化(Batch, N*k)してMLPに入力する形式に変更。
        さらに、Parallel MLPsをAutoEncoderに変更し、その潜在表現をFinal MLPに入力する。

        Args:
            nbe_fixed (bool): 変位(u)を使用するかどうかのフラグ。
            dim_u (int or None): 変位入力(u)の次元。Noneまたは0の場合、uに関連する処理はスキップされます。
            dim_sigma (int): 垂直応力入力(sigma)の次元。
            dim_tau (int): せん断応力入力(tau)の次元。
            feature_dim (int): 射影後およびAttentionに使用される特徴量の次元(k)。
            hidden_dim (int): MLPの隠れ層の次元。
            latent_dim (int): AutoEncoderの潜在層の次元。Final MLPの入力に使用される。
            output_dim (int): 最終出力の次元。
            num_heads (int): MultiheadAttentionのヘッド数。
            num_layers_mlp (int): 中間および最終MLPの層数。
            num_nodes_u (int): 変位(u)のノード数。MLP入力次元決定に使用。
            num_nodes_sigma (int): 応力(sigma/tau)のノード数。MLP入力次元決定に使用。
        """
        super().__init__()
        
        self.nbe_fixed = nbe_fixed
        # nbe_fixed=True (固定節点学習) の場合は output_dim を 6 に強制
        if self.nbe_fixed:
            output_dim = 6
            
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_nodes_u = num_nodes_u
        self.num_nodes_sigma = num_nodes_sigma
        
        # 1. 入力射影: (N, dim) -> (N, k) に変換
        self.proj_u = nn.Linear(dim_u, feature_dim)
        self.proj_sigma = nn.Linear(dim_sigma, feature_dim)
        self.proj_tau = nn.Linear(dim_tau, feature_dim)
        
        # 2. Attention機構
        # 図に従った6つのペアすべてのAttentionを用意
        self.attn_u_sigma = nn.MultiheadAttention(feature_dim, num_heads)
        self.attn_u_tau = nn.MultiheadAttention(feature_dim, num_heads)
        self.attn_sigma_u = nn.MultiheadAttention(feature_dim, num_heads)
        self.attn_tau_u = nn.MultiheadAttention(feature_dim, num_heads)
            
        self.attn_sigma_tau = nn.MultiheadAttention(feature_dim, num_heads)
        self.attn_tau_sigma = nn.MultiheadAttention(feature_dim, num_heads)
        
        # 3. 並列MLP (AutoEncoders)
        # 各Attention出力(Batch, N, k)を(Batch, N*k)にFlattenしてAEに入力
        
        # u originate features (Q=u) have N_u nodes
        mlp_input_dim_u = num_nodes_u * feature_dim
        # c_sigma_u (Q=sigma) -> N_sigma
        mlp_input_dim_sigma = num_nodes_sigma * feature_dim

        # Output of attention follows Query dimension
        self.ae_u_sigma = self._make_autoencoder(mlp_input_dim_u, latent_dim, num_layers_mlp)
        self.ae_u_tau = self._make_autoencoder(mlp_input_dim_u, latent_dim, num_layers_mlp)
        
        self.ae_sigma_u = self._make_autoencoder(mlp_input_dim_sigma, latent_dim, num_layers_mlp)
        self.ae_tau_u = self._make_autoencoder(mlp_input_dim_sigma, latent_dim, num_layers_mlp)
            
        self.ae_sigma_tau = self._make_autoencoder(mlp_input_dim_sigma, latent_dim, num_layers_mlp)
        self.ae_tau_sigma = self._make_autoencoder(mlp_input_dim_sigma, latent_dim, num_layers_mlp)
        
        # 4. 最終MLP
        # 6つの並列AEの潜在表現(latent_dim)を結合 -> (Batch, 6 * latent_dim)
        input_dim_final = 6 * latent_dim
            
        self.final_mlp = self._make_mlp(input_dim_final, hidden_dim, output_dim, num_layers_mlp)

    def _make_mlp(self, in_dim, h_dim, out_dim, num_layers):
        layers = []
        # First layer
        layers.append(nn.Linear(in_dim, h_dim))
        layers.append(nn.BatchNorm1d(h_dim)) # BatchNorm for (Batch, Dim) is standard BatchNorm1d
        layers.append(nn.Tanh())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(h_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Tanh())
            
        # Output layer of MLP
        layers.append(nn.Linear(h_dim, out_dim))
        
        return nn.Sequential(*layers)

    def _make_autoencoder(self, in_dim, latent_dim, num_layers):
        # Encoder
        encoder_layers = []
        h_dim = self.hidden_dim
        
        encoder_layers.append(nn.Linear(in_dim, h_dim))
        encoder_layers.append(nn.BatchNorm1d(h_dim))
        encoder_layers.append(nn.Tanh())
        
        for _ in range(num_layers - 2):
            encoder_layers.append(nn.Linear(h_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.Tanh())
            
        # Latent layer (no activation usually for bottleneck, or maybe LeakyReLU? sticking to linear projection for bottleneck is common or Tanh/Sigmoid if bounded. Keeping Linear)
        encoder_layers.append(nn.Linear(h_dim, latent_dim))
        encoder_layers.append(nn.BatchNorm1d(latent_dim)) 
        encoder_layers.append(nn.Tanh()) # Adding activation to latent space as implied by "hidden layer" usage for next stage

        encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_dim, h_dim))
        decoder_layers.append(nn.BatchNorm1d(h_dim))
        decoder_layers.append(nn.Tanh())
        
        for _ in range(num_layers - 2):
            decoder_layers.append(nn.Linear(h_dim, h_dim))
            decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.Tanh())
            
        decoder_layers.append(nn.Linear(h_dim, in_dim))
        # No activation for reconstruction output (assuming real values)
        decoder = nn.Sequential(*decoder_layers)
        
        return nn.ModuleDict({'encoder': encoder, 'decoder': decoder})

    def forward(self, u, sigma, tau, return_ae_loss=None):
        """
        Args:
            u: 変位テンソル (Batch, N, dim_u) または None
            sigma: 垂直応力テンソル (Batch, N, dim_sigma)
            tau: せん断応力テンソル (Batch, N, dim_tau)
            return_ae_loss (bool, optional): AutoEncoderの損失を計算するかどうか。
                                           Noneの場合はself.training (学習モード)に従う。
            
        Returns:
            out: (Batch, output_dim)
            ae_loss: scalar tensor (sum of MSE losses of all AEs) or 0.0
        """
        
        calc_ae_loss = return_ae_loss if return_ae_loss is not None else self.training

        # 入力が (Batch, N, Dim) であることを保証。もし (N, Dim) で与えられた場合は batch=0 をunsqueezeする。
        if u is not None and u.dim() == 2:
            u = u.unsqueeze(0)
            
        if sigma.dim() == 2:
            sigma = sigma.unsqueeze(0)
        if tau.dim() == 2:
            tau = tau.unsqueeze(0)
        
        # --- 1. 射影 (Projections) ---
        h_sigma = self.proj_sigma(sigma) # (Batch, N, feature_dim)
        h_tau = self.proj_tau(tau)      # (Batch, N, feature_dim)
        
        # u is expected to be present now
        h_u = self.proj_u(u)            # (Batch, N, feature_dim)
        h_u_t = h_u.permute(1, 0, 2)
        
        h_sigma_t = h_sigma.permute(1, 0, 2)
        h_tau_t = h_tau.permute(1, 0, 2)
        
        # --- 2. Attention ---
        
        # Cross Attentionの計算
        c_u_sigma, _ = self.attn_u_sigma(h_u_t, h_sigma_t, h_sigma_t)
        c_u_tau, _ = self.attn_u_tau(h_u_t, h_tau_t, h_tau_t)
        c_sigma_u, _ = self.attn_sigma_u(h_sigma_t, h_u_t, h_u_t)
        c_tau_u, _ = self.attn_tau_u(h_tau_t, h_u_t, h_u_t)
        
        c_u_sigma = c_u_sigma.permute(1, 0, 2)
        c_u_tau = c_u_tau.permute(1, 0, 2)
        c_sigma_u = c_sigma_u.permute(1, 0, 2)
        c_tau_u = c_tau_u.permute(1, 0, 2)
        
        c_sigma_tau, _ = self.attn_sigma_tau(h_sigma_t, h_tau_t, h_tau_t)
        c_tau_sigma, _ = self.attn_tau_sigma(h_tau_t, h_sigma_t, h_sigma_t)
        
        c_sigma_tau = c_sigma_tau.permute(1, 0, 2)
        c_tau_sigma = c_tau_sigma.permute(1, 0, 2)
        
        # --- 3. 並列AE (Parallel AutoEncoders) ---
        
        def run_ae(ae_module, x):
            # x: (Batch, N, Dim)
            b, n, d = x.shape
            x_flat = x.reshape(b, n * d)
            
            encoder = ae_module['encoder']
            decoder = ae_module['decoder']
            
            latent = encoder(x_flat)
            
            if calc_ae_loss:
                recon = decoder(latent)
                loss = F.mse_loss(recon, x_flat)
            else:
                loss = torch.tensor(0.0, device=x.device)
            
            return latent, loss

        ae_loss_total = 0.0

        l_u_sigma, loss_u_sigma = run_ae(self.ae_u_sigma, c_u_sigma)
        l_u_tau, loss_u_tau = run_ae(self.ae_u_tau, c_u_tau)
        l_sigma_u, loss_sigma_u = run_ae(self.ae_sigma_u, c_sigma_u)
        l_tau_u, loss_tau_u = run_ae(self.ae_tau_u, c_tau_u)
        
        ae_loss_total += (loss_u_sigma + loss_u_tau + loss_sigma_u + loss_tau_u)

        l_sigma_tau, loss_sigma_tau = run_ae(self.ae_sigma_tau, c_sigma_tau)
        l_tau_sigma, loss_tau_sigma = run_ae(self.ae_tau_sigma, c_tau_sigma)
        
        ae_loss_total += (loss_sigma_tau + loss_tau_sigma)
        
        # --- 4. 最終的な結合 (Final Combination) ---
        combined = torch.cat([
            l_u_sigma, 
            l_u_tau, 
            l_sigma_u, 
            l_sigma_tau, 
            l_tau_u, 
            l_tau_sigma
        ], dim=-1)
        
        # 最終MLP
        out = self.final_mlp(combined)
        out = torch.sigmoid(out) * 0.8 + 0.1
        
        return out, ae_loss_total
