from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn
import hydra
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from omegaconf import MISSING
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

from src.Dataloader.nbeAttentionDataset import NbeAttentionDataset
from src.Networks.nbe_attention_mlp import NbeAttentionMLP

from tqdm import tqdm

@dataclass
class TrainConfig:
    # data
    train_dir: str = MISSING
    val_dir: str = MISSING
    node_id: int = 10
    # optional node range: if both provided, train for each node in [node_min, node_max]
    node_min: Optional[int] = None
    node_max: Optional[int] = None
    preload: bool = False
    glob: str = "*.feather"
    alpha: float = 8.0
    global_normalize: bool = True
    
    # dataloader
    batch_size: int = 32
    num_workers: int = 4

    # model
    # nbe_fixed is now inferred from dataset
    feature_dim: int = 32
    hidden_dim: int = 128
    latent_dim: int = 64
    num_heads: int = 4
    num_layers_mlp: int = 3
    output_dim: Optional[int] = None # If None, inferred from dataset
    lambda_ae: float = 1.0

    # optimization
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 300
    
    # lr/early stopping
    lr_patience: int = 5
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    max_lr_reductions: int = 3
    early_stop_patience: int = 10
    lr_decay_start_epoch: int = 30
    lr_decay_step_size: int = 10

    # misc
    save_dir: str = "outputs/attention_mlp_train"
    seed: int = 42

def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass

def collate_fn(batch):
    # batch: list of dicts {inputs: {u, sigma, tau}, targets: {u, sigma, tau, merge}}
    
    u_list = [b["inputs"]["u"] for b in batch]
    sigma_list = [b["inputs"]["sigma"] for b in batch]
    tau_list = [b["inputs"]["tau"] for b in batch]
    
    # Stack inputs (Batch, Time, N, 3) 
    u = torch.stack(u_list, dim=0)
    sigma = torch.stack(sigma_list, dim=0)
    tau = torch.stack(tau_list, dim=0)
    
    # Targets (Batch, Time, Dim) (already merged in dataset: [u, sigma, tau])
    targets_merge = torch.stack([b["targets"]["merge"] for b in batch], dim=0)

    return u, sigma, tau, targets_merge

def run_training(cfg: TrainConfig) -> None:
    seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # determine node IDs to train
    from omegaconf import OmegaConf

    def _cfg_get(key: str):
        try:
            if OmegaConf.is_config(cfg):
                return OmegaConf.select(cfg, key)
        except Exception:
            pass
        # fall back to dict/getattr
        if isinstance(cfg, dict):
            return cfg.get(key)
        return getattr(cfg, key, None)

    node_min = _cfg_get('node_min')
    node_max = _cfg_get('node_max')
    node_id_cfg = _cfg_get('node_id')
    global_normalize = _cfg_get('global_normalize')

    if node_min is not None and node_max is not None:
        node_ids = list(range(int(node_min), int(node_max) + 1))
    else:
        node_ids = [int(node_id_cfg)]

    for node_id in node_ids:
        print(f"Starting training for node_id={node_id}")

        # datasets (per-node)
        train_ds = NbeAttentionDataset(
            data_dir=cfg.train_dir,
            node_id=node_id,
            preload=cfg.preload,
            glob=cfg.glob,
            alpha=cfg.alpha,
            global_normalize=global_normalize
        )
        val_ds = NbeAttentionDataset(
            data_dir=cfg.val_dir,
            node_id=node_id,
            preload=cfg.preload,
            glob=cfg.glob,
            alpha=cfg.alpha,
            global_normalize=global_normalize
        )

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, persistent_workers=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, persistent_workers=True, collate_fn=collate_fn)

        # Confirm dimensions from dataset
        # Inputs: (Time, N, 3) -> We use the last dim as feature dim
        # But wait, NbeAttentionDataset (after user's structure fix) returns:
        # u: (Time, N, 3), sigma: (Time, N, 3), tau: (Time, N, 3) 
        # (Assuming the user applied the transpose fix or is about to)
        # NbeAttentionMLP expects: (Batch, N, Dim) per time step? 
        # Actually Model Forward takes (Batch, N, Dim).
        # We need to process sequence carefully. Usually we process step-by-step or flatten time if it's not maintaining state.
        # But this is MLP based, so it treats Time*Batch as Batch dimension effectively, or just processes each step.
        # Target is (Time, 9) (central node only)
        # The Model output is (Batch, N, OutputDim).
        # However, dataset target is central node only at t+1. 
        # So we should valid output for the central node (index 0).
        
        sample_item = train_ds[0]
        # Check input dimensions
        # u: (Time, N, 3)
        u_sample = sample_item["inputs"]["u"]
        sigma_sample = sample_item["inputs"]["sigma"]

        dim_u = u_sample.shape[-1]
        dim_sigma = sigma_sample.shape[-1]
        dim_tau = sample_item["inputs"]["tau"].shape[-1]
        
        # Extract node counts for dynamic MLP sizing
        num_nodes_u = u_sample.shape[1]
        num_nodes_sigma = sigma_sample.shape[1]

        # Target dim (u+sigma+tau for central node) = 9 usually
        # But if center node is fixed, u is not predicted (only sigma+tau), so dim=6
        
        # We can detect this from the dataset's target shape or the `merge` tensor
        target_dim = sample_item["targets"]["merge"].shape[-1]
        
        # Override output_dim with target_dim unless explicitly forced
        output_dim = target_dim 
        if cfg.output_dim is not None:
             print(f"Warning: Overriding inferred output_dim {target_dim} with config value {cfg.output_dim}")
             output_dim = cfg.output_dim

        print(f"Training Node {node_id}: is_fixed={train_ds.is_center_node_fixed}, Output Dim={output_dim}, N_u={num_nodes_u}, N_sigma={num_nodes_sigma}")
        
        model = NbeAttentionMLP(
            nbe_fixed=train_ds.is_center_node_fixed,
            dim_u=dim_u,
            dim_sigma=dim_sigma,
            dim_tau=dim_tau,
            feature_dim=cfg.feature_dim,
            hidden_dim=cfg.hidden_dim,
            latent_dim=cfg.latent_dim,
            output_dim=output_dim,
            num_heads=cfg.num_heads,
            num_layers_mlp=cfg.num_layers_mlp,
            num_nodes_u=num_nodes_u,
            num_nodes_sigma=num_nodes_sigma
        )
        model.to(device)

        save_root = Path(cfg.save_dir) / f"{node_id}"
        save_root.mkdir(parents=True, exist_ok=True)
        run_dir = save_root
        writer = SummaryWriter(log_dir=str(run_dir / "tb"))

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        criterion = nn.MSELoss()

        best_val = float("inf")
        best_ckpt_path = run_dir / "best.pth"

        for epoch in range(1, cfg.epochs + 1):
            # LR Decay
            if epoch >= cfg.lr_decay_start_epoch and (epoch - cfg.lr_decay_start_epoch) % cfg.lr_decay_step_size == 0:
                 for param_group in optimizer.param_groups:
                    param_group['lr'] *= cfg.lr_factor

            model.train()
            train_losses = []
            
            for u_b, sigma_b, tau_b, targets_b in train_loader:
                # Inputs: (Batch, Time, N, Dim)
                # Targets: (Batch, Time, OutDim)
                
                # We merge Batch and Time dimensions for processing: (Batch*Time, N, Dim)
                b, t, n_u, d_u = u_b.shape
                _, _, n_s, d_s = sigma_b.shape
                _, _, n_t, d_t = tau_b.shape
                _, _, d_out = targets_b.shape

                u_in = u_b.view(b*t, n_u, d_u).to(device)
                sigma_in = sigma_b.view(b*t, n_s, d_s).to(device)
                tau_in = tau_b.view(b*t, n_t, d_t).to(device)
                
                y_true = targets_b.view(b*t, d_out).to(device)

                optimizer.zero_grad()
                
                # Forward
                # Output: (Batch*Time, N, OutputDim)
                # Note: Model returns output for ALL nodes as (Batch, N, OutDim).
                # But we only verify against central node (Index 0).
                
                pred, ae_loss = model(u=u_in, sigma=sigma_in, tau=tau_in)
                
                # Extract central node prediction
                pred_central = pred

                loss_pred = criterion(pred_central, y_true)
                loss = loss_pred + cfg.lambda_ae * ae_loss
                
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())

            train_loss = float(np.mean(train_losses)) if train_losses else 0.0

            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for u_b, sigma_b, tau_b, targets_b in val_loader:
                    b, t, n_u, d_u = u_b.shape
                    _, _, n_s, d_s = sigma_b.shape
                    _, _, n_t, d_t = tau_b.shape
                    
                    # Flatten batch and time
                    u_in = u_b.view(b*t, n_u, d_u).to(device)
                    sigma_in = sigma_b.view(b*t, n_s, d_s).to(device)
                    tau_in = tau_b.view(b*t, n_t, d_t).to(device)
                    
                    y_true = targets_b.view(b*t, -1).to(device)

                    pred, ae_loss = model(u=u_in, sigma=sigma_in, tau=tau_in)
                        
                    pred_central = pred
                    loss_pred = criterion(pred_central, y_true)
                    loss = loss_pred + cfg.lambda_ae * ae_loss
                    
                    val_losses.append(loss.item())

            val_loss = float(np.mean(val_losses)) if val_losses else 0.0

            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val,
                }, best_ckpt_path)

            print(f"node={node_id} Epoch {epoch}/{cfg.epochs}  train={train_loss:.6f}  val={val_loss:.6f}  lr={optimizer.param_groups[0]['lr']:.6e}")
        
        tqdm.write(f"Training finished for node {node_id}. Best val: {best_val:.6f}. Checkpoint saved to {best_ckpt_path}")
        writer.flush()
        writer.close()

@hydra.main(version_base=None, config_path="../../config/NBE", config_name="attention_mlp")
def main(cfg: TrainConfig) -> None:
    run_training(cfg)

if __name__ == "__main__":
    main()
