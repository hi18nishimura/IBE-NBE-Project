from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import MISSING
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.Dataloader.nbeDataset import NbeDataset
from src.Networks.nbe_mlp import NbeMLP

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
    hidden_size: int = 128
    num_layers: int = 9
    dropout: float = 0.0 # Not used in NbeMLP currently but kept for config compatibility
    output_size: Optional[int] = None

    # optimization
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 300
    
    # loss weighting
    loss_weight_alpha: float = 0.0
    loss_weight_gamma: float = 2.0
    loss_weight_time_beta: float = 0.0

    # lr/early stopping
    lr_patience: int = 5
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    max_lr_reductions: int = 3
    early_stop_patience: int = 10

    # misc
    save_dir: str = "outputs/mlp_train"
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
    # batch: list of dicts {inputs: [19,F], targets: [19,T]}
    inputs = torch.stack([b["inputs"] for b in batch], dim=0)
    targets = torch.stack([b["targets"] for b in batch], dim=0)
    return inputs, targets


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
    print(f"Alpha: {cfg.alpha}, Loss Weight Alpha: {cfg.loss_weight_alpha}, Loss Weight Gamma: {cfg.loss_weight_gamma}") 
    for node_id in node_ids:
        print(f"Starting training for node_id={node_id}")

        # datasets (per-node)
        train_ds = NbeDataset(
            data_dir=cfg.train_dir,
            node_id=node_id,
            preload=cfg.preload,
            glob=cfg.glob,
            alpha=cfg.alpha,
            global_normalize=global_normalize,
        )
        val_ds = NbeDataset(
            data_dir=cfg.val_dir,
            node_id=node_id,
            preload=cfg.preload,
            glob=cfg.glob,
            alpha=cfg.alpha,
            global_normalize=global_normalize,
        )

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, persistent_workers=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, persistent_workers=True, collate_fn=collate_fn)

        input_size = train_ds.input_feature_size
        target_size = train_ds.node_feature_counts[train_ds.node_id]
        output_size = cfg.output_size or target_size

        model = NbeMLP(
            input_dim=input_size, 
            output_dim=output_size, 
            hidden_dim=cfg.hidden_size, 
            num_hidden_layers=cfg.num_layers
        )
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(cfg.lr_factor),
            patience=cfg.lr_patience,
            min_lr=float(cfg.min_lr),
        )


        criterion = nn.MSELoss()

        save_root = Path(cfg.save_dir) / f"{node_id}"
        save_root.mkdir(parents=True, exist_ok=True)
        run_dir = save_root
        writer = SummaryWriter(log_dir=str(run_dir / "tb"))

        best_val = float("inf")
        epochs_since_improve = 0
        lr_reductions = 0
        
        best_ckpt = run_dir / "best.pth"

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            train_losses = []
            for batch in train_loader:
                xb, yb = batch
                xb = xb.to(device)
                yb = yb.to(device)
                
                # Reshape for MLP: (Batch, Time, Feats) -> (Batch * Time, Feats)
                B, T, F_in = xb.shape
                _, _, F_out = yb.shape
                
                xb_flat = xb.view(-1, F_in)
                yb_flat = yb.view(-1, F_out)
                
                optimizer.zero_grad()
                outputs = model(xb_flat)
                
                loss_weight_alpha = getattr(cfg, "loss_weight_alpha", 0.0)
                loss_weight_gamma = getattr(cfg, "loss_weight_gamma", 2.0)
                loss_weight_time_beta = getattr(cfg, "loss_weight_time_beta", 0.0)

                weights = 1.0

                if loss_weight_alpha > 0.0:
                    # W(Q) = 1 + alpha * |(Q - 0.5) / 0.4|^gamma
                    diff = torch.abs((yb_flat - 0.5) / 0.4)
                    weights = weights * (1.0 + loss_weight_alpha * torch.pow(diff, loss_weight_gamma))
                
                if loss_weight_time_beta > 0.0:
                    # W(t) = 1 + beta * (t / (T-1))
                    t_indices = torch.arange(T, device=device, dtype=torch.float32)
                    if T > 1:
                        t_norm = t_indices / (T - 1)
                    else:
                        t_norm = torch.zeros_like(t_indices)
                    
                    time_weights = 1.0 + loss_weight_time_beta * t_norm
                    # (T,) -> (B*T, 1)
                    time_weights_expanded = time_weights.repeat(B).view(-1, 1)
                    weights = weights * time_weights_expanded

                if isinstance(weights, float) and weights == 1.0:
                    loss = criterion(outputs, yb_flat)
                else:
                    loss = torch.mean(weights * (outputs - yb_flat) ** 2)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                train_losses.append(loss.item())

            train_loss = float(np.mean(train_losses)) if train_losses else 0.0

            # validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    xb, yb = batch
                    xb = xb.to(device)
                    yb = yb.to(device)
                    
                    B, T, F_in = xb.shape
                    _, _, F_out = yb.shape
                    
                    xb_flat = xb.view(-1, F_in)
                    yb_flat = yb.view(-1, F_out)
                    
                    outputs = model(xb_flat)
                    
                    # Handle potential dimension mismatch if output_size was forced differently (unlikely here but good practice)
                    min_dim = min(outputs.shape[-1], yb_flat.shape[-1])
                    
                    loss_weight_alpha = getattr(cfg, "loss_weight_alpha", 0.0)
                    loss_weight_gamma = getattr(cfg, "loss_weight_gamma", 2.0)
                    loss_weight_time_beta = getattr(cfg, "loss_weight_time_beta", 0.0)

                    yb_sliced = yb_flat[..., :min_dim]
                    out_sliced = outputs[..., :min_dim]
                    
                    weights = 1.0

                    if loss_weight_alpha > 0.0:
                        diff = torch.abs((yb_sliced - 0.5) / 0.4)
                        weights = weights * (1.0 + loss_weight_alpha * torch.pow(diff, loss_weight_gamma))
                    
                    if loss_weight_time_beta > 0.0:
                        t_indices = torch.arange(T, device=device, dtype=torch.float32)
                        if T > 1:
                            t_norm = t_indices / (T - 1)
                        else:
                            t_norm = torch.zeros_like(t_indices)
                        
                        time_weights = 1.0 + loss_weight_time_beta * t_norm
                        time_weights_expanded = time_weights.repeat(B).view(-1, 1)
                        weights = weights * time_weights_expanded

                    if isinstance(weights, float) and weights == 1.0:
                        loss = criterion(out_sliced, yb_sliced)
                    else:
                        loss = torch.mean(weights * (out_sliced - yb_sliced) ** 2)
                    
                    val_losses.append(loss.item())

            val_loss = float(np.mean(val_losses)) if val_losses else 0.0

            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

            improved = val_loss + 1e-5 < best_val
            if improved:
                best_val = val_loss
                epochs_since_improve = 0
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "cfg": cfg.__dict__ if hasattr(cfg, "__dict__") else str(cfg),
                }, best_ckpt)
            else:
                epochs_since_improve += 1

            if epochs_since_improve >= cfg.lr_patience:
                prev_lr = optimizer.param_groups[0]["lr"]
                scheduler.step(val_loss)
                cur_lr = optimizer.param_groups[0]["lr"]
                if cur_lr < prev_lr - 1e-12:
                    lr_reductions += 1
                    epochs_since_improve = 0

            if lr_reductions >= cfg.max_lr_reductions and epochs_since_improve >= cfg.early_stop_patience:
                print(f"Early stopping at epoch {epoch}: lr_reductions={lr_reductions}, epochs_since_improve={epochs_since_improve}")
                break

            print(f"node={node_id} Epoch {epoch}/{cfg.epochs}  train={train_loss:.6f}  val={val_loss:.6f}  lr={optimizer.param_groups[0]['lr']:.6e}")

        writer.flush()
        writer.close()
        tqdm.write(f"Training finished for node {node_id}. Best val: {best_val:.6f}. Checkpoint saved to {best_ckpt}")


@hydra.main(version_base=None, config_path="../../config/NBE", config_name="mlp")
def main(cfg: TrainConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
