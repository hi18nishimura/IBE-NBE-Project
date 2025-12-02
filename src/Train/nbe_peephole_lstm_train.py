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
from src.Networks.nbe_peephole_lstm import NbePeepholeLSTM, PeepholeLSTMCell

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
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.0
    output_size: Optional[int] = None

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

    # misc
    save_dir: str = "outputs/peephole_train"
    seed: int = 42


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        import random

        random.seed(seed)
    except Exception:
        pass


def he_init(module: nn.Module) -> None:
    """Apply He (Kaiming) initialization to Linear layers and peephole cell weights."""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    # PeepholeLSTMCell contains Linear layers `wx` and `wh` and peephole params
    if isinstance(module, PeepholeLSTMCell):
        nn.init.kaiming_uniform_(module.wx.weight, a=math.sqrt(5))
        if module.wx.bias is not None:
            nn.init.zeros_(module.wx.bias)
        nn.init.kaiming_uniform_(module.wh.weight, a=math.sqrt(5))
        if module.wh.bias is not None:
            nn.init.zeros_(module.wh.bias)
        # peephole vectors: initialize to zeros (conservative)
        nn.init.zeros_(module.w_ci)
        nn.init.zeros_(module.w_cf)
        nn.init.zeros_(module.w_co)


def collate_fn(batch):
    # batch: list of dicts {inputs: [19,F], targets: [19,T]}
    inputs = torch.stack([b["inputs"] for b in batch], dim=0)
    targets = torch.stack([b["targets"] for b in batch], dim=0)
    return inputs, targets


def run_training(cfg: TrainConfig) -> None:
    seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # determine node IDs to train
    # cfg may be a dataclass, a dict, or an OmegaConf DictConfig depending on
    # how hydra was invoked (and whether overrides were applied). Read values
    # robustly using OmegaConf.select when available.
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

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, persistent_workers=True)

        input_size = train_ds.input_feature_size
        target_size = train_ds.node_feature_counts[train_ds.node_id]
        output_size = cfg.output_size or target_size

        model = NbePeepholeLSTM(input_size=input_size, hidden_size=cfg.hidden_size, output_size=output_size, num_layers=cfg.num_layers, dropout=cfg.dropout)
        model.apply(he_init)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        # we'll use a manual lr reduction mechanism so we can count reductions
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.lr_factor,
            patience=cfg.lr_patience,
            min_lr=cfg.min_lr,
        )

        criterion = nn.MSELoss()

        # prepare logging: per-node save dir
        # Save to a single node-level directory (do not split by time step/run)
        save_root = Path(cfg.save_dir) / f"{node_id}"
        save_root.mkdir(parents=True, exist_ok=True)
        run_dir = save_root
        writer = SummaryWriter(log_dir=str(run_dir / "tb"))

        best_val = float("inf")
        epochs_since_improve = 0
        lr_reductions = 0
        last_lr = optimizer.param_groups[0]["lr"]

        best_ckpt = run_dir / "best.pth"

        #for epoch in tqdm(range(1, cfg.epochs + 1)):
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            train_losses = []
            for batch in train_loader:
                xb,yb = batch["inputs"], batch["targets"]
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                outputs, _ = model(xb)
                loss = criterion(outputs, yb)
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
                    xb,yb = batch["inputs"], batch["targets"]
                    xb = xb.to(device)
                    yb = yb.to(device)
                    outputs, _ = model(xb)
                    min_dim = min(outputs.shape[-1], yb.shape[-1])
                    loss = criterion(outputs[..., :min_dim], yb[..., :min_dim])
                    val_losses.append(loss.item())

            val_loss = float(np.mean(val_losses)) if val_losses else 0.0

            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

            improved = val_loss + 1e-5 < best_val
            if improved:
                best_val = val_loss
                epochs_since_improve = 0
                # save best
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "cfg": cfg.__dict__ if hasattr(cfg, "__dict__") else str(cfg),
                }, best_ckpt)
            else:
                epochs_since_improve += 1

            # if no improvement for lr_patience epochs, step scheduler
            if epochs_since_improve >= cfg.lr_patience:
                prev_lr = optimizer.param_groups[0]["lr"]
                scheduler.step(val_loss)
                cur_lr = optimizer.param_groups[0]["lr"]
                if cur_lr < prev_lr - 1e-12:
                    lr_reductions += 1
                    epochs_since_improve = 0

            # early stopping: if we've reduced lr enough times and no improvement for early_stop_patience
            if lr_reductions >= cfg.max_lr_reductions and epochs_since_improve >= cfg.early_stop_patience:
                print(f"Early stopping at epoch {epoch}: lr_reductions={lr_reductions}, epochs_since_improve={epochs_since_improve}")
                break

            print(f"node={node_id} Epoch {epoch}/{cfg.epochs}  train={train_loss:.6f}  val={val_loss:.6f}  lr={optimizer.param_groups[0]['lr']:.6e}")

        writer.flush()
        writer.close()
        tqdm.write(f"Training finished for node {node_id}. Best val: {best_val:.6f}. Checkpoint saved to {best_ckpt}")


@hydra.main(version_base=None, config_path="../../config/NBE", config_name="peephole_lstm")
def main(cfg: TrainConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
