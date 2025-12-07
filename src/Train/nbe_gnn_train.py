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

from src.Dataloader.nbeGNNDataset import NbeGNNDataset
from src.Networks.nbe_gnn import NbeGNN

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
    gnn_type: str = 'GCN'

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
    save_dir: str = "outputs/gnn_train"
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
    # batch: list of dicts {inputs: [19, N, F], targets: [19, F_target], edge_index: [2, E]}
    # inputs: (batch, 19, N, F)
    inputs = torch.stack([b["inputs"] for b in batch], dim=0)
    # targets: (batch, 19, F_target)
    targets = torch.stack([b["targets"] for b in batch], dim=0)
    
    # edge_index is the same for all samples in the batch (same subgraph structure)
    # We just take the first one.
    # Note: If we want to process the batch in parallel in the GNN, we need to construct a batched edge_index.
    edge_index = batch[0]["edge_index"]
    
    return inputs, targets, edge_index


def create_batched_edge_index(edge_index: torch.Tensor, batch_size: int, num_nodes: int) -> torch.Tensor:
    """
    Creates a batched edge_index for a batch of identical graphs.
    
    Args:
        edge_index: (2, E)
        batch_size: B
        num_nodes: N
        
    Returns:
        batched_edge_index: (2, E * B)
    """
    # edge_index: (2, E)
    # We want to repeat it B times, adding offset to node indices
    
    # offsets: (B, 1) -> (B, 1) * N -> (B, 1)
    # We want offsets like [0, N, 2N, ..., (B-1)N]
    offsets = torch.arange(batch_size, device=edge_index.device) * num_nodes
    # offsets: (B,)
    
    # Expand edge_index to (B, 2, E)
    # But we can't just add (B,) to (2, E).
    
    # Let's do it explicitly
    # edge_index[0] + offset, edge_index[1] + offset
    
    # (2, E) -> (1, 2, E)
    edge_index_expanded = edge_index.unsqueeze(0)
    
    # (B, 1, 1)
    offsets_expanded = offsets.view(-1, 1, 1)
    
    # (B, 2, E)
    batched_edges = edge_index_expanded + offsets_expanded
    
    # (2, B, E) -> (2, B*E)
    batched_edges = batched_edges.permute(1, 0, 2).reshape(2, -1)
    
    return batched_edges


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
        train_ds = NbeGNNDataset(
            data_dir=cfg.train_dir,
            node_id=node_id,
            preload=cfg.preload,
            glob=cfg.glob,
            alpha=cfg.alpha,
            global_normalize=global_normalize,
        )
        val_ds = NbeGNNDataset(
            data_dir=cfg.val_dir,
            node_id=node_id,
            preload=cfg.preload,
            glob=cfg.glob,
            alpha=cfg.alpha,
            global_normalize=global_normalize,
        )

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, persistent_workers=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, persistent_workers=True, collate_fn=collate_fn)

        # Input size is the feature dimension of a single node (e.g. 9)
        # train_ds.input_feature_size was for the flattened vector.
        # We need the feature size per node.
        # Based on NbeGNNDataset, it reshapes to (times, num_nodes, -1).
        # So we can check the shape of the first item.
        sample_item = train_ds[0]
        # inputs: (19, N, F)
        input_size = sample_item["inputs"].shape[-1]
        num_nodes = sample_item["inputs"].shape[1]
        
        target_size = train_ds.target_feature_size
        output_size = cfg.output_size or target_size

        model = NbeGNN(
            input_size=input_size, 
            hidden_size=cfg.hidden_size, 
            output_size=output_size, 
            num_layers=cfg.num_layers, 
            dropout=cfg.dropout,
            gnn_type=cfg.gnn_type
        )
        model.to(device)

        save_root = Path(cfg.save_dir) / f"{node_id}"
        save_root.mkdir(parents=True, exist_ok=True)
        run_dir = save_root
        writer = SummaryWriter(log_dir=str(run_dir / "tb"))

        def run_phase(phase_name: str, load_ckpt: Optional[Path] = None, save_name: str = "best.pth"):
            print(f"\n=== Starting {phase_name} phase ===")
            
            # Load checkpoint if provided (for fine-tuning)
            if load_ckpt is not None and load_ckpt.exists():
                print(f"Loading checkpoint from {load_ckpt}")
                checkpoint = torch.load(load_ckpt, weights_only=False)
                model.load_state_dict(checkpoint["model_state"])

            # Reset optimizer and scheduler for the new phase
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=cfg.lr_factor,
                patience=cfg.lr_patience,
                min_lr=cfg.min_lr,
            )
            criterion = nn.MSELoss()

            best_val = float("inf")
            epochs_since_improve = 0
            lr_reductions = 0
            best_ckpt_path = run_dir / save_name

            for epoch in range(1, cfg.epochs + 1):
                model.train()
                train_losses = []
                for batch in train_loader:
                    xb, yb, edge_index = batch
                    xb = xb.to(device)
                    yb = yb.to(device)
                    edge_index = edge_index.to(device)
                    
                    batch_size = xb.shape[0]
                    seq_len = xb.shape[1]
                    
                    batched_edge_index = create_batched_edge_index(edge_index, batch_size, num_nodes)
                    xb_reshaped = xb.permute(1, 0, 2, 3).reshape(seq_len, batch_size * num_nodes, -1)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(xb_reshaped, batched_edge_index)
                    
                    outputs = outputs.view(seq_len, batch_size, num_nodes, -1).permute(1, 0, 2, 3)
                    outputs_central = outputs[:, :, 0, :]
                    
                    loss = criterion(outputs_central, yb)
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
                        xb, yb, edge_index = batch
                        xb = xb.to(device)
                        yb = yb.to(device)
                        edge_index = edge_index.to(device)
                        
                        batch_size = xb.shape[0]
                        seq_len = xb.shape[1]
                        
                        batched_edge_index = create_batched_edge_index(edge_index, batch_size, num_nodes)
                        xb_reshaped = xb.permute(1, 0, 2, 3).reshape(seq_len, batch_size * num_nodes, -1)
                        
                        outputs = model(xb_reshaped, batched_edge_index)
                        outputs = outputs.view(seq_len, batch_size, num_nodes, -1).permute(1, 0, 2, 3)
                        outputs_central = outputs[:, :, 0, :]
                        
                        min_dim = min(outputs_central.shape[-1], yb.shape[-1])
                        loss = criterion(outputs_central[..., :min_dim], yb[..., :min_dim])
                        val_losses.append(loss.item())

                val_loss = float(np.mean(val_losses)) if val_losses else 0.0

                writer.add_scalar(f"loss/{phase_name}/train", train_loss, epoch)
                writer.add_scalar(f"loss/{phase_name}/val", val_loss, epoch)
                writer.add_scalar(f"lr/{phase_name}", optimizer.param_groups[0]["lr"], epoch)

                improved = val_loss + 1e-5 < best_val
                if improved:
                    best_val = val_loss
                    epochs_since_improve = 0
                    torch.save({
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch,
                        "cfg": cfg.__dict__ if hasattr(cfg, "__dict__") else str(cfg),
                    }, best_ckpt_path)
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

                print(f"[{phase_name}] node={node_id} Epoch {epoch}/{cfg.epochs}  train={train_loss:.6f}  val={val_loss:.6f}  lr={optimizer.param_groups[0]['lr']:.6e}")
            
            tqdm.write(f"{phase_name} finished for node {node_id}. Best val: {best_val:.6f}. Checkpoint saved to {best_ckpt_path}")
            return best_ckpt_path

        # 1. Pre-training Phase
        model.set_finetune(False)
        best_pretrain_ckpt = run_phase("pretrain", load_ckpt=None, save_name="best_pretrain.pth")

        # 2. Fine-tuning Phase
        model.set_finetune(True)
        run_phase("finetune", load_ckpt=best_pretrain_ckpt, save_name="best_finetune.pth")

        writer.flush()
        writer.close()


@hydra.main(version_base=None, config_path="../../config/NBE", config_name="gnn")
def main(cfg: TrainConfig) -> None:
    # Note: We are reusing the peephole_lstm config structure for convenience, 
    # but ideally we should have a separate config for GNN.
    run_training(cfg)


if __name__ == "__main__":
    main()
