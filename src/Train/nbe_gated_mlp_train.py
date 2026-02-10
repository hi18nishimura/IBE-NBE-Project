from __future__ import annotations
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import hydra
from omegaconf import MISSING, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.Dataloader.nbeDataset import NbeDataset
from src.Networks.nbe_gated_mlp import NbeGatedMLP

@dataclass
class TrainConfig:
    # data
    train_dir: str = MISSING
    val_dir: str = MISSING
    node_id: int = 10
    node_min: Optional[int] = None
    node_max: Optional[int] = None
    preload: bool = False
    glob: str = "*.feather"
    alpha: float = 8.0
    global_normalize: bool = True
    
    # dataloader
    batch_size: int = 32
    num_workers: int = 4

    # model architecture
    d_model: int = 64
    d_ffn: int = 128
    seq_len: int = 1
    num_layers: int = 6

    # optimization
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 300
    
    # lr/early stopping parameters
    lr_patience: int = 5
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    lr_decay_start_epoch: int = 30
    lr_decay_step_size: int = 10

    # misc
    save_dir: str = "outputs/gated_mlp_train"
    seed: int = 42

def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def collate_fn(batch):
    # batch is list of dicts {inputs: (19, F), targets: (19, F_out)}
    inputs = torch.stack([b["inputs"] for b in batch], dim=0)
    targets = torch.stack([b["targets"] for b in batch], dim=0)
    return inputs, targets

def run_training(cfg: TrainConfig) -> None:
    seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _cfg_get(key: str):
         # Helper to get from cfg whether it's Dict, Config, or Dataclass
         if OmegaConf.is_config(cfg):
             return OmegaConf.select(cfg, key)
         if isinstance(cfg, dict):
             return cfg.get(key)
         return getattr(cfg, key, None)

    node_min = _cfg_get('node_min')
    node_max = _cfg_get('node_max')
    node_id_cfg = _cfg_get('node_id')
    
    if node_min is not None and node_max is not None:
        node_ids = list(range(int(node_min), int(node_max) + 1))
    else:
        node_ids = [int(node_id_cfg)]

    for node_id in node_ids:
        print(f"Starting training for node_id={node_id}")
        
        train_ds = NbeDataset(
            data_dir=_cfg_get('train_dir'),
            node_id=node_id,
            preload=_cfg_get('preload'),
            glob=_cfg_get('glob'),
            alpha=_cfg_get('alpha'),
            global_normalize=_cfg_get('global_normalize')
        )
        val_ds = NbeDataset(
            data_dir=_cfg_get('val_dir'),
            node_id=node_id,
            preload=_cfg_get('preload'),
            glob=_cfg_get('glob'),
            alpha=_cfg_get('alpha'),
            global_normalize=_cfg_get('global_normalize')
        )
        
        train_loader = DataLoader(
            train_ds, 
            batch_size=_cfg_get('batch_size'), 
            shuffle=True, 
            num_workers=_cfg_get('num_workers'), 
            persistent_workers=True, 
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_ds, 
            batch_size=_cfg_get('batch_size'), 
            shuffle=False, 
            num_workers=_cfg_get('num_workers'), 
            persistent_workers=True, 
            collate_fn=collate_fn
        )
        
        input_size = train_ds.input_feature_size
        output_size = train_ds.target_feature_size
        
        print(f"Input size: {input_size}, Output size: {output_size}")

        model = NbeGatedMLP(
            input_dim=input_size, 
            output_dim=output_size,
            d_model=_cfg_get('d_model'),
            d_ffn=_cfg_get('d_ffn'),
            seq_len=_cfg_get('seq_len'),
            num_layers=_cfg_get('num_layers')
        )
        
        model.to(device)
        
        save_root = Path(_cfg_get('save_dir')) / f"{node_id}"
        save_root.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(save_root / "tb"))
        
        optimizer = torch.optim.Adam(model.parameters(), lr=_cfg_get('lr'), weight_decay=_cfg_get('weight_decay'))
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        
        epochs = _cfg_get('epochs')
        start_decay = _cfg_get('lr_decay_start_epoch')
        decay_step = _cfg_get('lr_decay_step_size')
        lr_factor = _cfg_get('lr_factor')
        
        for epoch in range(1, epochs + 1):
             # LR Decay
             if epoch >= start_decay and (epoch - start_decay) % decay_step == 0:
                 for g in optimizer.param_groups:
                     g['lr'] *= lr_factor
            
             model.train()
             train_losses = []
             for inputs, targets in train_loader:
                 inputs = inputs.to(device)
                 targets = targets.to(device)
                 
                 # Reshape [Batch, Time, Feat] -> [Batch*Time, Feat]
                 inputs = inputs.reshape(-1, input_size)
                 targets = targets.reshape(-1, output_size)
                 
                 optimizer.zero_grad()
                 outputs = model(inputs)
                 loss = criterion(outputs, targets)
                 loss.backward()
                 optimizer.step()
                 
                 train_losses.append(loss.item())
             
             train_loss = np.mean(train_losses)
             
             model.eval()
             val_losses = []
             with torch.no_grad():
                 for inputs, targets in val_loader:
                     inputs = inputs.to(device)
                     targets = targets.to(device)

                     # Reshape [Batch, Time, Feat] -> [Batch*Time, Feat]
                     inputs = inputs.reshape(-1, input_size)
                     targets = targets.reshape(-1, output_size)

                     outputs = model(inputs)
                     loss = criterion(outputs, targets)
                     val_losses.append(loss.item())
             
             val_loss = np.mean(val_losses)
             
             writer.add_scalar("loss/train", train_loss, epoch)
             writer.add_scalar("loss/val", val_loss, epoch)
             writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
             
             if val_loss < best_val_loss:
                 best_val_loss = val_loss
                 # Handle config saving whether it is a dict or OmegaConf
                 cfg_to_save = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
                 if not isinstance(cfg_to_save, dict) and hasattr(cfg_to_save, '__dict__'):
                     cfg_to_save = cfg_to_save.__dict__
                     
                 torch.save({'model_state_dict': model.state_dict(), 'cfg': cfg_to_save }, save_root / "best.pth")
             
             print(f"node={node_id} Epoch {epoch}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}  lr={optimizer.param_groups[0]['lr']:.6e}")
        
        tqdm.write(f"Training finished for node {node_id}. Best val: {best_val_loss:.6f}. Checkpoint saved to {save_root / 'best.pth'}")
        writer.close()

# Config name updated to gated_mlp
@hydra.main(version_base=None, config_path="../../config/NBE", config_name="gated_mlp")
def main(cfg: TrainConfig):
    run_training(cfg)

if __name__ == "__main__":
    main()
