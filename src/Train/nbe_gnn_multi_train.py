from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf, MISSING
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from tqdm import tqdm

from src.Dataloader.nbeGNNDataset import NbeGNNDataset
from src.Networks.nbe_gnn_multi_tower_network import MultiTowerGNN

@dataclass
class TrainConfig:
    # data
    train_dir: str = MISSING
    val_dir: str = MISSING
    max_node: int = 140
    preload: bool = False
    glob: str = "*.feather"
    alpha: float = 8.0
    global_normalize: bool = True
    force_flag: bool = False
    
    # dataloader
    batch_size: int = 32
    num_workers: int = 1

    # model parameters (shared across towers except input/output size)
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.0
    gnn_type: str = 'GAT'
    multi_fully_layer: int = 5
    return_only_central: bool = True

    # optimization
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 300
    
    # loss weighting
    loss_weight_alpha: float = 0.0
    loss_weight_gamma: float = 2.0
    loss_weight_time_beta: float = 10.0
    loss_weight_disp: float = 0.0

    # lr/early stopping
    lr_patience: int = 5
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    max_lr_reductions: int = 3
    early_stop_patience: int = 10
    lr_decay_start_epoch: int = 30
    lr_decay_step_size: int = 10

    # misc
    save_dir: str = "outputs/gnn_multi_train"
    seed: int = 42
    
    # path to input_output_size.csv
    input_output_size_path: str = "/workspace/dataset/liver_model_info/input_output_size.csv"
    liver_coord_file: str = "/workspace/dataset/liver_model_info/liver_coordinates.csv"

def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class MultiNodeDataset(Dataset):
    def __init__(self, node_datasets: Dict[str, NbeGNNDataset]):
        self.node_datasets = node_datasets
        # Sort keys numerically to ensure consistent order
        self.keys = sorted(list(node_datasets.keys()), key=lambda x: int(x))
        if not self.keys:
            self.length = 0
        else:
            self.length = len(node_datasets[self.keys[0]])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = {}
        for key in self.keys:
            data[key] = self.node_datasets[key][idx]
        return data

def collate_fn(batch_list):
    # batch_list: list of dicts. Each dict has keys "1", "2", ... and values are samples from NbeGNNDataset
    if not batch_list:
        return {}
        
    keys = batch_list[0].keys()
    batched_data = {}
    
    for key in keys:
        samples = [item[key] for item in batch_list]
        
        # Stack inputs: (B, 19, N, F)
        inputs = torch.stack([s["inputs"] for s in samples], dim=0)
        # Stack targets: (B, 19, F_target)
        targets = torch.stack([s["targets"] for s in samples], dim=0)
        
        # edge_index: take from first sample (static graph)
        edge_index = samples[0]["edge_index"]
        
        batch_dict = {
            "inputs": inputs,
            "targets": targets,
            "edge_index": edge_index
        }
        
        if samples[0].get("force_tensor") is not None:
            batch_dict["force_tensor"] = torch.stack([s["force_tensor"] for s in samples], dim=0)
            
        batched_data[key] = batch_dict
        
    return batched_data

def create_batched_edge_index(edge_index: torch.Tensor, batch_size: int, num_nodes: int) -> torch.Tensor:
    offsets = torch.arange(batch_size, device=edge_index.device) * num_nodes
    edge_index_expanded = edge_index.unsqueeze(0)
    offsets_expanded = offsets.view(-1, 1, 1)
    batched_edges = edge_index_expanded + offsets_expanded
    batched_edges = batched_edges.permute(1, 0, 2).reshape(2, -1)
    return batched_edges

def run_training(cfg: DictConfig) -> None:
    seed_all(cfg.seed)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # Load input/output sizes
    io_df = pd.read_csv(cfg.input_output_size_path)
    io_map = {str(row['node_id']): row for _, row in io_df.iterrows()}
    
    # Initialize datasets
    node_datasets_train = {}
    node_datasets_val = {}
    
    # Determine nodes to train
    target_nodes = []
    for i in range(1, cfg.max_node + 1):
        if str(i) in io_map:
            target_nodes.append(i)
            
    print(f"Initializing datasets for {len(target_nodes)} nodes...")
    
    for node_id in tqdm(target_nodes, desc="Loading Datasets"):
        node_str = str(node_id)
        
        # Train dataset
        node_datasets_train[node_str] = NbeGNNDataset(
            data_dir=cfg.train_dir,
            node_id=node_id,
            preload=cfg.preload,
            glob=cfg.glob,
            alpha=cfg.alpha,
            global_normalize=cfg.global_normalize,
            liver_coord_file=cfg.get("liver_coord_file", None),
            force_flag=cfg.force_flag
        )
        
        # Val dataset
        node_datasets_val[node_str] = NbeGNNDataset(
            data_dir=cfg.val_dir,
            node_id=node_id,
            preload=cfg.preload,
            glob=cfg.glob,
            alpha=cfg.alpha,
            global_normalize=cfg.global_normalize,
            liver_coord_file=cfg.get("liver_coord_file", None),
            force_flag=cfg.force_flag
        )

    train_ds = MultiNodeDataset(node_datasets_train)
    val_ds = MultiNodeDataset(node_datasets_val)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, 
                              num_workers=cfg.num_workers, persistent_workers=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, 
                            num_workers=cfg.num_workers, persistent_workers=True, collate_fn=collate_fn)
    
    # Initialize Model
    model_configs = {}
    for node_id in target_nodes:
        node_str = str(node_id)
        row = io_map[node_str]
        
        input_size = int(row['input_size'])
        if cfg.force_flag:
            input_size += 3
            
        model_configs[node_str] = {
            'input_size': input_size,
            'hidden_size': cfg.hidden_size,
            'output_size': int(row['output_size']),
            'num_layers': cfg.num_layers,
            'dropout': cfg.dropout,
            'gnn_type': cfg.gnn_type,
            'return_only_central': cfg.return_only_central,
            'multi_fully_layer': cfg.multi_fully_layer
        }
        
    model = MultiTowerGNN(model_configs)
    model.to(device)
    
    save_root = Path(cfg.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(save_root / "tb"))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()
    
    best_val = float("inf")
    
    print("Starting training...")
    
    for epoch in range(1, cfg.epochs + 1):
        # LR Decay
        if epoch >= cfg.lr_decay_start_epoch and (epoch - cfg.lr_decay_start_epoch) % cfg.lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= cfg.lr_factor
                
        model.train()
        train_losses = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False):
            optimizer.zero_grad()
            
            model_inputs = {}
            total_loss = 0.0
            
            # Prepare inputs for all towers
            for node_str, data in batch.items():
                xb = data["inputs"].to(device)
                edge_index = data["edge_index"].to(device)
                
                batch_size = xb.shape[0]
                num_nodes = xb.shape[2]
                
                batched_edge_index = create_batched_edge_index(edge_index, batch_size, num_nodes)
                
                model_inputs[node_str] = {
                    'x': xb,
                    'edge_index': batched_edge_index
                }
            
            # Forward pass
            outputs = model(model_inputs)
            
            # Calculate loss
            for node_str, pred in outputs.items():
                yb = batch[node_str]["targets"].to(device)
                
                loss_weight_disp = cfg.loss_weight_disp
                loss_weight_time_beta = cfg.loss_weight_time_beta
                
                weights = 1.0
                
                # Time weighting
                if loss_weight_time_beta > 0.0:
                    seq_len = pred.shape[1]
                    time_steps = torch.arange(seq_len, device=device)
                    t = time_steps / (seq_len - 1) if seq_len > 1 else torch.zeros_like(time_steps)
                    w = torch.exp(loss_weight_time_beta * t)
                    w = w / w.mean()
                    weights = w.view(1, -1, 1)
                
                # Disp weighting
                if pred.shape[-1] == 9 and loss_weight_disp > 0.0:
                    w_disp = torch.ones(9, device=device)
                    w_disp[3:6] = loss_weight_disp
                    if isinstance(weights, float):
                        weights = w_disp.view(1, 1, -1)
                    else:
                        weights = weights * w_disp.view(1, 1, -1)

                if isinstance(weights, torch.Tensor):
                    loss = (weights * (pred - yb) ** 2).mean()
                else:
                    loss = criterion(pred, yb)

                total_loss += loss
            
            total_loss = total_loss / len(outputs)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(total_loss.item())
            
        train_loss_avg = np.mean(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} Val", leave=False):
                model_inputs = {}
                total_loss = 0.0
                
                for node_str, data in batch.items():
                    xb = data["inputs"].to(device)
                    edge_index = data["edge_index"].to(device)
                    batch_size = xb.shape[0]
                    num_nodes = xb.shape[2]
                    batched_edge_index = create_batched_edge_index(edge_index, batch_size, num_nodes)
                    model_inputs[node_str] = {'x': xb, 'edge_index': batched_edge_index}
                
                outputs = model(model_inputs)
                
                for node_str, pred in outputs.items():
                    yb = batch[node_str]["targets"].to(device)
                    loss = criterion(pred, yb)
                    total_loss += loss
                
                total_loss = total_loss / len(outputs)
                val_losses.append(total_loss.item())
        
        val_loss_avg = np.mean(val_losses)
        
        writer.add_scalar("loss/train", train_loss_avg, epoch)
        writer.add_scalar("loss/val", val_loss_avg, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        
        print(f"Epoch {epoch}/{cfg.epochs}  train={train_loss_avg:.6f}  val={val_loss_avg:.6f}  lr={optimizer.param_groups[0]['lr']:.6e}")
        
        if val_loss_avg < best_val:
            best_val = val_loss_avg
            torch.save(model.state_dict(), save_root / "best_model.pth")
            print(f"Saved best model to {save_root / 'best_model.pth'}")

    writer.close()

@hydra.main(version_base=None, config_path="../../config/NBE", config_name="multi_gnn")
def main(cfg: DictConfig) -> None:
    # Convert DictConfig to TrainConfig to ensure type safety and default values
    # However, since we are using hydra, cfg is a DictConfig.
    # We can access keys directly or convert it.
    # For simplicity, let's just use cfg as is, but we need to make sure keys exist.
    # Or we can instantiate the dataclass from the config.
    
    # If we want to use the dataclass defaults for missing keys in yaml:
    # train_cfg = OmegaConf.structured(TrainConfig)
    # cfg = OmegaConf.merge(train_cfg, cfg)
    
    run_training(cfg)

if __name__ == "__main__":
    main()
