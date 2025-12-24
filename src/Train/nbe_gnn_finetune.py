from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import MISSING, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.Dataloader.nbeGNNDataset import NbeGNNDataset
from src.Networks.nbe_gnn_finetune_manager import NbeGNNFinetuneManager

@dataclass
class FinetuneConfig:
    # data
    train_dir: str = MISSING
    val_dir: str = MISSING
    node_id: int = 10
    model_dir: Optional[str] = None
    
    preload: bool = False
    glob: str = "*.feather"
    alpha: float = 8.0
    global_normalize: bool = True

    # dataloader
    batch_size: int = 32
    num_workers: int = 4

    # optimization
    lr: float = 1e-4
    weight_decay: float = 0.0
    epochs: int = 50
    
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
    lr_decay_start_epoch: int = 10
    lr_decay_step_size: int = 10

    # misc
    save_dir: str = "outputs/gnn_finetune"
    seed: int = 42

def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass

def collate_fn_dict(batch_list):
    # batch_list is a list of dictionaries (one per sample)
    # Each dict has keys corresponding to node_ids, and values are the items from NbeGNNDataset
    # We want to collate this into a dictionary of batched tensors
    
    # batch_list: [{node_id: {inputs: ..., targets: ...}}, ...]
    
    if not batch_list:
        return {}
        
    node_ids = batch_list[0].keys()
    batched_data = {}
    
    for node_id in node_ids:
        # Extract items for this node
        items = [b[node_id] for b in batch_list]
        
        # Stack inputs: (Batch, Time, N, F)
        inputs = torch.stack([item["inputs"] for item in items], dim=0)
        # Stack targets: (Batch, Time, F_target)
        targets = torch.stack([item["targets"] for item in items], dim=0)
        # Edge index (same for all in batch)
        edge_index = items[0]["edge_index"]
        
        batched_data[node_id] = {
            "inputs": inputs,
            "targets": targets,
            "edge_index": edge_index
        }
        
    return batched_data

class MultiNodeDataset(torch.utils.data.Dataset):
    def __init__(self, datasets: Dict[int, NbeGNNDataset]):
        self.datasets = datasets
        self.node_ids = list(datasets.keys())
        
        # Find minimum length
        self.length = min(len(ds) for ds in datasets.values())
        
        # Warn if mismatch
        for nid, ds in datasets.items():
            if len(ds) != self.length:
                print(f"Warning: Dataset length mismatch for node {nid}: {len(ds)} (truncated to {self.length})")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = {}
        for nid, ds in self.datasets.items():
            sample[nid] = ds[idx]
        return sample

def create_batched_edge_index(edge_index: torch.Tensor, batch_size: int, num_nodes: int) -> torch.Tensor:
    offsets = torch.arange(batch_size, device=edge_index.device) * num_nodes
    edge_index_expanded = edge_index.unsqueeze(0)
    offsets_expanded = offsets.view(-1, 1, 1)
    batched_edges = edge_index_expanded + offsets_expanded
    batched_edges = batched_edges.permute(1, 0, 2).reshape(2, -1)
    return batched_edges

def run_finetuning(cfg: FinetuneConfig) -> None:
    seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Datasets
    print(f"Initializing datasets for target node {cfg.node_id} and neighbors...")
    
    if cfg.model_dir is None:
        raise ValueError("model_dir must be provided via command line (e.g. model_dir=/path/to/models)")

    # Load target dataset to get neighbors
    target_ds = NbeGNNDataset(
        data_dir=cfg.train_dir,
        node_id=cfg.node_id,
        preload=cfg.preload,
        glob=cfg.glob,
        alpha=cfg.alpha,
        global_normalize=cfg.global_normalize,
    )
    
    # Identify all relevant nodes (Target + Neighbors)
    # node_order[0] is target, rest are neighbors
    relevant_nodes = target_ds.node_order
    print(f"Relevant nodes: {relevant_nodes}")
    
    train_datasets = {}
    val_datasets = {}
    
    # Initialize datasets for all relevant nodes
    for node_id in relevant_nodes:
        train_datasets[node_id] = NbeGNNDataset(
            data_dir=cfg.train_dir,
            node_id=node_id,
            preload=cfg.preload,
            glob=cfg.glob,
            alpha=cfg.alpha,
            global_normalize=cfg.global_normalize
        )
        val_datasets[node_id] = NbeGNNDataset(
            data_dir=cfg.val_dir,
            node_id=node_id,
            preload=cfg.preload,
            glob=cfg.glob,
            alpha=cfg.alpha,
            global_normalize=cfg.global_normalize,
        )

    train_multi_ds = MultiNodeDataset(train_datasets)
    val_multi_ds = MultiNodeDataset(val_datasets)
    
    train_loader = DataLoader(train_multi_ds, batch_size=cfg.batch_size, shuffle=True, 
                              num_workers=cfg.num_workers, collate_fn=collate_fn_dict)
    val_loader = DataLoader(val_multi_ds, batch_size=cfg.batch_size, shuffle=False, 
                            num_workers=cfg.num_workers, collate_fn=collate_fn_dict)

    # 2. Load Models
    print("Loading models...")
    manager = NbeGNNFinetuneManager(
        model_dir=cfg.model_dir,
        node_ids=relevant_nodes,
        device=str(device)
    )
    models = manager.get_models()
    
    # Check if all models are loaded
    loaded_nodes = list(models.keys())
    missing_nodes = set(relevant_nodes) - set(loaded_nodes)
    if missing_nodes:
        print(f"Warning: Models not found for nodes: {missing_nodes}. These will be treated as fixed/boundary if possible or ignored.")

    # 3. Setup Optimizers
    optimizers = {}
    criterion = nn.MSELoss()
    
    for node_id, model in models.items():
        optimizers[node_id] = torch.optim.Adam(
            model.parameters(), 
            lr=cfg.lr, 
            weight_decay=cfg.weight_decay
        )
    
    save_root = Path(cfg.save_dir) / f"target_{cfg.node_id}"
    save_root.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(save_root / "tb"))

    # 4. Training Loop
    best_val_loss = float("inf")
    
    for epoch in range(1, cfg.epochs + 1):
        # LR Decay
        if epoch >= cfg.lr_decay_start_epoch and (epoch - cfg.lr_decay_start_epoch) % cfg.lr_decay_step_size == 0:
            print(f"Decaying learning rate at epoch {epoch}")
            for opt in optimizers.values():
                for param_group in opt.param_groups:
                    if param_group['lr'] > cfg.min_lr:
                        param_group['lr'] *= cfg.lr_factor

        # Train
        manager.train()
        train_losses = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            # batch is {node_id: {inputs, targets, edge_index}}
            
            # Initialize current state at t=0
            # We need the state of ALL relevant nodes at t=0.
            # inputs: (Batch, Time, N, F)
            # For a node i, inputs[:, 0, 0, :] is the feature of node i at t=0 (since index 0 is central)
            
            batch_size = list(batch.values())[0]["inputs"].shape[0]
            seq_len = list(batch.values())[0]["inputs"].shape[1] # 19
            
            # Current state dictionary: {node_id: (Batch, F)}
            current_state = {}
            for node_id in relevant_nodes:
                # Get t=0 features
                # inputs shape: (Batch, Time, N, F)
                # We take time 0, node 0 (central)
                current_state[node_id] = batch[node_id]["inputs"][:, 0, 0, :].to(device)

            # Store predictions for loss calculation
            # {node_id: [(Batch, Output), ...]}
            all_predictions = {node_id: [] for node_id in models.keys()}
            
            # Loop over time steps
            # We predict t=1 to t=seq_len
            # Note: targets correspond to t=1..seq_len (shifted by 1 from inputs)
            
            loss = 0.0
            
            for t in range(seq_len):
                # Prepare inputs for each model
                model_inputs = {}
                edge_indices = {}
                
                for node_id in models.keys():
                    # Construct input for node_id
                    # Need to gather features of neighbors
                    # dataset.node_order gives the order [center, n1, n2, ...]
                    
                    neighbors_order = train_datasets[node_id].node_order
                    
                    # Gather features: (Batch, N_neighbors, F)
                    node_features_list = []
                    for neighbor_id in neighbors_order:
                        if neighbor_id in current_state:
                            node_features_list.append(current_state[neighbor_id])
                        else:
                            # Fallback for missing nodes (should not happen if relevant_nodes is complete)
                            # Or if neighbor is outside the cluster?
                            # If outside, we should probably use ground truth from batch if available, 
                            # but we only loaded relevant_datasets.
                            # If neighbor is not in relevant_nodes, we can't update it.
                            # We assume relevant_nodes covers the subgraph.
                            # If not, we might need to handle boundary conditions.
                            # For now, assume 0.5 (normalized 0)
                            node_features_list.append(torch.full((batch_size, 9), 0.5, device=device))

                    stacked_features = torch.stack(node_features_list, dim=1) # (Batch, N, F)
                    
                    # Add time dimension: (Batch, 1, N, F)
                    model_inputs[node_id] = stacked_features.unsqueeze(1)
                    
                    # Edge index
                    # Need batched edge index
                    num_nodes_local = len(neighbors_order)
                    ei = batch[node_id]["edge_index"].to(device)
                    batched_ei = create_batched_edge_index(ei, batch_size, num_nodes_local)
                    edge_indices[node_id] = batched_ei

                # Forward pass
                # outputs: {node_id: (Batch, 1, Output)} or (Batch, Output) depending on return_only_central
                # Manager returns whatever model returns. 
                # Models are initialized with return_only_central=True, so (Batch, Output) or (Batch, 1, Output)
                outputs = manager.forward(model_inputs, edge_indices)
                
                # Update current state and store predictions
                for node_id, output in outputs.items():
                    # output might be (Batch, Output)
                    if output.dim() == 3:
                        output = output.squeeze(1)
                        
                    all_predictions[node_id].append(output)
                    
                    # Update state for next step
                    # If output is 6D, prepend [0.5, 0.5, 0.5]
                    if output.shape[-1] == 6:
                        padding = torch.tensor([0.5, 0.5, 0.5], device=device).expand(batch_size, 3)
                        next_state = torch.cat([padding, output], dim=-1)
                    else:
                        next_state = output
                    
                    # Update state for next step
                    # We keep the graph connected (no detach) to allow Backpropagation Through Time (BPTT).
                    current_state[node_id] = next_state

            # Calculate Loss
            total_loss = 0.0
            count = 0
            
            for node_id, preds in all_predictions.items():
                # preds: List of (Batch, Output) -> (Batch, Seq, Output)
                preds_tensor = torch.stack(preds, dim=1)
                
                # Targets: (Batch, Seq, TargetDim)
                targets = batch[node_id]["targets"].to(device)
                
                # Align dimensions
                min_dim = min(preds_tensor.shape[-1], targets.shape[-1])
                preds_sliced = preds_tensor[..., :min_dim]
                targets_sliced = targets[..., :min_dim]
                
                # Weighted loss
                weights = 1.0
                
                if cfg.loss_weight_alpha > 0.0:
                    # W(Q) = 1 + alpha * |(Q - 0.5) / 0.4|^gamma
                    diff = torch.abs((targets_sliced - 0.5) / 0.4)
                    weights = weights * (1.0 + cfg.loss_weight_alpha * torch.pow(diff, cfg.loss_weight_gamma))
                
                if cfg.loss_weight_time_beta > 0.0:
                    # W(t) = 1 + beta * (t / (T-1))
                    seq_len_t = targets_sliced.shape[1]
                    t_indices = torch.arange(seq_len_t, device=device, dtype=torch.float32)
                    if seq_len_t > 1:
                        t_norm = t_indices / (seq_len_t - 1)
                    else:
                        t_norm = torch.zeros_like(t_indices)
                    time_weights = 1.0 + cfg.loss_weight_time_beta * t_norm
                    weights = weights * time_weights.view(1, -1, 1)
                
                if targets_sliced.shape[-1] == 9 and cfg.loss_weight_disp > 0.0:
                     disp_weights = torch.tensor([cfg.loss_weight_disp, cfg.loss_weight_disp, cfg.loss_weight_disp, 1, 1, 1, 1, 1, 1], device=device)
                     weights = weights * disp_weights.view(1, 1, -1)

                if isinstance(weights, float) and weights == 1.0:
                    node_loss = criterion(preds_sliced, targets_sliced)
                else:
                    node_loss = torch.mean(weights * (preds_sliced - targets_sliced) ** 2)
                total_loss += node_loss
                count += 1
            
            if count > 0:
                avg_loss = total_loss / count
                
                # Zero grads
                for opt in optimizers.values():
                    opt.zero_grad()
                    
                avg_loss.backward()
                
                # Step
                for opt in optimizers.values():
                    opt.step()
                    
                train_losses.append(avg_loss.item())

        train_loss_epoch = np.mean(train_losses) if train_losses else 0.0
        print(f"Epoch {epoch} Train Loss: {train_loss_epoch:.6f}")
        writer.add_scalar("loss/train", train_loss_epoch, epoch)

        # Validation (Teacher Forcing or Autoregressive? Usually Autoregressive for Val)
        # For simplicity, using same logic as train but no grad
        manager.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch_size = list(batch.values())[0]["inputs"].shape[0]
                seq_len = list(batch.values())[0]["inputs"].shape[1]
                
                current_state = {}
                for node_id in relevant_nodes:
                    current_state[node_id] = batch[node_id]["inputs"][:, 0, 0, :].to(device)

                all_predictions = {node_id: [] for node_id in models.keys()}
                
                for t in range(seq_len):
                    model_inputs = {}
                    edge_indices = {}
                    for node_id in models.keys():
                        neighbors_order = val_datasets[node_id].node_order
                        node_features_list = []
                        for neighbor_id in neighbors_order:
                            if neighbor_id in current_state:
                                node_features_list.append(current_state[neighbor_id])
                            else:
                                node_features_list.append(torch.full((batch_size, 9), 0.5, device=device))
                        stacked_features = torch.stack(node_features_list, dim=1)
                        model_inputs[node_id] = stacked_features.unsqueeze(1)
                        
                        num_nodes_local = len(neighbors_order)
                        ei = batch[node_id]["edge_index"].to(device)
                        batched_ei = create_batched_edge_index(ei, batch_size, num_nodes_local)
                        edge_indices[node_id] = batched_ei

                    outputs = manager.forward(model_inputs, edge_indices)
                    
                    for node_id, output in outputs.items():
                        if output.dim() == 3: output = output.squeeze(1)
                        all_predictions[node_id].append(output)
                        if output.shape[-1] == 6:
                            padding = torch.tensor([0.5, 0.5, 0.5], device=device).expand(batch_size, 3)
                            next_state = torch.cat([padding, output], dim=-1)
                        else:
                            next_state = output
                        current_state[node_id] = next_state

                total_loss = 0.0
                count = 0
                for node_id, preds in all_predictions.items():
                    preds_tensor = torch.stack(preds, dim=1)
                    targets = batch[node_id]["targets"].to(device)
                    min_dim = min(preds_tensor.shape[-1], targets.shape[-1])
                    preds_sliced = preds_tensor[..., :min_dim]
                    targets_sliced = targets[..., :min_dim]

                    # Weighted loss
                    weights = 1.0
                    
                    if cfg.loss_weight_alpha > 0.0:
                        diff = torch.abs((targets_sliced - 0.5) / 0.4)
                        weights = weights * (1.0 + cfg.loss_weight_alpha * torch.pow(diff, cfg.loss_weight_gamma))
                    
                    if cfg.loss_weight_time_beta > 0.0:
                        seq_len_t = targets_sliced.shape[1]
                        t_indices = torch.arange(seq_len_t, device=device, dtype=torch.float32)
                        if seq_len_t > 1:
                            t_norm = t_indices / (seq_len_t - 1)
                        else:
                            t_norm = torch.zeros_like(t_indices)
                        time_weights = 1.0 + cfg.loss_weight_time_beta * t_norm
                        weights = weights * time_weights.view(1, -1, 1)

                    if targets_sliced.shape[-1] == 9 and cfg.loss_weight_disp > 0.0:
                         disp_weights = torch.tensor([cfg.loss_weight_disp, cfg.loss_weight_disp, cfg.loss_weight_disp, 1, 1, 1, 1, 1, 1], device=device)
                         weights = weights * disp_weights.view(1, 1, -1)

                    if isinstance(weights, float) and weights == 1.0:
                        node_loss = criterion(preds_sliced, targets_sliced)
                    else:
                        node_loss = torch.mean(weights * (preds_sliced - targets_sliced) ** 2)

                    total_loss += node_loss
                    count += 1
                
                if count > 0:
                    val_losses.append((total_loss / count).item())

        val_loss_epoch = np.mean(val_losses) if val_losses else 0.0
        print(f"Epoch {epoch} Val Loss: {val_loss_epoch:.6f}")
        writer.add_scalar("loss/val", val_loss_epoch, epoch)
        
        # Save best
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            print(f"New best validation loss: {best_val_loss:.6f}")
            # Save all models
            for node_id, model in models.items():
                # Prepare checkpoint data
                checkpoint_data = {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss_epoch,
                }
                
                # Try to add config if available in manager
                if hasattr(manager, 'configs') and node_id in manager.configs:
                    checkpoint_data['cfg'] = manager.configs[node_id]

                # 1. Save to output dir (node_{node_id}_best_finetune.pth)
                save_path = save_root / f"node_{node_id}_best_finetune.pth"
                torch.save(checkpoint_data, save_path)
                
                # 2. Save to model dir (best_finetune.pth) - Overwrite
                model_dir_path = Path(cfg.model_dir) / str(node_id)
                if model_dir_path.exists():
                    overwrite_path = model_dir_path / "best_finetune.pth"
                    torch.save(checkpoint_data, overwrite_path)
                    # print(f"Saved/Overwrote {overwrite_path}")

    writer.close()

@hydra.main(version_base=None, config_path="../../config/NBE", config_name="gnn")
def main(cfg: FinetuneConfig) -> None:
    run_finetuning(cfg)

if __name__ == "__main__":
    main()