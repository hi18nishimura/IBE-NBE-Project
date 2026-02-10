from __future__ import annotations
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import hydra
from omegaconf import MISSING, OmegaConf
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd

from src.Networks.nbe_simple_mlp import NbeAutoEncoderMLP
from src.Dataloader.nbeDataset import oka_normalize_dataframe_fast

@dataclass
class RecursiveTrainConfig:
    # data
    train_dir: str = MISSING
    val_dir: str = MISSING
    
    node_connection_file: Optional[str] = None
    fixed_nodes_file: Optional[str] = None
    summary_overall_max: Optional[str] = None
    
    node_min: Optional[int] = None
    node_max: Optional[int] = None
    
    glob: str = "*.feather"
    alpha: float = 8.0
    global_normalize: bool = True
    
    # dataloader
    batch_size: int = 4  
    num_workers: int = 4

    # model
    load_model_dir: str = MISSING 
    
    # optimization
    lr: float = 1e-4 
    weight_decay: float = 0.0
    epochs: int = 100
    
    ae_loss_weight: float = 0.5
    
    # lr decay parameters
    lr_patience: int = 5
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    lr_decay_start_epoch: int = 30
    lr_decay_step_size: int = 10
    
    # sequence handling
    seq_len: int = 19 
    teacher_forcing_ratio: float = 0.0 
    
    # misc
    save_dir: str = "outputs/recursive_autoencoder_mlp_train"
    seed: int = 42

class SequenceDataset(Dataset):
    def __init__(self, data_dir, glob="*.feather", columns=None, 
                 summary_overall_max=None, alpha=8.0):
        self.files = sorted(Path(data_dir).glob(glob))
        self.columns = columns or ["dx", "dy", "dz", "Sxx", "Syy", "Szz", "Sxy", "Syz", "Szx"]
        self.alpha = alpha
        
        # Load Max Values for Normalization
        self.max_map = {}
        if summary_overall_max and Path(summary_overall_max).exists():
            df_max = pd.read_csv(summary_overall_max)
            for _, row in df_max.iterrows():
                self.max_map[str(row['feature'])] = float(row['max_value'])
        
        self.pwidth_array = np.array([self.max_map.get(c, 1.0) for c in self.columns])
        self.pwidth_array_broadcasted = self.pwidth_array[np.newaxis, :]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load feather
        df = pd.read_feather(self.files[idx])
        
        # Normalize efficiently
        feat_df = df[self.columns]
        norm_feat_df = oka_normalize_dataframe_fast(feat_df, self.pwidth_array_broadcasted, self.alpha)
        
        if 'force_node_id' in df.columns:
            norm_df = pd.concat([df[['time', 'node_id', 'force_node_id']], norm_feat_df], axis=1)
        else:
            dummy = pd.DataFrame({'force_node_id': [np.nan]*len(df)})
            norm_df = pd.concat([df[['time', 'node_id']], dummy, norm_feat_df], axis=1)
        
        # Reshape to (T, N, F)
        norm_df = norm_df.sort_values(['time', 'node_id'])
        
        unique_times = norm_df['time'].unique()
        unique_nodes = norm_df['node_id'].unique()
        
        T = len(unique_times)
        N = len(unique_nodes)
        F = len(self.columns)
        
        data = norm_df[self.columns].values.reshape(T, N, F)
        
        # Determine forced nodes
        force_ids = norm_df['force_node_id'].dropna().unique()
        force_nodes_mask = np.zeros(N, dtype=bool)
        
        # Map node_id to index 0..N-1
        node_to_idx = {nid: i for i, nid in enumerate(unique_nodes)}
        
        for fid in force_ids:
            if fid in node_to_idx:
                force_nodes_mask[node_to_idx[fid]] = True
                
        return {
            "data": torch.tensor(data, dtype=torch.float32), # (T, N, F)
            "node_ids": torch.tensor(unique_nodes, dtype=torch.long),
            "force_mask": torch.tensor(force_nodes_mask, dtype=torch.bool) # (N,)
        }

def collate_fn(batch):
    data = torch.stack([b['data'] for b in batch])
    node_ids = batch[0]['node_ids'] # Assume same structure
    force_mask = torch.stack([b['force_mask'] for b in batch])
    return data, node_ids, force_mask

def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_models_and_metadata(cfg: RecursiveTrainConfig, device):
    """
    Load all models and pre-calculate connectivity indices.
    """
    model_dir = Path(cfg.load_model_dir)
    print(f"Loading models from {model_dir}")
    
    # Load metadata
    if cfg.node_connection_file:
        node_conn_path = Path(cfg.node_connection_file)
    else:
        node_conn_path = Path("/workspace/dataset/liver_model_info/node_connections.csv")
    
    if cfg.fixed_nodes_file:
        fixed_node_path = Path(cfg.fixed_nodes_file)
    else:
        fixed_node_path = Path("/workspace/dataset/liver_model_info/fixed_nodes.csv")

    connections = {}
    if node_conn_path.exists():
        df = pd.read_csv(node_conn_path)
        for _, row in df.iterrows():
            nid = int(row['node_id'])
            neigh_raw = row.get('neighbors')
            if pd.isna(neigh_raw):
                connections[nid] = []
            elif isinstance(neigh_raw, str):
                connections[nid] = [int(x) for x in neigh_raw.split(',') if x.strip()]
            else:
                connections[nid] = []
    
    fixed_nodes = {}
    if fixed_node_path.exists():
        df = pd.read_csv(fixed_node_path)
        fixed_nodes = {int(r['node_id']): bool(r['is_fixed']) for _, r in df.iterrows()}

    # Check available models
    model_subdirs = sorted([d for d in model_dir.iterdir() if d.is_dir()])
    
    models = {}
    
    target_nodes = []
    for d in model_subdirs:
        try:
            nid = int(d.name)
            if cfg.node_min is not None and nid < cfg.node_min: continue
            if cfg.node_max is not None and nid > cfg.node_max: continue
            target_nodes.append(nid)
        except ValueError:
            pass
            
    print(f"Found {len(target_nodes)} models to load.")
    
    for nid in tqdm(target_nodes, desc="Loading Models"):
        best_path = model_dir / str(nid) / "best.pth"
        if not best_path.exists():
            continue
            
        checkpoint = torch.load(best_path, map_location='cpu')
        
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Infer shapes (AE encoder first layer)
        # NbeAutoEncoderMLP has ae_enc1.weight
        if 'ae_enc1.weight' in state_dict:
            input_dim = state_dict['ae_enc1.weight'].shape[1]
        elif 'fc1.weight' in state_dict:
            input_dim = state_dict['fc1.weight'].shape[1]
        else:
            print(f"Skipping {nid}: cannot infer input dim")
            continue
            
        if 'fc4.weight' in state_dict:
            output_dim = state_dict['fc4.weight'].shape[0]
        else:
            print(f"Skipping {nid}: cannot infer output dim")
            continue
            
        model = NbeAutoEncoderMLP(input_dim, output_dim)
        model.load_state_dict(state_dict)
        model.to(device)
        model.train() 
        
        models[nid] = model
        
    return models, connections, fixed_nodes

def build_gather_indices(node_ids_map, target_nodes, connections, fixed_nodes, features_list, device):
    """
    Precompute gather indices for each model.
    """
    feat_map = {name: i for i, name in enumerate(features_list)}
    num_features = len(features_list)
    disp_indices = [feat_map[x] for x in ['dx', 'dy', 'dz'] if x in feat_map]
    
    gather_indices = {}
    
    for nid in target_nodes:
        if nid not in connections:
            continue
            
        neighbors = [nid] + connections[nid] # Central then neighbors
        
        indices = []
        valid = True
        for neigh_id in neighbors:
            if neigh_id not in node_ids_map:
                valid = False
                break
                
            neigh_idx = node_ids_map[neigh_id]
            is_fixed = fixed_nodes.get(neigh_id, False)
            
            for f_idx in range(num_features):
                if is_fixed and f_idx in disp_indices:
                    continue
                indices.append(neigh_idx * num_features + f_idx)
        
        if valid:
            gather_indices[nid] = torch.tensor(indices, dtype=torch.long, device=device)
        
    return gather_indices

def run_training(cfg: RecursiveTrainConfig) -> None:
    seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    save_root = Path(cfg.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(save_root / "tb"))
    
    # 1. Dataset
    val_ds = SequenceDataset(
        data_dir=cfg.val_dir,
        glob=cfg.glob,
        summary_overall_max=cfg.summary_overall_max,
        alpha=cfg.alpha
    )
    # Peek at first item to build map
    sample = val_ds[0]
    sample_node_ids = sample['node_ids'].tolist()
    node_ids_map = {nid: i for i, nid in enumerate(sample_node_ids)}
    
    train_ds = SequenceDataset(
        data_dir=cfg.train_dir,
        glob=cfg.glob,
        summary_overall_max=cfg.summary_overall_max,
        alpha=cfg.alpha
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, 
        num_workers=cfg.num_workers, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, 
        num_workers=cfg.num_workers, collate_fn=collate_fn
    )
    
    # 2. Load Models
    models, connections, fixed_nodes = load_models_and_metadata(cfg, device)
    
    # 3. Build Gather Indices
    features = val_ds.columns
    gather_indices = build_gather_indices(
        node_ids_map, list(models.keys()), connections, fixed_nodes, features, device
    )
    
    valid_nids = set(gather_indices.keys())
    models = {k: v for k, v in models.items() if k in valid_nids}
    print(f"Training {len(models)} models with valid connections.")
    
    # 4. Optimizer
    all_params = []
    for m in models.values():
        all_params.extend(m.parameters())
    
    if len(all_params) == 0:
        print("No parameters to optimize!")
        return

    optimizer = torch.optim.Adam(all_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()
    
    start_decay = cfg.lr_decay_start_epoch
    decay_step = cfg.lr_decay_step_size
    lr_factor = cfg.lr_factor

    ae_loss_weight = cfg.ae_loss_weight

    # 5. Loop
    print(f"Starting Recursive Training... Steps: {len(train_loader)} per epoch")
    
    for epoch in range(1, cfg.epochs + 1):
        # LR Decay
        if epoch >= start_decay and (epoch - start_decay) % decay_step == 0:
            for g in optimizer.param_groups:
                g['lr'] *= lr_factor
            print(f"Decayed learning rate to {optimizer.param_groups[0]['lr']:.6e}")
        
        total_loss = 0
        total_main_loss = 0
        total_ae_loss = 0
        steps = 0
        
        for m in models.values(): m.train()
            
        for batch_idx, (data, node_ids, force_mask) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            data = data.to(device)
            force_mask = force_mask.to(device)
            
            batch_size, T_seq, N_nodes, F_feat = data.shape
            
            if T_seq < 2: continue
            
            current_state = data[:, 0, :, :].clone() # (Batch, N, F)
            
            limit_t = min(T_seq, cfg.seq_len)
            
            optimizer.zero_grad()
            batch_loss = 0
            
            batch_main_loss = 0
            batch_ae_loss = 0
            
            flat_gt_all = data.view(batch_size, T_seq, -1) # Flatten nodes for gathering GT
            
            for t in range(limit_t - 1):
                gt_next_state = data[:, t+1, :, :]
                flat_gt_next = flat_gt_all[:, t+1, :]
                
                pred_next_state = gt_next_state.clone() 
                flat_state = current_state.view(batch_size, -1)
                
                step_ae_losses = []
                
                for nid, model in models.items():
                    indices = gather_indices[nid]
                    model_input = flat_state[:, indices]
                    
                    # NbeAutoEncoderMLP returns (out, rec_x)
                    out, rec_x = model(model_input)
                    
                    # 1. Prediction update
                    n_idx = node_ids_map[nid]
                    is_fixed = fixed_nodes.get(nid, False)
                    
                    feat_indices = []
                    for f_i, fname in enumerate(features):
                        if is_fixed and fname in ['dx', 'dy', 'dz']: 
                            continue
                        feat_indices.append(f_i)
                    
                    if feat_indices:
                        pred_next_state[:, n_idx, feat_indices] = out
                        
                    # 2. AE Loss (Reconstruct Input's Next State)
                    # We compare rec_x against the features of neighbors at t+1 (Predictive AE)
                    # target_rec is the same slice of GT at t+1 as the input was at t
                    target_rec = flat_gt_next[:, indices]
                    
                    step_ae_losses.append(criterion(rec_x, target_rec))
                
                # Force constraint
                mask_expanded = force_mask.unsqueeze(-1).expand_as(pred_next_state)
                pred_next_state = torch.where(mask_expanded, gt_next_state, pred_next_state)
                
                # Main Loss
                step_main_loss = criterion(pred_next_state, gt_next_state)
                
                # AE Loss (Average over models)
                step_ae_loss = torch.mean(torch.stack(step_ae_losses)) if step_ae_losses else torch.tensor(0.0).to(device)
                
                step_loss = step_main_loss + ae_loss_weight * step_ae_loss
                
                batch_loss += step_loss
                batch_main_loss += step_main_loss
                batch_ae_loss += step_ae_loss
                
                # Update current_state
                use_teacher = (random.random() < cfg.teacher_forcing_ratio)
                if use_teacher:
                    current_state = gt_next_state
                else:
                    current_state = pred_next_state
                    
            batch_loss.backward()
            optimizer.step()
            
            total_loss += batch_loss.item()
            total_main_loss += batch_main_loss.item()
            total_ae_loss += batch_ae_loss.item()
            steps += 1
            
        avg_loss = total_loss / (steps + 1e-9)
        avg_main = total_main_loss / (steps + 1e-9)
        avg_ae = total_ae_loss / (steps + 1e-9)
        
        writer.add_scalar("loss/train_epoch", avg_loss, epoch)
        writer.add_scalar("loss/train_main", avg_main, epoch)
        writer.add_scalar("loss/train_ae", avg_ae, epoch)
        
        print(f"Epoch {epoch}: Loss={avg_loss:.6f} (Main={avg_main:.6f}, AE={avg_ae:.6f})")
        
        # Validation
        if epoch % 5 == 0 or epoch == cfg.epochs:
            val_loss = 0
            val_steps = 0
            for m in models.values(): m.eval()
            
            with torch.no_grad():
                 for data, _, force_mask in val_loader:
                    data = data.to(device)
                    force_mask = force_mask.to(device)
                    batch_size, T_seq, N_nodes, F_feat = data.shape
                    
                    current_state = data[:, 0, :, :].clone()
                    batch_val_loss = 0
                    
                    limit_t = min(T_seq, cfg.seq_len)
                    for t in range(limit_t - 1):
                        gt_next_state = data[:, t+1, :, :]
                        pred_next_state = gt_next_state.clone()
                        flat_state = current_state.view(batch_size, -1)
                        flat_gt_next = data[:, t+1, :, :].view(batch_size, -1)

                        ae_losses = []
                        for nid, model in models.items():
                            indices = gather_indices[nid]
                            model_input = flat_state[:, indices]
                            out, rec_x = model(model_input)
                            
                            n_idx = node_ids_map[nid]
                            is_fixed = fixed_nodes.get(nid, False)
                            
                            feat_indices = []
                            for f_i, fname in enumerate(features):
                                if is_fixed and fname in ['dx', 'dy', 'dz']: continue
                                feat_indices.append(f_i)
                            
                            if feat_indices:
                                pred_next_state[:, n_idx, feat_indices] = out
                            
                            target_rec = flat_gt_next[:, indices]
                            ae_losses.append(criterion(rec_x, target_rec))
                                
                        mask_expanded = force_mask.unsqueeze(-1).expand_as(pred_next_state)
                        pred_next_state = torch.where(mask_expanded, gt_next_state, pred_next_state)
                        
                        step_main = criterion(pred_next_state, gt_next_state)
                        step_ae = torch.mean(torch.stack(ae_losses)) if ae_losses else 0.0
                        
                        batch_val_loss += step_main + ae_loss_weight * step_ae
                        
                        current_state = pred_next_state
                    
                    val_loss += batch_val_loss
                    val_steps += 1
            
            avg_val_loss = val_loss / (val_steps + 1e-9)
            writer.add_scalar("loss/val_epoch", avg_val_loss, epoch)
            print(f"Epoch {epoch}: Val Loss = {avg_val_loss:.6f}")
        
        # Save checkpoints
        if epoch % 10 == 0 or epoch == cfg.epochs:
            for nid, model in models.items():
                node_save_dir = save_root / str(nid)
                node_save_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'cfg': OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
                }, node_save_dir / "best_recursive.pth")
                
    writer.close()

@hydra.main(version_base=None, config_path="../../config/NBE", config_name="recursive_autoencoder_mlp")
def main(cfg: RecursiveTrainConfig):
    run_training(cfg)

if __name__ == "__main__":
    main()
