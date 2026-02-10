from __future__ import annotations
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict
import hydra
from omegaconf import MISSING, OmegaConf
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd
import gc

# Add workspace root to sys.path if not present
if "/workspace" not in sys.path:
    sys.path.append("/workspace")

from src.Networks.nbe_simple_mlp import NbeSimpleMLP, NbeSimpleSplitMLP
from src.Dataloader.nbeDataset import oka_normalize_dataframe_fast, oka_normalize_array

@dataclass
class RecursiveTrainConfig:
    # data
    train_dir: str = MISSING
    val_dir: str = MISSING
    
    # Structure info
    node_connection_file: Optional[str] = None
    fixed_nodes_file: Optional[str] = None
    summary_overall_max: Optional[str] = None
    node_displacement_file: Optional[str] = None
    
    # Filter nodes
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
    use_split_mlp: bool = False
    
    # optimization
    lr: float = 1e-4
    weight_decay: float = 0.0
    epochs: int = 100
    
    # lr decay
    lr_patience: int = 5
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    lr_decay_start_epoch: int = 30
    lr_decay_step_size: int = 10
    
    # sequence handling
    seq_len: int = 19
    teacher_forcing_ratio: float = 0.0
    
    # misc
    save_dir: str = "outputs/force_simple_mlp_recursive_train"
    seed: int = 42

class ForceSequenceDataset(Dataset):
    def __init__(self, data_dir, glob="*.feather", columns=None, 
                 summary_overall_max=None, alpha=8.0):
        self.files = sorted(Path(data_dir).glob(glob))
        self.columns = columns or ["dx", "dy", "dz", "Sxx", "Syy", "Szz", "Sxy", "Syz", "Szx"]
        self.alpha = alpha
        
        self.max_map = {}
        if summary_overall_max and Path(summary_overall_max).exists():
            df_max = pd.read_csv(summary_overall_max)
            for _, row in df_max.iterrows():
                self.max_map[str(row['feature'])] = float(row['max_value'])
             
        self.pwidth_array = np.array([self.max_map.get(c, 1.0) for c in self.columns])
        self.pwidth_array_broadcasted = self.pwidth_array[np.newaxis, :]
        self.force_pwidths = np.array([self.max_map.get(c, 1.0) for c in ["dx", "dy", "dz"]])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load feather
        df = pd.read_feather(self.files[idx])
        
        # 1. Normalize Main Features
        feat_df = df[self.columns]
        norm_feat_df = oka_normalize_dataframe_fast(feat_df, self.pwidth_array_broadcasted, self.alpha)
        
        # Identify Force Node
        fid = -1
        if 'force_node_id' in df.columns:
            fids = df['force_node_id'].unique()
            fids = fids[~np.isnan(fids)]
            if len(fids) > 0:
                fid = int(fids[0])
            norm_df = pd.concat([df[['time', 'node_id', 'force_node_id']], norm_feat_df], axis=1)
        else:
            dummy = pd.DataFrame({'force_node_id': [np.nan]*len(df)})
            norm_df = pd.concat([df[['time', 'node_id']], dummy, norm_feat_df], axis=1)
            
        # 2. Extract Force Sequence (Normalized)
        if fid != -1:
            # Force node trajectory
            force_rows = df[df['node_id'] == fid].sort_values('time')
            force_vals = force_rows[['dx', 'dy', 'dz']].values
            force_seq_norm = oka_normalize_array(force_vals, self.force_pwidths, self.alpha)
        else:
            # Placeholder length must match T
            # We don't know T yet, but assume it matches unique times
            unique_times = df['time'].unique()
            force_seq_norm = np.zeros((len(unique_times), 3), dtype=np.float32)

        # Reshape Main Data to (T, N, F)
        norm_df = norm_df.sort_values(['time', 'node_id'])
        
        unique_times = norm_df['time'].unique()
        unique_nodes = norm_df['node_id'].unique()
        
        T = len(unique_times)
        N = len(unique_nodes)
        F = len(self.columns)
        
        # Determine forced nodes mask (spatial)
        node_to_idx = {nid: i for i, nid in enumerate(unique_nodes)}
        force_nodes_mask = np.zeros(N, dtype=bool)
        if fid != -1 and fid in node_to_idx:
            force_nodes_mask[node_to_idx[fid]] = True
        
        # Ensure force_seq matches T
        if force_seq_norm.shape[0] != T:
            # Handle mismatch (padding or truncating)
            if force_seq_norm.shape[0] > T:
                force_seq_norm = force_seq_norm[:T]
            else:
                pad = np.zeros((T - force_seq_norm.shape[0], 3), dtype=np.float32)
                force_seq_norm = np.concatenate([force_seq_norm, pad], axis=0)

        data = norm_df[self.columns].values.reshape(T, N, F)
        
        return {
            "data": torch.tensor(data, dtype=torch.float32), 
            "node_ids": torch.tensor(unique_nodes, dtype=torch.long),
            "force_mask": torch.tensor(force_nodes_mask, dtype=torch.bool),
            "force_seq": torch.tensor(force_seq_norm, dtype=torch.float32),
            "force_node_id": fid
        }

def collate_fn(batch):
    data = torch.stack([b['data'] for b in batch])
    node_ids = batch[0]['node_ids'] 
    force_mask = torch.stack([b['force_mask'] for b in batch])
    force_seq = torch.stack([b['force_seq'] for b in batch])
    force_node_ids = torch.tensor([b['force_node_id'] for b in batch], dtype=torch.long)
    return data, node_ids, force_mask, force_seq, force_node_ids

def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_node_distances(csv_path: Path):
    """
    Load node_displacement_features.csv into a tensor lookup.
    Returns: distances[source_id, target_id] -> [x_nor, y_nor, z_nor]
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")
        
    df = pd.read_csv(csv_path)
    
    max_node_id = max(df['node_id'].max(), df['target_node_id'].max())
    dist_matrix = np.zeros((int(max_node_id) + 1, int(max_node_id) + 1, 3), dtype=np.float32)
    
    # Populate (source, target) -> val
    # df has node_id (source, actually central), target_node_id (neighbor or force)
    # The feature file is usually sparse (only close neighbors + some randoms).
    # But for ForceDataset, we assume every node knows distance to *Force Node*.
    # So we hope the CSV is complete enough or we handle zeros.
    # Actually, feature_node_displacement computes all-pairs or relevant pairs?
    # Usually it's limited. But if we miss it, we use 0.
    
    # Efficient fill
    node_ids = df['node_id'].values.astype(int)
    target_ids = df['target_node_id'].values.astype(int)
    vals = df[['x_nor', 'y_nor', 'z_nor']].values.astype(np.float32)
    
    dist_matrix[node_ids, target_ids] = vals
    
    return dist_matrix

def load_models_and_metadata(cfg: RecursiveTrainConfig, device):
    model_dir = Path(cfg.load_model_dir)
    print(f"Loading models from {model_dir}")
    
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
        
        if 'fc1.weight' in state_dict:
            input_dim = state_dict['fc1.weight'].shape[1]
        else:
            continue
        
        # Output dim check
        if 'fc4.weight' in state_dict:
            output_dim = state_dict['fc4.weight'].shape[0]
            is_split = False
        elif 'fc4_1.weight' in state_dict:
            if 'fc4_3.weight' in state_dict:
                output_dim = 9
            else:
                output_dim = 6
            is_split = True
        else:
            continue
            
        if is_split:
            model = NbeSimpleSplitMLP(input_dim, output_dim)
        else:
            model = NbeSimpleMLP(input_dim, output_dim)
            
        model.load_state_dict(state_dict)
        model.to(device)
        model.train() 
        models[nid] = model
        
    return models, connections, fixed_nodes

def build_gather_indices(node_ids_map, target_nodes, connections, fixed_nodes, features_list, device):
    feat_map = {name: i for i, name in enumerate(features_list)}
    num_features = len(features_list)
    disp_indices = [feat_map[x] for x in ['dx', 'dy', 'dz'] if x in feat_map]
    
    gather_indices = {}
    
    for nid in target_nodes:
        if nid not in connections: continue
            
        neighbors = [nid] + connections[nid]
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
    
    # Load Distance Matrix
    if cfg.node_displacement_file:
        disp_file = Path(cfg.node_displacement_file)
    else:
        disp_file = Path("/workspace/dataset/liver_model_info/node_displacement_features.csv")
    
    print("Loading Node Distances...")
    dist_matrix_np = load_node_distances(disp_file)
    dist_matrix = torch.tensor(dist_matrix_np, dtype=torch.float32, device=device)
    
    # 1. Dataset
    val_ds = ForceSequenceDataset(
        data_dir=cfg.val_dir,
        glob=cfg.glob,
        summary_overall_max=cfg.summary_overall_max,
        alpha=cfg.alpha
    )
    sample = val_ds[0]
    sample_node_ids = sample['node_ids'].tolist()
    node_ids_map = {nid: i for i, nid in enumerate(sample_node_ids)}
    
    train_ds = ForceSequenceDataset(
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

    print(f"Starting Recursive Training... Steps: {len(train_loader)} per epoch")
    
    for epoch in range(1, cfg.epochs + 1):
        if epoch >= start_decay and (epoch - start_decay) % decay_step == 0:
            for g in optimizer.param_groups:
                g['lr'] *= lr_factor
            print(f"Decayed learning rate to {optimizer.param_groups[0]['lr']:.6e}")
        
        total_loss = 0
        steps = 0
        
        for m in models.values(): m.train()
            
        for batch_idx, (data, node_ids, force_mask, force_seq, force_node_ids) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            data = data.to(device)
            # force_mask = force_mask.to(device)
            force_seq = force_seq.to(device) # (B, T, 3)
            force_node_ids = force_node_ids.to(device) # (B,)
            
            batch_size, T_seq, N_nodes, F_feat = data.shape
            
            if T_seq < 2: continue
            
            current_state = data[:, 0, :, :].clone() 
            
            limit_t = min(T_seq, cfg.seq_len)
            
            optimizer.zero_grad()
            batch_loss = 0
            
            for t in range(limit_t - 1):
                gt_next_state = data[:, t+1, :, :]
                pred_next_state = gt_next_state.clone() 
                
                flat_state = current_state.view(batch_size, -1)
                
                # Pre-fetch force info for this step
                # Force Input: (B, 3)
                current_force = force_seq[:, t, :]  
                
                for nid, model in models.items():
                    # 1. Spatial Inputs
                    indices = gather_indices[nid]
                    spatial_input = flat_state[:, indices] # (B, Input_spatial)
                    
                    # 2. Force Vector (B, 3) -> current_force
                    
                    # 3. Distance Vector
                    # We need dist(nid, force_node_id) for each batch element
                    # dist_matrix: (MaxNode, MaxNode, 3)
                    # We want dist_matrix[nid, force_node_ids[:]] -> (B, 3)
                    dist_vec = dist_matrix[nid, force_node_ids]
                    
                    # Concatenate Inputs
                    # Input order in NbeForceDataset: [spatial, force, dist]
                    # Note: force is 3, dist is 3.
                    model_input = torch.cat([spatial_input, current_force, dist_vec], dim=1)
                    
                    out = model(model_input)
                    
                    n_idx = node_ids_map[nid]
                    is_fixed = fixed_nodes.get(nid, False)
                    
                    # Update pred_next_state for this node
                    output_feat_count = out.shape[1] 
                    # Assuming output corresponds to first N features of columns
                    # Usually dx,dy,dz (3) or +stress (9)
                    
                    start_col = 0
                    if is_fixed:
                         # Skip dx,dy,dz updates
                         # Assuming first 3 are dx,dy,dz
                         if output_feat_count > 3:
                             pred_next_state[:, n_idx, 3:output_feat_count] = out[:, 3:]
                    else:
                         pred_next_state[:, n_idx, :output_feat_count] = out
                
                loss = criterion(pred_next_state, gt_next_state)
                batch_loss += loss
                
                # Teacher forcing
                if np.random.rand() < cfg.teacher_forcing_ratio:
                    current_state = gt_next_state.clone()
                else:
                    current_state = pred_next_state.detach() # BPTT truncation usually
                    # Or keep gradient? Recursive training usually keeps gradient for BPTT
                    # The original code used:
                    # current_state = pred_next_state (if we want BPTT)
                    # original code:
                    # current_state = pred_next_state.clone() ?? 
                    # Wait, original code:
                    # loss = criterion(...)
                    # batch_loss += loss (accumulate graph)
                    # Then backward() at end of sequence.
                    # So we must use pred_next_state AS IS for next step input.
                    current_state = pred_next_state
            
            # Normalize loss by sequence len
            batch_loss = batch_loss / (limit_t - 1)
            batch_loss.backward()
            optimizer.step()
            
            total_loss += batch_loss.item()
            steps += 1
            
        avg_loss = total_loss / (steps + 1e-9)
        writer.add_scalar("loss/train_epoch", avg_loss, epoch)
        print(f"Epoch {epoch}: Train Loss = {avg_loss:.6f}")
        
        # Validation / Saving
        if epoch % 5 == 0 or epoch == cfg.epochs:
            # Validation logic could be added here similar to nbe_mlp_recursive_train.py
            
            # Save checkpoints in the format of nbe_mlp_recursive_train.py
            for nid, model in models.items():
                node_save_dir = save_root / str(nid)
                node_save_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'cfg': OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
                }, node_save_dir / "best_recursive.pth")

    writer.close()

@hydra.main(version_base=None, config_path="../../config/NBE", config_name="force_recursive_mlp")
def main(cfg: RecursiveTrainConfig):
    run_training(cfg)

if __name__ == "__main__":
    main()
