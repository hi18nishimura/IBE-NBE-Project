from __future__ import annotations

import os
import torch
import torch.nn as nn
import hydra
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from omegaconf import DictConfig, OmegaConf, MISSING
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

from src.Dataloader.nbeAttentionDataset import NbeAttentionDataset
from src.Networks.nbe_attention_mlp import NbeAttentionMLP

# Re-using the config dataclass for type hinting mostly, though hydra gives DictConfig
# Ideally this should be shared, but defining inline to avoid import issues if not in a module
def collate_fn(batch):
    # batch: list of dicts {inputs: {u, sigma, tau}, targets: {u, sigma, tau, merge}}
    
    u_list = [b["inputs"]["u"] for b in batch]
    sigma_list = [b["inputs"]["sigma"] for b in batch]
    tau_list = [b["inputs"]["tau"] for b in batch]
    
    # Stack inputs (Batch, Time, N, 3) 
    u = torch.stack(u_list, dim=0)
    sigma = torch.stack(sigma_list, dim=0)
    tau = torch.stack(tau_list, dim=0)
    
    # Targets (Batch, Time, Dim)
    targets_merge = torch.stack([b["targets"]["merge"] for b in batch], dim=0)

    return u, sigma, tau, targets_merge

def oka_denormalize(normed_vals: np.ndarray, pwidths: np.ndarray, alpha: float) -> np.ndarray:
    """
    Inverse of oka_normalize.
    x = sign(y - 0.5) * (|y - 0.5| / 0.4)^alpha * pwidth
    """
    # normed_vals: (N_samples, Dim)
    # pwidths: (Dim,) or (1, Dim)
    
    y = normed_vals
    term = y - 0.5
    signs = np.sign(term)
    abs_term = np.abs(term)
    
    # factor = (|y - 0.5| / 0.4) ^ alpha
    # Use broadcasting for pwidths
    factor = (abs_term / 0.4) ** alpha
    
    return signs * factor * pwidths

@hydra.main(version_base=None, config_path="../../config/NBE", config_name="attention_mlp")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Helper to get config values
    def _cfg_get(key: str):
        try:
            if OmegaConf.is_config(cfg):
                return OmegaConf.select(cfg, key)
        except Exception:
            pass
        if isinstance(cfg, dict):
            return cfg.get(key)
        return getattr(cfg, key, None)

    node_min = _cfg_get('node_min')
    node_max = _cfg_get('node_max')
    node_id_cfg = _cfg_get('node_id')
    global_normalize = _cfg_get('global_normalize')

    # Determine nodes to evaluate
    if node_min is not None and node_max is not None:
        node_ids = list(range(int(node_min), int(node_max) + 1))
    else:
        node_ids = [int(node_id_cfg)]

    # Use val_dir as test_dir if not specified otherwise
    # In this script we assume checking accuracy on validation/test set
    test_dir = _cfg_get('val_dir')
    
    # Save root for evaluation results
    base_save_dir = Path(_cfg_get('save_dir').replace("_train", "_eval")) if _cfg_get('save_dir') else Path("outputs/attention_mlp_eval")
    base_save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating on data from: {test_dir}")
    print(f"Results will be saved to: {base_save_dir}")

    for node_id in node_ids:
        print(f"\n--- Evaluating Node ID: {node_id} ---")
        
        # Load Dataset
        try:
            test_ds = NbeAttentionDataset(
                data_dir=test_dir,
                node_id=node_id,
                preload=cfg.preload,
                glob=cfg.glob,
                alpha=cfg.alpha,
                global_normalize=global_normalize
            )
        except Exception as e:
            print(f"Failed to load dataset for node {node_id}: {e}")
            continue

        if len(test_ds) == 0:
            print(f"Dataset for node {node_id} is empty.")
            continue

        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fn)

        # Infer dimensions
        sample_item = test_ds[0]
        u_sample = sample_item["inputs"]["u"]
        sigma_sample = sample_item["inputs"]["sigma"]
        dim_u = u_sample.shape[-1]
        dim_sigma = sigma_sample.shape[-1]
        dim_tau = sample_item["inputs"]["tau"].shape[-1]
        num_nodes_u = u_sample.shape[1]
        num_nodes_sigma = sigma_sample.shape[1]
        
        target_dim = sample_item["targets"]["merge"].shape[-1]
        output_dim = target_dim
        if cfg.output_dim is not None:
             output_dim = cfg.output_dim

        is_fixed = test_ds.is_center_node_fixed
        print(f"Node {node_id}: Fixed={is_fixed}, OutputDim={output_dim}")

        # Initialize Model
        model = NbeAttentionMLP(
            nbe_fixed=is_fixed,
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

        # Load Checkpoint
        # Assuming folder structure: outputs/attention_mlp_train/{node_id}/best.pth
        # We need to construct the path to the trained model. 
        # The cfg.save_dir points to the train output root.
        train_save_dir = Path(cfg.save_dir)
        ckpt_path = train_save_dir / str(node_id) / "best.pth"
        
        if not ckpt_path.exists():
            print(f"Checkpoint not found at {ckpt_path}. Skipping node {node_id}.")
            continue
            
        print(f"Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Run Inference
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for u_b, sigma_b, tau_b, targets_b in tqdm(test_loader, desc=f"Evaluating Node {node_id}"):
                b, t_len, n_u, d_u = u_b.shape
                _, _, n_s, d_s = sigma_b.shape
                _, _, n_t, d_t = tau_b.shape
                
                # Targets: (Batch, Time, Dim)
                y_true = targets_b.to(device)

                preds_list = []
                # Process each time step sequentially (or just loop over time dimension)
                for t in range(t_len):
                    u_in = u_b[:, t, :, :].to(device)       # (Batch, N, Dim)
                    sigma_in = sigma_b[:, t, :, :].to(device) # (Batch, N, Dim)
                    tau_in = tau_b[:, t, :, :].to(device)     # (Batch, N, Dim)
                    
                    pred_step, _ = model(u=u_in, sigma=sigma_in, tau=tau_in)
                    preds_list.append(pred_step)
                
                # Stack predictions -> (Batch, Time, OutputDim)
                pred = torch.stack(preds_list, dim=1)
                
                all_preds.append(pred.cpu().numpy())
                all_targets.append(y_true.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0) # (TotalSamples, Time, Dim)
        all_targets = np.concatenate(all_targets, axis=0) # (TotalSamples, Time, Dim)

        # Denormalize
        try:
            full_pwidths = test_ds.pwidth_array_broadcasted.flatten() # Expected size (9,)
            
            # Select pwidths matching output dimensions
            if output_dim == 9:
                target_pwidths = full_pwidths
            elif output_dim == 6:
                # If fixed node, outputs are [sigma, tau], corresponding to columns 3:9
                target_pwidths = full_pwidths[3:]
            else:
                print(f"Warning: Unexpected output_dim {output_dim}. Using first {output_dim} pwidths.")
                target_pwidths = full_pwidths[:output_dim]
                
            print(f"Denormalizing with alpha={cfg.alpha}")
            all_preds = oka_denormalize(all_preds, target_pwidths, cfg.alpha)
            all_targets = oka_denormalize(all_targets, target_pwidths, cfg.alpha)
            
        except AttributeError:
            print("Warning: pwidth_array_broadcasted not found in dataset. Skipping denormalization.")
        except Exception as e:
            print(f"Error during denormalization: {e}. Skipping.")

        # Handling Output Dimensions
        # If 9 dims: [u(3), sigma(3), tau(3)]
        # If 6 dims: [sigma(3), tau(3)]
        
        errors = all_preds - all_targets # (TotalSamples, Time, Dim)
        
        # Save raw results
        node_save_dir = base_save_dir / str(node_id)
        node_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Flatten for histogram and overall stats
        errors_flat = errors.reshape(-1, errors.shape[-1])
        
        # Calculate Statistics
        stats = {}
        
        # Define component slices
        components = {}
        if output_dim == 9:
            components['u'] = (0, 3)
            components['sigma'] = (3, 6)
            components['tau'] = (6, 9)
        elif output_dim == 6:
            components['sigma'] = (0, 3)
            components['tau'] = (3, 6)
        else:
            # Generic fallback
            components['all'] = (0, output_dim)

        # 1. Error Statistics (Mean, Std, AbsMean per component)
        stats_list = []
        for name, (start, end) in components.items():
            comp_error_flat = errors_flat[:, start:end]
            
            # Per-dimension stats
            mae = np.mean(np.abs(comp_error_flat), axis=0)
            me = np.mean(comp_error_flat, axis=0)
            std = np.std(comp_error_flat, axis=0)
            
            # Aggregated stats (norm or mean across dims)
            mean_mae = np.mean(np.abs(comp_error_flat))
            mean_std = np.mean(np.std(comp_error_flat, axis=0))

            stats[f"{name}_mae"] = mean_mae
            stats[f"{name}_std"] = mean_std
            
            stats_list.append({
                "Component": name,
                "MAE": mean_mae,
                "MeanError": np.mean(me),
                "StdError": mean_std
            })
            
            # 3. Error Distribution (Histogram)
            plt.figure(figsize=(10, 6))
            # Flatten all dimensions of this component for histogram
            sns.histplot(comp_error_flat.flatten(), kde=True, bins=50)
            plt.title(f"Error Distribution: {name} (Node {node_id})")
            plt.xlabel("Error")
            plt.ylabel("Count")
            plt.grid(True)
            plt.savefig(node_save_dir / f"error_hist_{name}.png")
            plt.close()
        
        # Save stats to CSV
        pd.DataFrame(stats_list).to_csv(node_save_dir / "error_statistics.csv", index=False)
        print(f"Saved statistics to {node_save_dir / 'error_statistics.csv'}")

        # 2. Time Series Error
        # Plot Mean Error Norm per Time Step (across all batch samples)
        # errors shape: (TotalSamples, Time, Dim)
        
        plt.figure(figsize=(12, 6))
        
        # Prepare x-axis
        time_steps = errors.shape[1]
        # Targets are from t=2 to t=20 (19 steps)
        x_axis = np.arange(2, 2 + time_steps)
        
        for name, (start, end) in components.items():
            # Extract component error: (Batch, Time, CompDim)
            comp_error = errors[:, :, start:end]
            
            # Compute L2 norm for each sample at each time step: (Batch, Time)
            # norm over CompDim axis=2
            error_norm_per_sample = np.linalg.norm(comp_error, axis=2)
            
            # Mean over Batch axis=0 -> (Time,)
            mean_error_norm_per_step = np.mean(error_norm_per_sample, axis=0)
            
            plt.plot(x_axis, mean_error_norm_per_step, marker='o', label=f"{name} Mean Error Norm", alpha=0.8)
            
        plt.title(f"Mean Error Evolution over Time Window (Node {node_id})")
        plt.xlabel("Time Step (t)")
        plt.ylabel("Mean L2 Error Norm")
        plt.xticks(np.arange(2, 2 + time_steps, 1))
        plt.legend()
        plt.grid(True)
        plt.savefig(node_save_dir / "time_series_error_mean.png")
        plt.close()

if __name__ == "__main__":
    main()
