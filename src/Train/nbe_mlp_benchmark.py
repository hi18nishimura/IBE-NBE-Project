from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.Dataloader.nbeDataset import NbeDataset
from src.Networks.nbe_mlp import NbeMLP
from src.Evaluate.nbe_normalize_inverse import oka_denormalize

def load_model(model_path: Path, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int, device: torch.device) -> NbeMLP:
    model = NbeMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_layers
    )
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def evaluate_node(
    node_id: int,
    model_dir: Path,
    data_dir: str,
    device: torch.device,
    hidden_size: int,
    num_layers: int,
    alpha: float = 8.0,
    global_normalize: bool = False
) -> pd.DataFrame:
    
    # Load Dataset
    dataset = NbeDataset(
        data_dir=data_dir,
        node_id=node_id,
        preload=False,
        glob="*.feather",
        alpha=alpha,
        global_normalize=global_normalize
    )
    
    # Load Model
    model_path = model_dir / str(node_id) / "best.pth"
    if not model_path.exists():
        # Try checking if it's directly in the dir (fallback or different structure?)
        # But user said 1-to-1 and structure implies folders.
        # Let's just print error and return empty
        print(f"Model not found: {model_path}")
        return pd.DataFrame()

    model = load_model(model_path, dataset.input_feature_size, dataset.target_feature_size, hidden_size, num_layers, device)
    
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    
    # Prepare max values for denormalization
    max_vals_list = [dataset.max_map.get(col, 1.0) for col in dataset.columns]
    max_vals_tensor = torch.tensor(max_vals_list, dtype=torch.float32).to(device)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            xb, yb = batch["inputs"], batch["targets"]
            xb = xb.to(device)
            yb = yb.to(device)
            
            # Reshape: (B, T, F) -> (B*T, F)
            B, T, F_in = xb.shape
            _, _, F_out = yb.shape
            
            xb_flat = xb.view(-1, F_in)
            yb_flat = yb.view(-1, F_out)
            
            # Forward
            outputs = model(xb_flat)
            
            # Denormalize
            outputs_denorm = oka_denormalize(outputs, max_vals_tensor, alpha)
            targets_denorm = oka_denormalize(yb_flat, max_vals_tensor, alpha)
            
            all_preds.append(outputs_denorm.cpu().numpy())
            all_targets.append(targets_denorm.cpu().numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate Errors
    errors = np.abs(all_preds - all_targets)
    
    # Create DataFrame
    df = pd.DataFrame(errors, columns=[f"{col}_error" for col in dataset.columns])
    df["node_id"] = node_id
    
    return df

def plot_error_distribution(df: pd.DataFrame, node_id: int):
    error_cols = [col for col in df.columns if col.endswith("_error")]
    if not error_cols:
        return

    n_cols = 3
    n_rows = (len(error_cols) + n_cols - 1) // n_cols
    
    save_dir = Path(f"/workspace/plots/error_distribution/{node_id}")
    save_dir.mkdir(parents=True, exist_ok=True)

    for use_log in [False, True]:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, col in enumerate(error_cols):
            ax = axes[i]
            ax.hist(df[col], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
            
            title = f"Distribution of {col}"
            if use_log:
                title += " (Log Scale)"
                ax.set_yscale('log')
                
            ax.set_title(title)
            ax.set_xlabel("Error")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)
            
            # Add mean annotation
            mean_val = df[col].mean()
            ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.4f}')
            ax.legend()
            
        # Hide unused subplots
        for j in range(len(error_cols), len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        
        filename = "mlp_noweight_error_distribution_log.png" if use_log else "mlp_noweight_error_distribution.png"
        plot_path = save_dir / filename
        
        plt.savefig(plot_path)
        print(f"Saved error distribution plot to {plot_path}")
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Benchmark NbeMLP models")
    parser.add_argument("model_dir", type=Path, help="Directory containing node model folders")
    parser.add_argument("data_dir", type=str, help="Directory containing validation/test data")
    parser.add_argument("node_id", type=int, help="Node ID to evaluate")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size of MLP")
    parser.add_argument("--num_layers", type=int, default=9, help="Number of hidden layers")
    parser.add_argument("--alpha", type=float, default=8.0, help="Oka normalization alpha")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.csv"), help="Output CSV file")
    parser.add_argument("--global_normalize", action="store_true", help="Use global normalization")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = []
    node_id = args.node_id
    
    print(f"Processing node {node_id}...")
    try:
        df = evaluate_node(
            node_id,
            args.model_dir,
            args.data_dir,
            device,
            args.hidden_size,
            args.num_layers,
            args.alpha,
            args.global_normalize
        )
        if not df.empty:
            results.append(df)
    except Exception as e:
        print(f"Error processing node {node_id}: {e}")
            
    if results:
        final_df = pd.concat(results, ignore_index=True)
        final_df.to_csv(args.output, index=False)
        print(f"Saved benchmark results to {args.output}")
        
        # Print summary
        print("\nOverall Mean Errors:")
        print(final_df.drop(columns=["node_id"]).mean())
        print("\nOverall Max Errors:")
        print(final_df.drop(columns=["node_id"]).abs().max())
        
        plot_error_distribution(final_df, args.node_id)
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
