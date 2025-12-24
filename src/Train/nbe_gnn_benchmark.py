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

from src.Dataloader.nbeGNNDataset import NbeGNNDataset
from src.Networks.nbe_gnn import NbeGNN
from src.Evaluate.nbe_normalize_inverse import oka_denormalize

def create_batched_edge_index(edge_index: torch.Tensor, batch_size: int, num_nodes: int) -> torch.Tensor:
    """
    Creates a batched edge_index for a batch of identical graphs.
    """
    # edge_index: (2, E)
    # Returns: (2, E * batch_size)
    
    edge_indices = []
    for i in range(batch_size):
        # Shift indices by i * num_nodes
        offset = i * num_nodes
        edge_indices.append(edge_index + offset)
    
    return torch.cat(edge_indices, dim=1)

def load_model(model_path: Path, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int, gnn_type: str, device: torch.device, multi_fully_layer: int) -> NbeGNN:
    model = NbeGNN(
        input_size=input_dim,
        hidden_size=hidden_dim,
        output_size=output_dim,
        num_layers=num_layers,
        gnn_type=gnn_type,
        return_only_central=True, # Assuming we are evaluating central node targets
        multi_fully_layer=multi_fully_layer
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
    gnn_type: str,
    alpha: float = 8.0,
    global_normalize: bool = True,
    force_flag: bool = False,
    multi_fully_layer: int = 1
) -> pd.DataFrame:
    
    # Load Dataset
    dataset = NbeGNNDataset(
        data_dir=data_dir,
        node_id=node_id,
        preload=False,
        glob="*.feather",
        alpha=alpha,
        global_normalize=global_normalize,
        force_flag=force_flag
    )
    
    # Load Model
    #model_path = model_dir / str(node_id) / "best_finetune.pth"
    model_path = model_dir / str(node_id) / "best_pretrain.pth"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return pd.DataFrame()

    # Get input size from dataset sample
    sample = dataset[0]
    # inputs: (Seq, Nodes, Feats)
    input_size = sample["inputs"].shape[-1]
    if force_flag:
        input_size += 3
    target_size = dataset.target_feature_size
    
    model = load_model(model_path, input_size, target_size, hidden_size, num_layers, gnn_type, device, multi_fully_layer)
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Prepare max values for denormalization
    max_vals_list = [dataset.max_map.get(col, 1.0) for col in dataset.columns]
    max_vals_tensor = torch.tensor(max_vals_list, dtype=torch.float32).to(device)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # batch is a dict of stacked tensors from collate_fn (default collate)
            # inputs: (B, Seq, Nodes, F)
            # targets: (B, Seq, F_out)
            # edge_index: (B, 2, E) -> This is what default collate does for list of tensors
            
            xb = batch["inputs"].to(device)
            yb = batch["targets"].to(device)
            
            # Handle edge_index
            # Since all graphs in the batch have the same structure (subgraph around node_id),
            # we can take the first edge_index and batch it.
            # batch["edge_index"] from default collate will be (B, 2, E)
            # We just need one (2, E)
            base_edge_index = batch["edge_index"][0].to(device)
            
            B, S, N, F_in = xb.shape
            
            # Create batched edge_index for the whole batch
            batched_edge_index = create_batched_edge_index(base_edge_index, B, N).to(device)
            
            # Forward
            # NbeGNN expects (B, S, N, F) and batched edge_index
            if force_flag:
                force_tensor = batch["force_tensor"].to(device)
                outputs = model(xb, batched_edge_index, force_tensor)
            else:
                outputs = model(xb, batched_edge_index)
            # print(xb.shape,outputs.shape,batched_edge_index.shape)
            # exit()
            # outputs: (B, S, F_out) (since return_only_central=True)
            
            # Flatten for evaluation: (B*S, F_out)
            outputs_flat = outputs.reshape(-1, target_size)
            targets_flat = yb.reshape(-1, target_size)
            
            # Denormalize
            outputs_denorm = oka_denormalize(outputs_flat, max_vals_tensor, alpha)
            targets_denorm = oka_denormalize(targets_flat, max_vals_tensor, alpha)
            
            all_preds.append(outputs_denorm.cpu().numpy())
            all_targets.append(targets_denorm.cpu().numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Determine column names based on target size
    if target_size == 9:
        feature_cols = ["dx", "dy", "dz", "Sxx", "Syy", "Szz", "Sxy", "Syz", "Szx"]
    elif target_size == 6:
        feature_cols = ["Sxx", "Syy", "Szz", "Sxy", "Syz", "Szx"]
    else:
        # Fallback or use dataset columns if available and matching size
        if len(dataset.columns) == target_size:
            feature_cols = dataset.columns
        else:
            feature_cols = [f"feat_{i}" for i in range(target_size)]

    # Calculate Errors
    errors = np.abs(all_preds - all_targets)
    
    # Create DataFrame
    # Add errors
    data_dict = {f"{col}_error": errors[:, i] for i, col in enumerate(feature_cols)}
    # Add predictions
    data_dict.update({f"{col}_pred": all_preds[:, i] for i, col in enumerate(feature_cols)})
    # Add targets
    data_dict.update({f"{col}_target": all_targets[:, i] for i, col in enumerate(feature_cols)})
    
    df = pd.DataFrame(data_dict)
    df["node_id"] = node_id
    
    return df

def plot_error_distribution(df: pd.DataFrame, node_id: int):
    # Plot Errors
    error_cols = [col for col in df.columns if col.endswith("_error")]
    if error_cols:
        _plot_cols(df, error_cols, node_id, "error")

    # Plot Predictions
    pred_cols = [col for col in df.columns if col.endswith("_pred")]
    if pred_cols:
        _plot_cols(df, pred_cols, node_id, "prediction")

def _plot_cols(df: pd.DataFrame, cols: List[str], node_id: int, plot_type: str):
    n_cols = 3
    n_rows = (len(cols) + n_cols - 1) // n_cols
    
    save_dir = Path(f"/workspace/plots/{plot_type}_distribution/{node_id}")
    save_dir.mkdir(parents=True, exist_ok=True)

    for use_log in [False, True]:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, col in enumerate(cols):
            ax = axes[i]
            ax.hist(df[col], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
            
            title = f"Distribution of {col}"
            if use_log:
                title += " (Log Scale)"
                ax.set_yscale('log')
                
            ax.set_title(title)
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)
            
            # Add mean annotation
            mean_val = df[col].mean()
            ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.4f}')
            ax.legend()
            
        # Hide unused subplots
        for j in range(len(cols), len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        
        filename = f"gnn_{plot_type}_distribution_log.png" if use_log else f"gnn_{plot_type}_distribution.png"
        plot_path = save_dir / filename
        
        plt.savefig(plot_path)
        print(f"Saved {plot_type} distribution plot to {plot_path}")
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Benchmark NbeGNN models")
    parser.add_argument("model_dir", type=Path, help="Directory containing node model folders")
    parser.add_argument("data_dir", type=str, help="Directory containing validation/test data")
    parser.add_argument("node_id", type=int, help="Node ID to evaluate")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size of GNN")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--gnn_type", type=str, default="GAT", help="Type of GNN (GCN, GAT)")
    parser.add_argument("--alpha", type=float, default=8.0, help="Oka normalization alpha")
    parser.add_argument("--output", type=Path, default=Path("gnn_benchmark_results.csv"), help="Output CSV file")
    parser.add_argument("--global_normalize", action="store_true", default=True, help="Use global normalization")
    parser.add_argument("--force_flag", action="store_true", help="Use force input")
    parser.add_argument("--multi_fully_layer", type=int, default=1, help="Number of fully connected layers in readout")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Processing node {args.node_id}...")
    try:
        df = evaluate_node(
            args.node_id,
            args.model_dir,
            args.data_dir,
            device,
            args.hidden_size,
            args.num_layers,
            args.gnn_type,
            args.alpha,
            args.global_normalize,
            args.force_flag,
            args.multi_fully_layer
        )
        
        if not df.empty:
            df.to_csv(args.output, index=False)
            print(f"Saved benchmark results to {args.output}")
            
            # Print summary
            print("\nOverall Mean Errors:")
            print(df.drop(columns=["node_id"]).mean())

            print("\nOverall Max Errors:")
            print(df.drop(columns=["node_id"]).abs().max())
            
            plot_error_distribution(df, args.node_id)
        else:
            print("No results generated.")
            
    except Exception as e:
        print(f"Error processing node {args.node_id}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
