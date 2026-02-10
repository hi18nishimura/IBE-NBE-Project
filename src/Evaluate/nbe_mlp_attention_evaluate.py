import argparse
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf

# Add workspace root to sys.path
sys.path.append("/workspace")

from src.Networks.nbe_attention_mlp import NbeAttentionMLP
from src.Dataloader.nbeAttentionDataset import NbeAttentionDataset
from src.Evaluate.nbe_normalize_inverse import oka_denormalize

def oka_normalize_tensor(x, max_values, alpha=8.0):
    """
    Apply Oka normalization to a tensor.
    x: (F,) or (N, F)
    max_values: (F,)
    """
    eps = 1e-8
    signs = torch.sign(x)
    absvals = torch.abs(x)
    normalized = signs * 0.4 * torch.pow(absvals / (max_values + eps), 1.0/alpha) + 0.5
    return normalized

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate NBE Attention MLP models recursively")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing node subdirectories (outputs/attention_mlp_train)")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing dataset files")
    parser.add_argument("--output_dir", type=str, default="outputs/attention_mlp_recursive_eval", help="Directory to save results")
    parser.add_argument("--feature_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers_mlp", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--global_normalize", action="store_true", default=True)
    return parser.parse_args()

def load_models_and_datasets(args):
    model_dir_path = Path(args.model_dir)
    dataset_dir_path = Path(args.dataset_dir)
    
    if not model_dir_path.exists():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

    models = {}
    node_params = {}
    neighbor_map = {}
    fixed_nodes_map = {} # node_id -> is_fixed
    datasets = {}
    
    # Subdirectories named by node_id
    subdirs = [d for d in model_dir_path.iterdir() if d.is_dir()]
    
    # Pre-search auxiliary files
    summary_overall_max = "/workspace/dataset/bin/toy_all_model/train/summary_overall_max_values.csv"
    if not Path(summary_overall_max).exists():
        possible = list(dataset_dir_path.rglob("summary_overall_max_values.csv"))
        if possible: summary_overall_max = possible[0]

    for subdir in tqdm(subdirs, desc="Loading models/datasets"):
        try:
            node_id = int(subdir.name)
        except ValueError:
            continue
            
        ckpt_path = subdir / "best.pth"
        if not ckpt_path.exists():
            continue
            
        # Initialize Dataset to get dimensions
        try:
            dataset = NbeAttentionDataset(
                data_dir=dataset_dir_path,
                node_id=node_id,
                alpha=args.alpha,
                global_normalize=args.global_normalize,
                summary_overall_max=summary_overall_max,
                preload=False
            )
        except Exception as e:
            print(f"Failed to load dataset for node {node_id}: {e}")
            continue

        if len(dataset) == 0:
            continue
            
        # Infer dimensions
        sample = dataset[0]
        dim_u = sample['inputs']['u'].shape[-1]
        dim_sigma = sample['inputs']['sigma'].shape[-1]
        dim_tau = sample['inputs']['tau'].shape[-1]
        num_nodes_u = sample['inputs']['u'].shape[1]
        num_nodes_sigma = sample['inputs']['sigma'].shape[1]
        
        target_dim = sample['targets']['merge'].shape[-1]
        is_fixed = dataset.is_center_node_fixed

        # Init Model
        model = NbeAttentionMLP(
            nbe_fixed=is_fixed,
            dim_u=dim_u,
            dim_sigma=dim_sigma,
            dim_tau=dim_tau,
            feature_dim=args.feature_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            output_dim=target_dim,
            num_heads=args.num_heads,
            num_layers_mlp=args.num_layers_mlp,
            num_nodes_u=num_nodes_u,
            num_nodes_sigma=num_nodes_sigma
        )
        
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Failed to load weights for node {node_id}: {e}")
            continue
            
        model.eval()
        models[node_id] = model
        datasets[node_id] = dataset
        neighbor_map[node_id] = dataset.node_order # [center, n1, n2...]
        fixed_nodes_map[node_id] = dataset.fixed_nodes # dict {nid: bool} or similar
        
        # Max Values for normalization
        max_vals_list = [dataset.max_map.get(col, 1.0) for col in dataset.columns]
        
        node_params[node_id] = {
            "alpha": args.alpha,
            "max_values_map": dataset.max_map,
            "is_fixed": is_fixed,
            "target_dim": target_dim
        }

    return models, node_params, neighbor_map, fixed_nodes_map, datasets

def get_time1_data_normalized(datasets, idx, node_params):
    """
    Get t=1 normalized state for all nodes.
    Used to initialize the recursion.
    """
    norm_state = {}
    
    for node_id, ds in datasets.items():
        # Get processed sample
        sample = ds[idx]
        
        # At t=0 (Time 1):
        # inputs correspond to t=1 state.
        u_t1 = sample['inputs']['u'][0] # (N_u, 3)
        sigma_t1 = sample['inputs']['sigma'][0]
        tau_t1 = sample['inputs']['tau'][0]
        
        is_fixed = node_params[node_id]['is_fixed']
        
        # Prepare tensors for central node components
        if is_fixed:
            # For fixed nodes, u is logically 0 in physical space.
            # In Oka normalization, 0 maps to 0.5.
            val_u = torch.ones(3) * 0.5 
        else:
            val_u = u_t1[0]
        
        # sigma, tau always exist
        val_sigma = sigma_t1[0]
        val_tau = tau_t1[0]
        
        norm_state[node_id] = {
            'u': val_u,
            'sigma': val_sigma,
            'tau': val_tau
        }
    return norm_state

def build_gpu_indices(models, neighbor_map, fixed_nodes_map, device):
    """
    Precompute neighbors indices on GPU to avoid CPU overhead in loop.
    Returns:
        indices_u: dict[node_id] -> LongTensor (indices for u input)
        indices_st: dict[node_id] -> LongTensor (indices for sigma/tau input)
    """
    indices_u = {}
    indices_st = {}
    
    for node_id in models.keys():
        neighbors = neighbor_map[node_id] # [center, n1, n2...]
        target_fixed_map = fixed_nodes_map[node_id]
        
        u_list = []
        st_list = []
        
        for nid in neighbors:
            # st always included
            st_list.append(nid)
            
            # u included only if not fixed
            if not target_fixed_map.get(nid, False):
                u_list.append(nid)
                
        indices_u[node_id] = torch.tensor(u_list, dtype=torch.long, device=device)
        indices_st[node_id] = torch.tensor(st_list, dtype=torch.long, device=device)
        
    return indices_u, indices_st

def run_evaluation(models, node_params, neighbor_map, fixed_nodes_map, datasets, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move models to GPU
    for m in models.values():
        m.to(device)
        m.eval() # ensure eval mode
        
    # Precompute indices
    indices_u_gpu, indices_st_gpu = build_gpu_indices(models, neighbor_map, fixed_nodes_map, device)
        
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not datasets:
        return

    first_node = list(datasets.keys())[0]
    num_samples = len(datasets[first_node])
    
    # Determine max node id to size the global tensor
    all_nodes = set(models.keys())
    for nlist in neighbor_map.values():
        all_nodes.update(nlist)
    max_node_id = max(all_nodes)
    
    # 0.5 is the zero-point in Oka normalization
    initial_tensor_val = 0.5
    
    for idx in range(num_samples):
        print(f"Evaluating Sample {idx+1}/{num_samples}")
        
        # 1. Initialize normalized state at t=1
        # We build the big GPU tensor directly
        # Fill with 0.5 (neutral) initially
        current_state = torch.full((max_node_id + 1, 3, 3), initial_tensor_val, device=device)
        
        # Fill initial values from dataset (CPU -> GPU)
        t1_data = get_time1_data_normalized(datasets, idx, node_params)
        for nid, val in t1_data.items():
            # val['u'], val['sigma'], val['tau'] are (3,) tensors
            node_state = torch.stack([val['u'], val['sigma'], val['tau']]) # (3, 3)
            current_state[nid] = node_state.to(device)
            
        # Get Force Node Info (Ground Truth Displacement)
        ds_first = datasets[first_node]
        force_info_raw = ds_first.get_force_node_info(idx)
        
        # Prepare Forced Values Tensor on GPU
        force_vals_norm = {}
        for fnid, raw_disp in force_info_raw.items():
            if fnid in node_params:
                params = node_params[fnid]
                alpha = params['alpha']
                max_map = params['max_values_map']
                max_u = torch.tensor([max_map.get(c, 1.0) for c in ['dx', 'dy', 'dz']], device=device)
                
                # raw_disp comes from dataset.get_force_node_info, which ALREADY applies normalization.
                # So we just convert to tensor.
                norm_tensor = torch.from_numpy(raw_disp).float().to(device)
                force_vals_norm[fnid] = norm_tensor
        

        # History lists to store results (CPU)
        history = []
        
        # 2. Loop T = 2 to 20 (19 steps)
        next_state = current_state.clone()
        
        for step in tqdm(range(19), desc=f"Sample {idx+1}"):
            
            # Iterate over "active" nodes (those we have models for)
            for node_id, model in models.items():
                
                # Fast gather inputs
                idx_u = indices_u_gpu[node_id]
                idx_st = indices_st_gpu[node_id]
                
                u_in = current_state[idx_u, 0, :].unsqueeze(0) # (1, Nu, 3)
                sigma_in = current_state[idx_st, 1, :].unsqueeze(0) # (1, Nst, 3)
                tau_in = current_state[idx_st, 2, :].unsqueeze(0)   # (1, Nst, 3)
                
                # Inference
                with torch.no_grad():
                    pred, _ = model(u=u_in, sigma=sigma_in, tau=tau_in)
                    pred = pred[0] # (OutputDim,)
                
                # Update next state directly on GPU
                start_ptr = 0
                if node_params[node_id]['target_dim'] == 9:
                    next_state[node_id, 0, :] = pred[0:3]
                    start_ptr = 3
                
                next_state[node_id, 1, :] = pred[start_ptr:start_ptr+3]
                next_state[node_id, 2, :] = pred[start_ptr+3:start_ptr+6]
                
            # Update global state (GPU swap)
            current_state.copy_(next_state)
            
            # OVERWRITE Forced Nodes with Ground Truth for the CURRENT step (which is now T = step + 2)
            # step 0 -> Predict T=2 -> Current Time is 2.
            # force_vals_norm index: 0 is T=1, 1 is T=2...
            # So index needed is step + 1.
            target_time_idx = step + 1
            
            for fnid, norm_tensor in force_vals_norm.items():
                if target_time_idx < norm_tensor.shape[0]:
                    # u is channel 0
                    current_state[fnid, 0, :] = norm_tensor[target_time_idx]
            
            # Save to history (CPU transfer of the whole state)
            history.append(current_state.cpu())
            
        save_results_fast(idx, history, datasets, output_path, node_params)

def save_results_fast(idx, history, datasets, output_dir, node_params):
    # Optimized saving
    first_node = list(datasets.keys())[0]
    target_file = datasets[first_node].files[idx]
    stem = target_file.stem 
    
    # Pre-fetch normalization params for all nodes to avoid dict lookup in loop
    norm_configs = {}
    for nid, params in node_params.items():
        alpha = params['alpha']
        max_map = params['max_values_map']
        max_u = torch.tensor([max_map.get(c, 1.0) for c in ['dx', 'dy', 'dz']])
        max_sigma = torch.tensor([max_map.get(c, 1.0) for c in ['Sxx', 'Syy', 'Szz']])
        max_tau = torch.tensor([max_map.get(c, 1.0) for c in ['Sxy', 'Syz', 'Szx']])
        norm_configs[nid] = (alpha, max_u, max_sigma, max_tau, params['is_fixed'])

    dfs = []
    
    # history is list of Tensors (MaxNode+1, 3, 3)
    active_nodes = sorted(list(node_params.keys()))
    
    for step, state_tensor in enumerate(history):
        time = step + 2
        
        # Bulk convert to numpy for active nodes
        active_states = state_tensor[active_nodes] # (NumActive, 3, 3)
        
        rows = []
        for i, nid in enumerate(active_nodes):
            val = active_states[i] # (3, 3)
            # 0:u, 1:s, 2:t
            
            alpha, max_u, max_s, max_t, is_fixed = norm_configs[nid]
            
            # Denormalize
            if is_fixed:
                u_phys = np.zeros(3)
            else:
                u_phys = oka_denormalize(val[0], max_u, alpha).numpy()
                
            s_phys = oka_denormalize(val[1], max_s, alpha).numpy()
            t_phys = oka_denormalize(val[2], max_t, alpha).numpy()

            row = {
                'time': time,
                'node_id': nid,
                'dx': u_phys[0], 'dy': u_phys[1], 'dz': u_phys[2],
                'Sxx': s_phys[0], 'Syy': s_phys[1], 'Szz': s_phys[2],
                'Sxy': t_phys[0], 'Syz': t_phys[1], 'Szx': t_phys[2]
            }
            rows.append(row)
        if rows:
            dfs.append(pd.DataFrame(rows))
            
    if not dfs: return

    result_df = pd.concat(dfs, ignore_index=True)
    
    # Load correct DF for T=1 and Comparison
    correct_df = pd.read_feather(target_file)
    t1_df = correct_df[correct_df['time'] == 1].copy()
    
    cols = ['time', 'node_id', 'dx', 'dy', 'dz', 'Sxx', 'Syy', 'Szz', 'Sxy', 'Syz', 'Szx', 'x', 'y', 'z']
    common_cols = [c for c in cols if c in t1_df.columns]
    
    full_df = pd.concat([t1_df[common_cols], result_df], ignore_index=True)
    full_df = full_df.sort_values(['node_id', 'time'])
    
    # Reconstruct X, Y, Z recursively
    full_df['cum_dx'] = full_df.groupby('node_id')['dx'].cumsum()
    full_df['cum_dy'] = full_df.groupby('node_id')['dy'].cumsum()
    full_df['cum_dz'] = full_df.groupby('node_id')['dz'].cumsum()
    
    # Use transform('first') to access T=1 initial values efficiently
    init_map = t1_df.set_index('node_id')[['x','y','z']]
    full_df['init_x'] = full_df['node_id'].map(init_map['x'])
    full_df['init_y'] = full_df['node_id'].map(init_map['y'])
    full_df['init_z'] = full_df['node_id'].map(init_map['z'])
    
    first_dx = full_df.groupby('node_id')['dx'].transform('first')
    first_dy = full_df.groupby('node_id')['dy'].transform('first')
    first_dz = full_df.groupby('node_id')['dz'].transform('first')

    full_df['x'] = full_df['init_x'] + full_df['cum_dx'] - first_dx
    full_df['y'] = full_df['init_y'] + full_df['cum_dy'] - first_dy
    full_df['z'] = full_df['init_z'] + full_df['cum_dz'] - first_dz

    merged = pd.merge(full_df, correct_df, on=['time', 'node_id'], suffixes=('', '_correct'), how='left')
    
    for c in ['x','y','z','dx','dy','dz','Sxx','Syy','Szz','Sxy','Syz','Szx']:
         if c in merged.columns and f"{c}_correct" in merged.columns:
             merged[f"{c}_error"] = (merged[c] - merged[f"{c}_correct"]).abs()
             
    save_path = Path(output_dir) / f"{stem}_results.csv"
    merged.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    args = parse_args()
    data = load_models_and_datasets(args)
    if data:
        models, node_params, neighbor_map, fixed_nodes_map, datasets = data
        run_evaluation(models, node_params, neighbor_map, fixed_nodes_map, datasets, args.output_dir)
