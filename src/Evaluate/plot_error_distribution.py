import pandas as pd
import argparse
from pathlib import Path
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

def load_data(result_dir):
    result_dir = Path(result_dir)
    if not result_dir.exists():
        print(f"Directory not found: {result_dir}")
        return None

    csv_files = list(result_dir.glob("*_results.csv"))
    if not csv_files:
        csv_files = [f for f in result_dir.glob("*.csv") if "error_stats" not in f.name and "error_plots" not in str(f) and "hist_data" not in f.name]
    
    if not csv_files:
        print(f"No CSV files found in {result_dir}")
        return None

    print(f"Found {len(csv_files)} CSV files in {result_dir}. Processing...")

    all_dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not all_dfs:
        return None

    combined_df = pd.concat(all_dfs, ignore_index=True)

    if 'time' in combined_df.columns:
        combined_df = combined_df[combined_df['time'] != 1]
    
    return combined_df

def get_group_data(df, cols, op, fixed_nodes, filter_fixed):
    target_df = df.copy()
    if filter_fixed and fixed_nodes:
        if 'node_id' in target_df.columns:
            target_df = target_df[~target_df['node_id'].isin(fixed_nodes)]
    
    existing_cols = [c for c in cols if c in df.columns]
    if not existing_cols:
        return None
        
    data_subset = target_df[existing_cols].dropna()
    if data_subset.empty:
        return None

    if op == "norm":
        return np.sqrt((data_subset ** 2).sum(axis=1))
    elif op == "mean":
        return data_subset.mean(axis=1)
    return None

def main():
    parser = argparse.ArgumentParser(description="Plot error histograms from result CSV files.")
    parser.add_argument("result_dir1", type=str, help="First directory containing result CSV files")
    parser.add_argument("--result_dir2", type=str, default=None, help="Second directory containing result CSV files for comparison")
    parser.add_argument("--output_dir", type=str, default="error_plots", help="Directory to save plots (relative to result_dir1)")
    parser.add_argument("--fixed_nodes_file", type=str, default="/workspace/dataset/liver_model_info/fixed_nodes.csv", help="Path to fixed_nodes.csv")
    parser.add_argument("--log_scale", action="store_true", help="Use log scale for y-axis (frequency)")
    parser.add_argument("--bins", type=int, default=50, help="Number of bins for histogram")
    parser.add_argument("--label1", type=str, default="Dir1", help="Label for first directory")
    parser.add_argument("--label2", type=str, default="Dir2", help="Label for second directory")
    args = parser.parse_args()

    # Load fixed nodes
    fixed_nodes = []
    if args.fixed_nodes_file and os.path.exists(args.fixed_nodes_file):
        try:
            fixed_df = pd.read_csv(args.fixed_nodes_file)
            if 'is_fixed' in fixed_df.columns:
                fixed_nodes = fixed_df[fixed_df['is_fixed'] == True]['node_id'].tolist()
                print(f"Loaded {len(fixed_nodes)} fixed nodes from {args.fixed_nodes_file}")
            else:
                print(f"Warning: 'is_fixed' column not found in {args.fixed_nodes_file}")
        except Exception as e:
            print(f"Error reading fixed nodes file: {e}")
    else:
        print(f"Fixed nodes file not found or not provided: {args.fixed_nodes_file}")

    # Load data
    df1 = load_data(args.result_dir1)
    if df1 is None:
        return

    df2 = None
    if args.result_dir2:
        df2 = load_data(args.result_dir2)
        if df2 is None:
            print("Warning: Could not load data from result_dir2. Proceeding with result_dir1 only.")

    output_dir = Path(args.result_dir1) / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define groups and operations
    groups = {
        "Displacement_delta": {
            "cols": ['dx_error', 'dy_error', 'dz_error'],
            "op": "norm",
            "filter_fixed": True
        },
        "Coordinate": {
            "cols": ['x_error', 'y_error', 'z_error'],
            "op": "norm",
            "filter_fixed": True
        },
        "Normal_Stress": {
            "cols": ['Sxx_error', 'Syy_error', 'Szz_error'],
            "op": "mean",
            "filter_fixed": False
        },
        "Shear_Stress": {
            "cols": ['Sxy_error', 'Syz_error', 'Szx_error'],
            "op": "mean",
            "filter_fixed": False
        }
    }
    
    # Process groups
    processed_cols = set()
    
    for group_name, config in groups.items():
        cols = config["cols"]
        op = config["op"]
        filter_fixed = config["filter_fixed"]

        data1 = get_group_data(df1, cols, op, fixed_nodes, filter_fixed)
        if data1 is None:
            continue
        
        processed_cols.update([c for c in cols if c in df1.columns])

        data2 = None
        if df2 is not None:
            data2 = get_group_data(df2, cols, op, fixed_nodes, filter_fixed)

        # Save frequency data for data1
        counts1, bin_edges1 = np.histogram(data1, bins=args.bins)
        freq_df1 = pd.DataFrame({
            'bin_start': bin_edges1[:-1],
            'bin_end': bin_edges1[1:],
            'count': counts1
        })
        csv_filename1 = f"hist_data_{group_name}_{args.label1}.csv"
        freq_df1.to_csv(output_dir / csv_filename1, index=False)

        plt.figure(figsize=(10, 6))
        
        if data2 is not None:
            # Determine common bins range
            min_val = min(data1.min(), data2.min())
            max_val = max(data1.max(), data2.max())
            bins = np.linspace(min_val, max_val, args.bins)
            
            plt.hist([data1, data2], bins=bins, label=[args.label1, args.label2], density=False, log=args.log_scale)
            plt.legend()
            
            # Save frequency data for data2
            counts2, _ = np.histogram(data2, bins=bins) # Use same bins for consistency if possible, but here we used linspace
            # Re-calculate for CSV using args.bins logic if needed, or just save what we plotted
            # Let's save independent histogram data for CSV
            counts2_csv, bin_edges2_csv = np.histogram(data2, bins=args.bins)
            freq_df2 = pd.DataFrame({
                'bin_start': bin_edges2_csv[:-1],
                'bin_end': bin_edges2_csv[1:],
                'count': counts2_csv
            })
            csv_filename2 = f"hist_data_{group_name}_{args.label2}.csv"
            freq_df2.to_csv(output_dir / csv_filename2, index=False)
            
        else:
            plt.hist(data1, bins=args.bins, color='skyblue', edgecolor='black', log=args.log_scale)
        
        title = f"Error Distribution: {group_name} ({op})"
        if args.log_scale:
            title += " (Log Scale)"
        if filter_fixed and fixed_nodes:
            title += " (Fixed Nodes Excluded)"
            
        plt.title(title)
        plt.xlabel("Error Value")
        plt.ylabel("Frequency")
        plt.grid(True, which="both", ls="-", alpha=0.5)
        
        filename = f"hist_{group_name}.png"
        save_path = output_dir / filename
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot to {save_path}")

    # Process remaining columns individually
    error_cols = [col for col in df1.columns if col.endswith('_error')]
    remaining_cols = [c for c in error_cols if c not in processed_cols]
    
    for col in remaining_cols:
        # For individual columns, we assume no special op, just raw values
        # And we check if it looks like displacement to apply filter
        is_disp = col in {'dx_error', 'dy_error', 'dz_error', 'x_error', 'y_error', 'z_error'}
        
        data1 = get_group_data(df1, [col], "mean", fixed_nodes, is_disp) # "mean" on single col is identity
        if data1 is None:
            continue

        data2 = None
        if df2 is not None:
            data2 = get_group_data(df2, [col], "mean", fixed_nodes, is_disp)

        # Save CSV for data1
        counts1, bin_edges1 = np.histogram(data1, bins=args.bins)
        freq_df1 = pd.DataFrame({'bin_start': bin_edges1[:-1], 'bin_end': bin_edges1[1:], 'count': counts1})
        freq_df1.to_csv(output_dir / f"hist_data_{col}_{args.label1}.csv", index=False)

        plt.figure(figsize=(10, 6))
        if data2 is not None:
            min_val = min(data1.min(), data2.min())
            max_val = max(data1.max(), data2.max())
            bins = np.linspace(min_val, max_val, args.bins)
            
            plt.hist([data1, data2], bins=bins, label=[args.label1, args.label2], log=args.log_scale)
            plt.legend()
            
            counts2, bin_edges2 = np.histogram(data2, bins=args.bins)
            freq_df2 = pd.DataFrame({'bin_start': bin_edges2[:-1], 'bin_end': bin_edges2[1:], 'count': counts2})
            freq_df2.to_csv(output_dir / f"hist_data_{col}_{args.label2}.csv", index=False)
        else:
            plt.hist(data1, bins=args.bins, color='skyblue', edgecolor='black', log=args.log_scale)
        
        title = f"Error Distribution: {col}"
        if args.log_scale:
            title += " (Log Scale)"
        if is_disp and fixed_nodes:
            title += " (Fixed Nodes Excluded)"
            
        plt.title(title)
        plt.xlabel("Error Value")
        plt.ylabel("Frequency")
        plt.grid(True, which="both", ls="-", alpha=0.5)
        
        filename = f"hist_{col}.png"
        save_path = output_dir / filename
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    main()
