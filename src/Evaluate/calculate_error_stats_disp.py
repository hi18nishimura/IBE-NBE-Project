import pandas as pd
import argparse
from pathlib import Path
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

def load_and_filter_data(result_dir_path, args, fixed_nodes):
    result_dir = Path(result_dir_path)
    if not result_dir.exists():
        print(f"Directory not found: {result_dir}")
        return None

    # Find CSV files
    csv_files = list(result_dir.glob("*_results.csv"))
    if not csv_files:
        csv_files = [f for f in result_dir.glob("*.csv") if "error_stats" not in f.name]
    
    if not csv_files:
        print(f"No CSV files found in {result_dir}.")
        return None

    print(f"Found {len(csv_files)} CSV files in {result_dir}. Processing...")

    gt_dir = Path("/workspace/dataset/bin/toy_all_model/test")
    all_filtered_dfs = []
    
    for f in csv_files:
        try:
            # Load result CSV
            df = pd.read_csv(f)
            
            # Construct feather path
            feather_name = f.name.replace("_results.csv", ".feather")
            feather_path = gt_dir / feather_name
            
            if not feather_path.exists():
                # print(f"Warning: Ground truth file not found: {feather_path}")
                continue
                
            # Load feather
            gt_df = pd.read_feather(feather_path)
            
            # Calculate displacement magnitude in GT
            gt_df['d_mag'] = np.sqrt(
                gt_df['dx']**2 + 
                gt_df['dy']**2 + 
                gt_df['dz']**2
            )
            
            # Filter GT by threshold
            gt_filtered = gt_df[gt_df['d_mag'] > args.threshold]
            
            if gt_filtered.empty:
                continue
                
            # Create a set of (time, node_id) keys from filtered GT
            keys = set(zip(gt_filtered['time'], gt_filtered['node_id']))
            
            # Filter result df using these keys
            df_temp = df.set_index(['time', 'node_id'])
            filtered_rows = df_temp.index.isin(keys)
            df_filtered = df[filtered_rows].copy()
            
            if not df_filtered.empty:
                all_filtered_dfs.append(df_filtered)
                
        except Exception as e:
            print(f"Error processing {f}: {e}")

    if not all_filtered_dfs:
        print(f"No valid data loaded or no data matched the threshold criteria in {result_dir}.")
        return None

    combined_df = pd.concat(all_filtered_dfs, ignore_index=True)

    # Filter out time == 1
    if 'time' in combined_df.columns:
        combined_df = combined_df[combined_df['time'] != 1]

    # Check if required columns exist
    required_error_cols = {'dx_error', 'dy_error', 'dz_error'}
        
    if not required_error_cols.issubset(combined_df.columns):
        print(f"Error: Missing error columns {required_error_cols - set(combined_df.columns)}")
        return None

    filtered_df = combined_df
    
    # Filter fixed nodes if applicable
    if fixed_nodes and 'node_id' in filtered_df.columns:
        filtered_df = filtered_df[~filtered_df['node_id'].isin(fixed_nodes)]
    
    if filtered_df.empty:
        return None

    # Calculate d_error for filtered data
    filtered_df['d_error'] = np.sqrt(
        filtered_df['dx_error']**2 + 
        filtered_df['dy_error']**2 + 
        filtered_df['dz_error']**2
    )
    
    return filtered_df

def main():
    parser = argparse.ArgumentParser(description="Calculate error statistics for large displacement nodes.")
    parser.add_argument("result_dir", type=str, help="Directory containing result CSV files")
    parser.add_argument("--result_dir2", type=str, default=None, help="Second directory for comparison")
    parser.add_argument("--output", type=str, default="error_stats_disp_summary.csv", help="Output summary CSV file name")
    parser.add_argument("--threshold", type=float, default=1.0, help="Displacement threshold (default: 1.0)")
    parser.add_argument("--fixed_nodes_file", type=str, default="/workspace/dataset/liver_model_info/fixed_nodes.csv", help="Path to fixed_nodes.csv")
    parser.add_argument("--plot", action="store_true", help="Plot error distribution")
    parser.add_argument("--plot_output", type=str, default="error_dist_disp.png", help="Output plot file name")
    parser.add_argument("--log_scale", action="store_true", help="Use log scale for y-axis")
    parser.add_argument("--bins", type=int, default=50, help="Number of bins")
    parser.add_argument("--label1", type=str, default="Dir1", help="Label for first directory")
    parser.add_argument("--label2", type=str, default="Dir2", help="Label for second directory")
    parser.add_argument("--x_max", type=float, default=None, help="Maximum value for x-axis")
    parser.add_argument("--log_x", action="store_true", help="Use log scale for x-axis")
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

    # Process first directory
    filtered_df = load_and_filter_data(args.result_dir, args, fixed_nodes)
    if filtered_df is None:
        return

    print(f"Filtered data size (Dir1): {len(filtered_df)} rows")

    # Process second directory if provided
    filtered_df2 = None
    if args.result_dir2:
        print(f"\nProcessing second directory: {args.result_dir2}")
        filtered_df2 = load_and_filter_data(args.result_dir2, args, fixed_nodes)
        if filtered_df2 is not None:
            print(f"Filtered data size (Dir2): {len(filtered_df2)} rows")

    # Calculate statistics for Dir1
    stats_dict = {
        'd_error': {
            'mean': filtered_df['d_error'].mean(),
            'max': filtered_df['d_error'].max(),
            'std': filtered_df['d_error'].std(),
            'median': filtered_df['d_error'].median(),
            'count': len(filtered_df)
        }
    }

    print("\n=== Overall Statistics (Dir1) ===")
    stats_df = pd.DataFrame(stats_dict).T
    print(stats_df)

    # Save overall stats for Dir1
    result_dir = Path(args.result_dir)
    output_path = result_dir / args.output
    stats_df.to_csv(output_path)
    print(f"\nOverall statistics saved to {output_path}")

    # Per Time Step Statistics for Dir1
    if 'time' in filtered_df.columns:
        print("\n=== Per Time Step Statistics (Dir1) ===")
        
        grouped = filtered_df.groupby('time')['d_error']
        t_mean = grouped.mean().rename("d_error_mean")
        t_max = grouped.max().rename("d_error_max")
        t_std = grouped.std().rename("d_error_std")
        t_count = grouped.count().rename("count")
        
        time_stats = pd.concat([t_mean, t_max, t_std, t_count], axis=1)
        
        # Save per time stats
        time_stats_file = result_dir / "error_stats_disp_per_time.csv"
        time_stats.to_csv(time_stats_file)
        print(f"Per time statistics saved to {time_stats_file}")
        
        print("\nPreview:")
        print(time_stats.head())

    # Plot error distribution
    if args.plot:
        plot_path = result_dir / args.plot_output
        print(f"\nPlotting error distribution to {plot_path}...")
        
        plt.figure(figsize=(10, 6))
        
        data_list = []
        labels = []
        
        data_list.append(filtered_df['d_error'])
        labels.append(args.label1)
        
        if filtered_df2 is not None:
            data_list.append(filtered_df2['d_error'])
            labels.append(args.label2)
            
        # Determine bins
        all_data = pd.concat(data_list)
        min_val = all_data.min()
        max_val = all_data.max()
        
        if args.x_max:
            max_val = args.x_max
        
        if args.log_x:
            if min_val <= 0:
                min_val = 1e-6 # Avoid log(0)
            bins = np.logspace(np.log10(min_val), np.log10(max_val), args.bins)
            plt.xscale('log')
        else:
            bins = np.linspace(min_val, max_val, args.bins)
            
        if len(data_list) > 1:
            for i, data in enumerate(data_list):
                plt.hist(data, bins=bins, label=labels[i], log=args.log_scale, alpha=0.5)
            plt.legend()
        else:
            plt.hist(data_list[0], bins=bins, label=labels[0], log=args.log_scale, edgecolor='black')
            
        # Save histogram data
        for i, data in enumerate(data_list):
            counts, _ = np.histogram(data, bins=bins)
            hist_df = pd.DataFrame({
                'bin_start': bins[:-1],
                'bin_end': bins[1:],
                'count': counts
            })
            label_safe = labels[i].replace(" ", "_")
            hist_csv_path = result_dir / f"hist_data_disp_{args.threshold}_{label_safe}.csv"
            hist_df.to_csv(hist_csv_path, index=False)
            print(f"Histogram data saved to {hist_csv_path}")
        
        title = f"Error Distribution (Displacement > {args.threshold})"
        if args.log_scale:
            title += " (Log Y)"
        if args.log_x:
            title += " (Log X)"
            
        plt.title(title)
        plt.xlabel("d_error")
        plt.ylabel("Frequency")
        plt.grid(True, which="both", ls="-", alpha=0.5)
        
        if args.x_max:
            plt.xlim(right=args.x_max)
        
        plt.savefig(plot_path)
        plt.close()
        print("Plot saved.")

if __name__ == "__main__":
    main()
