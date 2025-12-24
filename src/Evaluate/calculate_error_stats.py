import pandas as pd
import argparse
from pathlib import Path
import glob
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Calculate error statistics from result CSV files.")
    parser.add_argument("result_dir", type=str, help="Directory containing result CSV files")
    parser.add_argument("--output", type=str, default="error_stats_summary.csv", help="Output summary CSV file name")
    parser.add_argument("--fixed_nodes_file", type=str, default="/workspace/dataset/liver_model_info/fixed_nodes.csv", help="Path to fixed_nodes.csv")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        print(f"Directory not found: {result_dir}")
        return

    # Load fixed nodes
    fixed_nodes = []
    if args.fixed_nodes_file and os.path.exists(args.fixed_nodes_file):
        try:
            fixed_df = pd.read_csv(args.fixed_nodes_file)
            # Assuming 'is_fixed' is boolean or string 'True'
            if 'is_fixed' in fixed_df.columns:
                fixed_nodes = fixed_df[fixed_df['is_fixed'] == True]['node_id'].tolist()
                print(f"Loaded {len(fixed_nodes)} fixed nodes from {args.fixed_nodes_file}")
            else:
                print(f"Warning: 'is_fixed' column not found in {args.fixed_nodes_file}")
        except Exception as e:
            print(f"Error reading fixed nodes file: {e}")
    else:
        print(f"Fixed nodes file not found or not provided: {args.fixed_nodes_file}")

    # Find CSV files
    csv_files = list(result_dir.glob("*_results.csv"))
    if not csv_files:
        # Fallback to all csvs if no _results pattern found, but exclude stats files to avoid recursion
        csv_files = [f for f in result_dir.glob("*.csv") if "error_stats" not in f.name]
    
    if not csv_files:
        print("No CSV files found in the directory.")
        return

    print(f"Found {len(csv_files)} CSV files. Processing...")

    all_dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not all_dfs:
        print("No valid data loaded.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Filter out time == 1
    if 'time' in combined_df.columns:
        print("Excluding time == 1 data...")
        combined_df = combined_df[combined_df['time'] != 1]

    # Calculate d_error (Euclidean norm of displacement errors)
    if {'dx_error', 'dy_error', 'dz_error'}.issubset(combined_df.columns):
        print("Calculating d_error (Euclidean norm of dx, dy, dz errors)...")
        combined_df['d_error'] = np.sqrt(
            combined_df['dx_error']**2 + 
            combined_df['dy_error']**2 + 
            combined_df['dz_error']**2
        )

    # Calculate d_cumsum_error (Euclidean norm of x, y, z errors at time 20)
    if {'x_error', 'y_error', 'z_error'}.issubset(combined_df.columns):
        print("Calcul ating d_cumsum_error (Euclidean norm of x, y, z errors at time 20)...")
        d_coord_error = np.sqrt(
            combined_df['x_error']**2 + 
            combined_df['y_error']**2 + 
            combined_df['z_error']**2
        )
        combined_df['d_cumsum_error'] = d_coord_error.where(combined_df['time'] == 20)

    # Identify error columns
    error_cols = [col for col in combined_df.columns if col.endswith('_error')]
    
    if not error_cols:
        print("No columns ending with '_error' found.")
        return

    print(f"Calculating statistics for columns: {error_cols}")

    # Overall Statistics
    stats_dict = {}
    disp_error_cols = {'dx_error', 'dy_error', 'dz_error', 'x_error', 'y_error', 'z_error', 'd_error', 'd_cumsum_error'}

    for col in error_cols:
        target_df = combined_df
        # Apply fixed node filtering for displacement related errors
        if fixed_nodes and col in disp_error_cols:
            if 'node_id' in target_df.columns:
                target_df = target_df[~target_df['node_id'].isin(fixed_nodes)]
            else:
                print(f"Warning: 'node_id' column missing, cannot filter fixed nodes for {col}")
        
        stats_dict[col] = {
            'mean': target_df[col].mean(),
            'max': target_df[col].max(),
            'std': target_df[col].std(),
            'median': target_df[col].median()
        }

    print("\n=== Overall Statistics ===")
    stats_df = pd.DataFrame(stats_dict).T
    print(stats_df)

    # Save overall stats
    output_path = result_dir / args.output
    stats_df.to_csv(output_path)
    print(f"\nOverall statistics saved to {output_path}")

    # Per Time Step Statistics
    if 'time' in combined_df.columns:
        print("\n=== Per Time Step Statistics ===")
        
        # We need to calculate per-time stats column by column to handle filtering
        time_stats_list = []
        
        for col in error_cols:
            target_df = combined_df
            if fixed_nodes and col in disp_error_cols:
                if 'node_id' in target_df.columns:
                    target_df = target_df[~target_df['node_id'].isin(fixed_nodes)]
            
            grouped = target_df.groupby('time')[col]
            t_mean = grouped.mean().rename(f"{col}_mean")
            t_max = grouped.max().rename(f"{col}_max")
            t_std = grouped.std().rename(f"{col}_std")
            
            time_stats_list.extend([t_mean, t_max, t_std])
            
        time_stats = pd.concat(time_stats_list, axis=1)
        
        # Save per time stats
        time_stats_file = result_dir / "error_stats_per_time.csv"
        
        # Sort columns to keep metrics for same variable together
        sorted_cols = sorted(time_stats.columns)
        time_stats = time_stats[sorted_cols]
        
        time_stats.to_csv(time_stats_file)
        print(f"Per time statistics saved to {time_stats_file}")
        
        # Display a preview of per-time stats (e.g., for the first error column)
        first_err_col = error_cols[0]
        print(f"\nPreview for {first_err_col} (Mean/Max):")
        print(time_stats[[f"{first_err_col}_mean", f"{first_err_col}_max"]].head())
    else:
        print("'time' column not found, skipping per-time statistics.")

if __name__ == "__main__":
    main()
