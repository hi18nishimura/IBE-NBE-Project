import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def calculate_error_stats(input_path, output_dir):
    """
    Calculate error statistics from a CSV file or a directory of CSV files containing error data.
    
    Args:
        input_path (str): Path to the input CSV file or directory.
        output_dir (str): Directory to save the output CSV files.
    """
    if os.path.isdir(input_path):
        import glob
        all_files = glob.glob(os.path.join(input_path, "*.csv"))
        if not all_files:
            print(f"No CSV files found in directory: {input_path}")
            return
        
        print(f"Found {len(all_files)} CSV files in {input_path}. Loading...")
        df_list = []
        for filename in tqdm(all_files, desc="Loading CSV files"):
            try:
                temp_df = pd.read_csv(filename)
                df_list.append(temp_df)
            except Exception as e:
                print(f"Error reading CSV file {filename}: {e}")
        
        if not df_list:
            print("No valid data loaded.")
            return
            
        df = pd.concat(df_list, ignore_index=True)
        print(f"Combined data shape: {df.shape}")

    elif os.path.isfile(input_path):
        print(f"Loading data from {input_path}...")
        try:
            df = pd.read_csv(input_path)
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return
    else:
        print(f"Error: Input path '{input_path}' not found.")
        return

    # Base variables
    base_vars = {
        'position': ['x', 'y', 'z'],
        'displacement': ['dx', 'dy', 'dz'],
        'stress_diag': ['Sxx', 'Syy', 'Szz'],
        'stress_offdiag': ['Sxy', 'Syz', 'Szx']
    }

    # Collect all error columns available in the dataframe
    # We look for columns ending with '_error'
    error_columns = [col for col in df.columns if col.endswith('_error')]
    
    # Calculate requested composite errors
    
    # 1. Euclidean error for x, y, z
    if all(f"{v}_error" in df.columns for v in base_vars['position']):
        df['position_euclidean_error'] = np.sqrt(
            df['x_error']**2 + df['y_error']**2 + df['z_error']**2
        )
        error_columns.append('position_euclidean_error')

    # 2. Euclidean error for dx, dy, dz
    if all(f"{v}_error" in df.columns for v in base_vars['displacement']):
        df['displacement_euclidean_error'] = np.sqrt(
            df['dx_error']**2 + df['dy_error']**2 + df['dz_error']**2
        )
        error_columns.append('displacement_euclidean_error')

    # 3. RSS for Sxx, Syy, Szz
    if all(f"{v}_error" in df.columns for v in base_vars['stress_diag']):
        df['stress_diag_rss_error'] = np.sqrt(
            df['Sxx_error']**2 + df['Syy_error']**2 + df['Szz_error']**2
        )
        error_columns.append('stress_diag_rss_error')

    # 4. RSS for Sxy, Syz, Szx
    if all(f"{v}_error" in df.columns for v in base_vars['stress_offdiag']):
        df['stress_offdiag_rss_error'] = np.sqrt(
            df['Sxy_error']**2 + df['Syz_error']**2 + df['Szx_error']**2
        )
        error_columns.append('stress_offdiag_rss_error')

    if not error_columns:
        print("No error columns found to analyze.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # --- Global Statistics (All time) ---
    # Average and Maximum
    global_mean = df[error_columns].mean()
    global_max = df[error_columns].max()
    
    global_stats = pd.DataFrame({
        'mean': global_mean, 
        'max': global_max
    })
    
    global_stats_path = os.path.join(output_dir, 'global_error_stats.csv')
    global_stats.to_csv(global_stats_path)
    print(f"Global stats saved to {global_stats_path}")

    # --- Temporal Statistics (Per time) ---
    if 'time' in df.columns:
        # Group by time and calculate mean/max across all nodes at that time
        temporal_grouped = df.groupby('time')[error_columns]
        
        temporal_mean = temporal_grouped.mean()
        temporal_max = temporal_grouped.max()

        temporal_mean_path = os.path.join(output_dir, 'temporal_mean_error.csv')
        temporal_max_path = os.path.join(output_dir, 'temporal_max_error.csv')

        temporal_mean.to_csv(temporal_mean_path)
        temporal_max.to_csv(temporal_max_path)
        
        print(f"Temporal mean stats saved to {temporal_mean_path}")
        print(f"Temporal max stats saved to {temporal_max_path}")
    else:
        print("Column 'time' not present. Skipping temporal statistics.")

    # --- Nodal Statistics (Per node) ---
    node_col = 'node_id'
    if node_col in df.columns:
        # Group by node_id and calculate mean/max across all time steps for that node
        nodal_grouped = df.groupby(node_col)[error_columns]
        
        nodal_mean = nodal_grouped.mean()
        nodal_max = nodal_grouped.max()

        # Custom logic: Restrict position errors to time=20
        if 'time' in df.columns:
            target_time = 20
            # Try exact match first
            mask = df['time'] == target_time
            if not mask.any() and np.issubdtype(df['time'].dtype, np.floating):
                 mask = np.isclose(df['time'], target_time)
            
            df_time = df[mask]
            
            if not df_time.empty:
                pos_cols = [c for c in ['x_error', 'y_error', 'z_error', 'position_euclidean_error'] if c in nodal_mean.columns]
                if pos_cols:
                    print(f"Recalculating nodal mean for {pos_cols} using only time={target_time}...")
                    nodal_mean_time = df_time.groupby(node_col)[pos_cols].mean()
                    nodal_mean.update(nodal_mean_time)

        nodal_mean_path = os.path.join(output_dir, 'nodal_mean_error.csv')
        nodal_max_path = os.path.join(output_dir, 'nodal_max_error.csv')

        nodal_mean.to_csv(nodal_mean_path)
        nodal_max.to_csv(nodal_max_path)
        
        print(f"Nodal mean stats saved to {nodal_mean_path}")
        print(f"Nodal max stats saved to {nodal_max_path}")
    else:
        print(f"Column '{node_col}' not present. Skipping nodal statistics.")

def main():
    parser = argparse.ArgumentParser(description='Calculate error statistics.')
    parser.add_argument('input_path', type=str, help='Path to the input CSV file or directory containing results.')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save the output stats.')
    
    args = parser.parse_args()
    
    input_path = args.input_path
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Default output directory: same directory as input file/folder, folder named 'error_statistics'
        if os.path.isdir(input_path):
             base_dir = input_path
        else:
             base_dir = os.path.dirname(os.path.abspath(input_path))
        
        output_dir = os.path.join(base_dir, 'error_statistics')
        
    calculate_error_stats(input_path, output_dir)

if __name__ == '__main__':
    main()
