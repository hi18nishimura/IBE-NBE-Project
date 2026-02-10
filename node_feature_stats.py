import argparse
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

def oka_normalize_dataframe_fast(df: pd.DataFrame, pwidths: np.ndarray, alpha: float) -> pd.DataFrame:
    """
    Oka normalizationをNumPyのベクトル演算でDataFrame全体に適用する。

    :param df: 処理対象のDataFrame
    :param pwidths: 各列に対応するpwidthパラメータ (形状: (1, N_cols) または (N_rows, N_cols))
    :param alpha: スケーリングパラメータ
    :return: 正規化されたDataFrame
    """
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    # 1. データとパラメータをNumPy配列に変換
    vals = df.values
    
    # Ensure pwidths is at least 2D for broadcasting consistency with original implementation assumptions if needed
    # But usually numpy handles (N,) broadcasting to (M, N) fine. 
    # The original implementation explicitly used [0, :] indexing assuming (1, N).
    
    if pwidths.ndim == 1:
        pwidths = pwidths.reshape(1, -1)

    # 2. ゼロ除算やNaN除外の条件処理 (ブロードキャストを利用)
    # 処理対象となる列のインデックスを特定 (pwidth > 0 かつ 有限値)
    valid_cols_mask = (pwidths > 0) & np.isfinite(pwidths)
    
    # 全ての列が無効な場合は、0.5で埋めて返す
    if not np.any(valid_cols_mask):
        return pd.DataFrame(np.full_like(vals, 0.5, dtype=float), index=df.index, columns=df.columns)

    # 3. ゼロ除算を避けるため、NaN/infを無視する設定で演算
    with np.errstate(divide='ignore', invalid='ignore'):
        # 4. 要素ごとのサインと絶対値を取得
        signs = np.sign(vals)
        absvals = np.abs(vals)
        
        # 5. ファクターとべき乗の計算 (ブロードキャストが自動的に適用される)
        power_term = 1.0 / alpha
        
        # pwidthsが有効な列でのみ計算を行う
        # invalid_cols (pwidth<=0) の factor は NaN となり、その列の結果も NaN になるため、
        # 後で 0.5 でマスクする。
        
        # factor = 0.4 / (pwidth ** (1.0 / alpha))
        factor = np.full_like(pwidths, np.nan, dtype=float)
        
        # Assuming pwidths is (1, N_cols) basically
        if pwidths.shape[0] == 1:
             factor[:, valid_cols_mask[0, :]] = 0.4 / (pwidths[:, valid_cols_mask[0, :]] ** power_term)
        else:
             # Fallback if pwidths matches rows
             factor[valid_cols_mask] = 0.4 / (pwidths[valid_cols_mask] ** power_term)
        
        # normed = signs * factor * (absvals ** (1.0 / alpha)) + 0.5
        normed = signs * factor * (absvals ** power_term) + 0.5

    # 6. pwidth <= 0 (または NaN) の列を 0.5 で置き換える (元の関数の動作を再現)
    # np.where(条件, 真の場合の値, 偽の場合の値)
    
    # Broadcast mask if necessary
    if valid_cols_mask.shape != vals.shape:
        valid_cols_mask = np.broadcast_to(valid_cols_mask, vals.shape)

    result_vals = np.where(~valid_cols_mask, 0.5, normed)

    return pd.DataFrame(result_vals, index=df.index, columns=df.columns)

def main():
    parser = argparse.ArgumentParser(description='Analyze feature distribution for a specific node_id from feather files.')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the directory containing feather files.')
    parser.add_argument('--node_id', type=str, required=True, help='Target node_id to filter. Use "all" to include all nodes.')
    parser.add_argument('--output_plot', type=str, default='feature_histograms.png', help='Path to save the histogram plot.')
    parser.add_argument('--summary_file', type=str, default='/workspace/dataset/bin/toy_all_model/train/summary_overall_max_values.csv', help='Path to summary max values csv.')
    parser.add_argument('--alpha', type=float, default=8.0, help='Alpha for Oka normalization.')
    parser.add_argument('--output_plot_norm', type=str, default='feature_histograms_norm.png', help='Path to save the normalized histogram plot.')
    args = parser.parse_args()

    input_dir = args.input_dir
    
    target_node_id = None
    if args.node_id.lower() != 'all':
        try:
            target_node_id = int(args.node_id)
        except ValueError:
            print(f"Error: node_id must be an integer or 'all'. Got '{args.node_id}'")
            return
    else:
        print("Processing all nodes...")

    
    # Load max map for normalization
    max_map = {}
    if os.path.exists(args.summary_file):
        try:
            df_max = pd.read_csv(args.summary_file)
            # Use the global maximum for each feature across all nodes contained in the summary file
            # This handles both overall summary (1 row per feature) and per-node summary (multiple rows per feature)
            if 'max_value' in df_max.columns and 'feature' in df_max.columns:
                 # Group by feature and find the maximum max_value to ensure we use the global max
                 # regardless of whether the file is per-node or overall.
                 max_stats = df_max.groupby('feature')['max_value'].max()
                 max_map = max_stats.to_dict()
                 print(f"Loaded global max values from {args.summary_file}")
            else:
                 print(f"Warning: Expected columns 'feature' and 'max_value' not found in {args.summary_file}.")
        except Exception as e:
            print(f"Error loading summary file: {e}")
    else:
        print(f"Warning: Summary file {args.summary_file} not found. Normalization might fail or be incorrect.")

    # Target columns to analyze
    target_columns = ['dx', 'dy', 'dz', 'Sxx', 'Syy', 'Szz', 'Sxy', 'Syz', 'Szx']
    
    # Find all feather files in the specified directory
    files = glob.glob(os.path.join(input_dir, '*.feather'))
    
    if not files:
        print(f"No feather files found in {input_dir}")
        return

    print(f"Found {len(files)} feather files in {input_dir}. Processing...")
    
    data_list = []
    for f in tqdm(files):
        try:
            # Read only necessary columns if possible to save memory? 
            # pd.read_feather supports columns argument but we need to filter by node_id first.
            # If node_id is not in target_columns, we need it too.
            # Usually read_feather is fast. Reading full file then filtering.
            
            df = pd.read_feather(f)
            
            if 'node_id' not in df.columns:
                print(f"Warning: 'node_id' column not found in {os.path.basename(f)}. Skipping.")
                continue
            
            # Filter by node_id if specific node_id is requested
            if target_node_id is not None:
                filtered_df = df[df['node_id'] == target_node_id]
            else:
                filtered_df = df
            
            if filtered_df.empty:
                continue

            # Select only target columns that exist in the dataframe
            existing_cols = [col for col in target_columns if col in filtered_df.columns]
            
            if not existing_cols:
                continue
                
            data_list.append(filtered_df[existing_cols])
            
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not data_list:
        if target_node_id is not None:
             print(f"No data found for node_id {target_node_id}.")
        else:
             print(f"No data found in {input_dir}.")
        return

    # Concatenate all data
    full_df = pd.concat(data_list, ignore_index=True)
    
    if full_df.empty:
        if target_node_id is not None:
            print(f"No data collected for node_id {target_node_id}.")
        else:
            print(f"No data collected.")
        return

    print(f"\nFeature distribution for node_id: {args.node_id}")
    print(f"Total samples: {len(full_df)}")
    print("-" * 50)
    
    # Calculate statistics
    stats = full_df.describe()

    print(stats)
    
    print("-" * 50)
    # Check for missing columns
    missing_cols = set(target_columns) - set(full_df.columns)
    if missing_cols:
        print(f"Warning: The following columns were not found in the data: {missing_cols}")

    # Plot histograms
    num_cols = len(full_df.columns)
    if num_cols > 0:
        # Automatic grid size calculation
        n_cols_plot = 3
        n_rows_plot = math.ceil(num_cols / n_cols_plot)

        fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(15, 5 * n_rows_plot))
        
        # Ensure axes is iterable even if only one plot
        if n_rows_plot * n_cols_plot == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # Calculate max frequency for shared y-axis
        max_freq = 0
        for col in full_df.columns:
            counts, _ = np.histogram(full_df[col].dropna(), bins=50)
            if len(counts) > 0:
                max_freq = max(max_freq, counts.max())
        
        # Add a small margin
        ylim_max = max_freq * 1.05

        for i, col in enumerate(full_df.columns):
            ax = axes[i]
            full_df[col].hist(ax=ax, bins=50)
            ax.set_title(col)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            if ylim_max > 0:
                ax.set_ylim(0, ylim_max)

        # Hide empty subplots
        for j in range(num_cols, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(args.output_plot)
        print(f"Histograms saved to {args.output_plot}")

    # Normalize and Plot Normalized Histograms
    if not full_df.empty:
        norm_df = full_df.copy()
        
        # Construct pwidths for the columns present in full_df
        # Use a default of 0.0 if not found, which results in 0.5 masked value
        pwidths_list = [max_map.get(col, 0.0) for col in full_df.columns]
        pwidths = np.array(pwidths_list)
        
        # Apply normalization
        norm_df = oka_normalize_dataframe_fast(norm_df, pwidths, args.alpha)

        print(f"\nNormalized Feature distribution stats:")
        print(norm_df.describe())

        # Plot Normalized Histograms
        if num_cols > 0:
            fig_norm, axes_norm = plt.subplots(n_rows_plot, n_cols_plot, figsize=(15, 5 * n_rows_plot))
            
            if n_rows_plot * n_cols_plot == 1:
                axes_norm = [axes_norm]
            else:
                axes_norm = axes_norm.flatten()

            for i, col in enumerate(norm_df.columns):
                ax = axes_norm[i]
                norm_df[col].hist(ax=ax, bins=50)
                ax.set_title(col + ' (Normalized)')
                ax.set_xlabel('Normalized Value (Should be around 0.5)')
                ax.set_ylabel('Frequency')

            # Hide empty subplots
            for j in range(num_cols, len(axes_norm)):
                axes_norm[j].axis('off')

            plt.tight_layout()
            plt.savefig(args.output_plot_norm)
            print(f"Normalized histograms saved to {args.output_plot_norm}")

if __name__ == '__main__':
    main()
