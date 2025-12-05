import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer
import seaborn as sns

def oka_normalize_series(series: pd.Series, pwidth: float, alpha: float = 1.2) -> pd.Series:
    """Oka の式に従って series を正規化して返す。
    p_out = sign(p_in) * (0.4 / pwidth^(1/alpha)) * |p_in|^(1/alpha) + 0.5
    """
    if alpha <= 0:
        raise ValueError("alpha は正の値を指定してください")
    vals = series.to_numpy(dtype=float)
    signs = np.sign(vals)
    absvals = np.abs(vals)
    # pwidth が 0 のときはすべて 0.5 とする（情報がないため中央に置く）
    if pwidth == 0 or np.isnan(pwidth):
        return pd.Series(np.full_like(vals, 0.5, dtype=float), index=series.index)
    factor = 0.4 / (pwidth ** (1.0 / alpha))
    # avoid negative/NaN issues: keep NaN as NaN
    with np.errstate(invalid='ignore'):
        normed = signs * factor * (absvals ** (1.0 / alpha)) + 0.5
    return pd.Series(normed, index=series.index)

def main():
    parser = argparse.ArgumentParser(description='Plot correlation between force and displacement/stress.')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing feather files or path to a feather file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the plots.')
    parser.add_argument('--sample_size', type=int, default=100000, help='Number of samples to plot to speed up rendering. Set to 0 to plot all data.')
    parser.add_argument('--oka_normalize', action='store_true', help='Apply Oka normalization before plotting and PCA.')
    parser.add_argument('--yeo-johnson', action='store_true', help='Apply Yeo-Johnson normalization before plotting and PCA.')
    parser.add_argument('--node_id', type=int, help='Specific node_id to analyze. If not specified, all nodes are used.')
    args = parser.parse_args()

    input_path = args.input_dir
    output_dir = args.output_dir
    sample_size = args.sample_size
    use_oka_normalize = args.oka_normalize
    use_yeo_johnson = args.yeo_johnson
    target_node_id = args.node_id

    os.makedirs(output_dir, exist_ok=True)

    df = None

    # Load data
    if os.path.isfile(input_path) and input_path.endswith('.feather'):
        print(f"Reading feather file: {input_path}")
        df = pd.read_feather(input_path)
    elif os.path.isdir(input_path):
        print(f"Reading feather files from directory: {input_path}")
        feather_files = glob.glob(os.path.join(input_path, "*.feather"))
        if not feather_files:
            print("No feather files found in the directory.")
            return
        
        dfs = []
        for f in tqdm(feather_files, desc="Reading files"):
            try:
                sub_df = pd.read_feather(f)
                
                # Extract displacement of the force node
                if 'force_node_id' in sub_df.columns and 'node_id' in sub_df.columns and 'time' in sub_df.columns:
                    # Find rows corresponding to the force node
                    # Assuming force_node_id indicates the node where force is applied
                    force_node_rows = sub_df[sub_df['node_id'] == sub_df['force_node_id']]
                    
                    if not force_node_rows.empty:
                        # Extract time, displacement and position columns
                        cols_to_extract = ['time', 'dx', 'dy', 'dz']
                        if all(c in sub_df.columns for c in ['x', 'y', 'z']):
                            cols_to_extract.extend(['x', 'y', 'z'])

                        force_data = force_node_rows[cols_to_extract].copy()
                        rename_dict = {
                            'dx': 'force_node_dx',
                            'dy': 'force_node_dy',
                            'dz': 'force_node_dz',
                            'x': 'force_node_x',
                            'y': 'force_node_y',
                            'z': 'force_node_z'
                        }
                        force_data = force_data.rename(columns=rename_dict)
                        
                        # Remove duplicates if multiple rows exist for same time (should not happen in standard format)
                        force_data = force_data.drop_duplicates(subset=['time'])
                        
                        # Merge back to the dataframe based on time
                        sub_df = pd.merge(sub_df, force_data, on='time', how='left')
                
                dfs.append(sub_df)
            except Exception as e:
                print(f"Error reading {f}: {e}")
        
        if not dfs:
            print("No valid feather files loaded.")
            return

        df = pd.concat(dfs, ignore_index=True)
        
        # Save combined dataframe
        combined_path = os.path.join(output_dir, 'combined_data.feather')
        print(f"Saving combined dataframe to: {combined_path}")
        df.to_feather(combined_path)
    else:
        print("Invalid input path. Must be a feather file or a directory containing feather files.")
        return

    if df is None or df.empty:
        print("DataFrame is empty.")
        return

    # Filter by node_id if specified
    if target_node_id is not None:
        if 'node_id' in df.columns:
            print(f"Filtering data for node_id: {target_node_id}")
            df = df[df['node_id'] == target_node_id]
            if df.empty:
                print(f"No data found for node_id: {target_node_id}")
                return
        else:
            print("Warning: 'node_id' column not found in the dataset. Ignoring --node_id argument.")

    # Calculate distance to force node if node_id is specified and coordinates are available
    if target_node_id is not None and all(col in df.columns for col in ['x', 'y', 'z', 'force_node_x', 'force_node_y', 'force_node_z']):
        print("Calculating distance to force node...")
        df['distance_to_force_node'] = np.sqrt(
            (df['x'] - df['force_node_x'])**2 +
            (df['y'] - df['force_node_y'])**2 +
            (df['z'] - df['force_node_z'])**2
        )

    # Sampling to speed up plotting
    if sample_size > 0 and len(df) > sample_size:
        print(f"Sampling {sample_size} points from {len(df)} total points to speed up plotting...")
        df = df.sample(n=sample_size, random_state=42)

    force_cols = ['force_node_dx', 'force_node_dy', 'force_node_dz']
    displacement_cols = ['dx', 'dy', 'dz']
    stress_cols = ['Sxx', 'Syy', 'Szz', 'Sxy', 'Syz', 'Szx']

    # Check if columns exist
    available_force = [col for col in force_cols if col in df.columns]
    available_disp = [col for col in displacement_cols if col in df.columns]
    available_stress = [col for col in stress_cols if col in df.columns]

    if not available_force:
        print("Missing force node displacement columns (force_node_dx, force_node_dy, force_node_dz).")
        print("Make sure 'force_node_id', 'node_id', and 'time' columns exist in the input data.")
        return

    if not available_disp and not available_stress:
        print("Missing displacement and stress columns.")
        return

    target_cols = available_disp + available_stress

    # Apply absolute value to all features
    print("Applying absolute value to all features...")
    features_to_abs = available_force + target_cols
    df[features_to_abs] = df[features_to_abs].abs()

    # Apply Oka normalization if requested
    if use_oka_normalize:
        print("Applying Oka normalization...")
        features_to_normalize = available_force + target_cols
        for col in features_to_normalize:
            # pwidth is the max absolute value of the column
            pwidth = df[col].abs().max()
            if pwidth == 0:
                pwidth = 1.0 # Avoid division by zero if max is 0
            df[col] = oka_normalize_series(df[col], pwidth=pwidth)
        
        # Update output directory to indicate normalization
        output_dir = os.path.join(output_dir, 'oka_normalized')
        os.makedirs(output_dir, exist_ok=True)

    # Apply Yeo-Johnson normalization if requested
    if use_yeo_johnson:
        print("Applying Yeo-Johnson normalization...")
        features_to_normalize = available_force + target_cols
        pt = PowerTransformer(method='yeo-johnson', standardize=True)
        df[features_to_normalize] = pt.fit_transform(df[features_to_normalize])
        
        # Update output directory to indicate normalization
        output_dir = os.path.join(output_dir, 'yeo_johnson_normalized')
        os.makedirs(output_dir, exist_ok=True)

    # PCA Analysis
    print("Performing PCA analysis...")
    features = available_force + target_cols
    
    # Standardize the data before PCA
    df_pca = df[features].copy()
    df_pca = (df_pca - df_pca.mean()) / df_pca.std()
    
    pca = PCA()
    pca.fit(df_pca)
    
    # Plot PCA components heatmap
    plt.figure(figsize=(12, 8))
    components_df = pd.DataFrame(
        pca.components_,
        columns=features,
        index=[f'PC{i+1}' for i in range(len(features))]
    )
    
    sns.heatmap(components_df, cmap='coolwarm', center=0, annot=True, fmt='.2f')
    plt.title('PCA Components Heatmap')
    plt.tight_layout()
    
    pca_save_path = os.path.join(output_dir, 'pca_components_heatmap.png')
    plt.savefig(pca_save_path)
    plt.close()
    print(f"Saved PCA heatmap to {pca_save_path}")

    # Plot explained variance ratio
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(features) + 1), np.cumsum(pca.explained_variance_ratio_), 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Explained Variance Ratio')
    plt.grid(True)
    
    variance_save_path = os.path.join(output_dir, 'pca_explained_variance.png')
    plt.savefig(variance_save_path)
    plt.close()
    print(f"Saved PCA variance plot to {variance_save_path}")

    print("Plotting correlations...")
    
    # Plot scatter plots
    plot_tasks = [(f, t) for f in available_force for t in target_cols]
    
    for force, target in tqdm(plot_tasks, desc="Generating plots"):
        plt.figure(figsize=(10, 8))
        
        # Color by distance if available and node_id is specified
        if target_node_id is not None and 'distance_to_force_node' in df.columns:
            scatter = plt.scatter(df[force], df[target], c=df['distance_to_force_node'], cmap='viridis', alpha=0.5, s=10, rasterized=True)
            cbar = plt.colorbar(scatter)
            cbar.set_label('Distance to Force Node')
        # Fallback to force magnitude
        elif 'force_magnitude' in df.columns and df['force_magnitude'].nunique() > 1:
            scatter = plt.scatter(df[force], df[target], c=df['force_magnitude'], cmap='viridis', alpha=0.5, s=10, rasterized=True)
            cbar = plt.colorbar(scatter)
            cbar.set_label('Force Magnitude')
        else:
            plt.scatter(df[force], df[target], alpha=0.5, s=10, rasterized=True)
        
        plt.xlabel(f'Force Node Displacement {force}')
        plt.ylabel(f'{target}')
        plt.title(f'Correlation: {force} vs {target}')
        plt.grid(True)
        
        save_path = os.path.join(output_dir, f'correlation_{force}_{target}.png')
        plt.savefig(save_path)
        plt.close()

if __name__ == "__main__":
    main()
