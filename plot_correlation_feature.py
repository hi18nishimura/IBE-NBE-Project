#指定したディレクトリにあるfeatherファイルを全て読み込み、特徴量間の相関関係をプロットする関数
# 変位（dx,dy,dz)と応力（Sxx,Syy,Szz,Sxy,Syz,Szx）の相関関係をプロットする
# 変位と応力に関係があれば、特徴量としてネットワークに含める根拠になる
# コマンドライン引数でデータセットのディレクトリとプロット結果を保存するディレクトリを指定する
# 例: python3 plot_correlation_feature.py --input_dir /workspace/dataset/bin/toy_all_model/train --output_dir /workspace/plots/correlation_features
# プロットしたいグラフ
# 各変位と応力の散布図("f_x","f_y","f_z"列からユークリッド距離を計算して、その値を使って色を付ける)
# 全結合したデータフレームはfeather形式で保存する。--input_dirで指定された値がfeatherファイルであれば、そのまま読み込む

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
    parser = argparse.ArgumentParser(description='Plot correlation between displacement and stress.')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing feather files or path to a feather file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the plots.')
    parser.add_argument('--sample_size', type=int, default=100000, help='Number of samples to plot to speed up rendering. Set to 0 to plot all data.')
    parser.add_argument('--oka_normalize', action='store_true', help='Apply Oka normalization before plotting and PCA.')
    parser.add_argument('--yeo-johnson', action='store_true', help='Apply Yeo-Johnson normalization before plotting and PCA.')
    args = parser.parse_args()

    input_path = args.input_dir
    output_dir = args.output_dir
    sample_size = args.sample_size
    use_oka_normalize = args.oka_normalize
    use_yeo_johnson = args.yeo_johnson

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
                dfs.append(pd.read_feather(f))
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

    # Sampling to speed up plotting
    if sample_size > 0 and len(df) > sample_size:
        print(f"Sampling {sample_size} points from {len(df)} total points to speed up plotting...")
        df = df.sample(n=sample_size, random_state=42)

    # Calculate force magnitude for coloring
    if all(col in df.columns for col in ['f_x', 'f_y', 'f_z']):
        df['force_magnitude'] = np.sqrt(df['f_x']**2 + df['f_y']**2 + df['f_z']**2)
    else:
        print("Warning: f_x, f_y, f_z columns not found. Coloring by force magnitude will be skipped.")
        df['force_magnitude'] = 0 

    displacement_cols = ['dx', 'dy', 'dz']
    stress_cols = ['Sxx', 'Syy', 'Szz', 'Sxy', 'Syz', 'Szx']

    # Check if columns exist
    available_disp = [col for col in displacement_cols if col in df.columns]
    available_stress = [col for col in stress_cols if col in df.columns]

    if not available_disp or not available_stress:
        print("Missing displacement or stress columns.")
        return

    # Apply Oka normalization if requested
    if use_oka_normalize:
        print("Applying Oka normalization...")
        features_to_normalize = available_disp + available_stress
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
        features_to_normalize = available_disp + available_stress
        pt = PowerTransformer(method='yeo-johnson', standardize=True)
        df[features_to_normalize] = pt.fit_transform(df[features_to_normalize])
        
        # Update output directory to indicate normalization
        output_dir = os.path.join(output_dir, 'yeo_johnson_normalized')
        os.makedirs(output_dir, exist_ok=True)

    # PCA Analysis
    print("Performing PCA analysis...")
    features = available_disp + available_stress
    
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
    plot_tasks = [(d, s) for d in available_disp for s in available_stress]
    
    for disp, stress in tqdm(plot_tasks, desc="Generating plots"):
        plt.figure(figsize=(10, 8))
        
        # Scatter plot with color mapping
        if 'force_magnitude' in df.columns and df['force_magnitude'].nunique() > 1:
            scatter = plt.scatter(df[disp], df[stress], c=df['force_magnitude'], cmap='viridis', alpha=0.5, s=10, rasterized=True)
            cbar = plt.colorbar(scatter)
            cbar.set_label('Force Magnitude')
        else:
            plt.scatter(df[disp], df[stress], alpha=0.5, s=10, rasterized=True)
        
        plt.xlabel(f'Displacement {disp}')
        plt.ylabel(f'Stress {stress}')
        plt.title(f'Correlation: {disp} vs {stress}')
        plt.grid(True)
        
        save_path = os.path.join(output_dir, f'correlation_{disp}_{stress}.png')
        plt.savefig(save_path)
        plt.close()
        # print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    main()
