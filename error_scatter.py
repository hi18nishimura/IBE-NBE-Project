import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 引数の設定
    parser = argparse.ArgumentParser(description='Create scatter plots of (correct vs error) for specified features from CSV files.')
    parser.add_argument('directory', type=str, help='Path to the directory containing result CSV files.')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save the output plots.')
    parser.add_argument('--file_pattern', type=str, default='*.csv', help='Glob pattern for CSV files (default: *.csv).')

    args = parser.parse_args()

    # ファイルの検索
    search_path = os.path.join(args.directory, args.file_pattern)
    files = glob.glob(search_path)

    if not files:
        print(f"No files found matching {search_path}")
        return

    print(f"Found {len(files)} files. Reading data...")

    # プロット対象の特徴量
    features = ['dx', 'dy', 'dz', 'Sxx', 'Syy', 'Szz', 'Sxy', 'Syz', 'Szx']
    
    # データを格納する辞書
    data_store = {feat: {'correct': [], 'error': []} for feat in features}

    # 各CSVファイルを読み込み
    for index, fpath in enumerate(files):
        try:
            df = pd.read_csv(fpath)
            
            for feat in features:
                col_correct = f"{feat}_correct"
                col_error = f"{feat}_error"

                if col_correct in df.columns and col_error in df.columns:
                    # 欠損値を除去してリストに追加
                    valid_data = df[[col_correct, col_error]].dropna()
                    data_store[feat]['correct'].extend(valid_data[col_correct].tolist())
                    data_store[feat]['error'].extend(valid_data[col_error].tolist())
        
            if (index + 1) % 100 == 0:
                print(f"Processed {index + 1}/{len(files)} files...")

        except Exception as e:
            print(f"Error reading {fpath}: {e}")

    print("Data reading complete. Generating plots...")

    # プロットのレイアウト設定 (3x3)
    num_cols = 3
    num_rows = 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 18))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        x_data = data_store[feat]['correct']
        y_data = data_store[feat]['error']

        if x_data and y_data:
            # 散布図のプロット
            # alpha: 透明度, s: 点のサイズ
            ax.scatter(x_data, y_data, alpha=0.3, s=2, edgecolors='none')
            ax.set_xlabel(f'{feat}_correct')
            ax.set_ylabel(f'{feat}_error')
            ax.set_title(f'{feat} Error Scatter')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # y=0 の補助線
            ax.axhline(0, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
        else:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            ax.set_title(f'{feat}')
    
    # 余ったプロットエリアを非表示
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    
    # 保存
    output_path = os.path.join(args.output_dir, 'error_scatter_plot.png')
    plt.savefig(output_path, dpi=300)
    print(f"Scatter plot saved to {output_path}")

if __name__ == "__main__":
    main()