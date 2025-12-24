import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_comparison(before_path, after_path, output_dir):
    # Load data
    print(f"Loading before data from {before_path}")
    try:
        df_before = pd.read_csv(before_path)
    except FileNotFoundError:
        print(f"Error: File not found at {before_path}")
        return

    print(f"Loading after data from {after_path}")
    try:
        df_after = pd.read_csv(after_path)
    except FileNotFoundError:
        print(f"Error: File not found at {after_path}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Columns to plot (excluding node_id if present)
    columns = [c for c in df_before.columns if 'error' in c]
    
    if not columns:
        print("No columns with 'error' found in the CSV.")
        return

    # Prepare summary plots
    num_plots = len(columns)
    cols_grid = 3
    rows_grid = (num_plots + cols_grid - 1) // cols_grid
    
    fig_lin, axes_lin = plt.subplots(rows_grid, cols_grid, figsize=(5 * cols_grid, 4 * rows_grid))
    fig_log, axes_log = plt.subplots(rows_grid, cols_grid, figsize=(5 * cols_grid, 4 * rows_grid))
    
    axes_lin = axes_lin.flatten()
    axes_log = axes_log.flatten()

    for i, col in enumerate(columns):
        if col not in df_after.columns:
            print(f"Warning: Column {col} not found in after dataset. Skipping.")
            continue

        print(f"Plotting {col}...")
        
        # Determine common bin range
        min_val = min(df_before[col].min(), df_after[col].min())
        max_val = max(df_before[col].max(), df_after[col].max())
        bins = 50
        
        # Plot Linear Scale
        plt.figure(figsize=(10, 6))
        plt.hist(df_before[col], bins=bins, range=(min_val, max_val), alpha=0.5, label='Before', color='blue', density=False)
        plt.hist(df_after[col], bins=bins, range=(min_val, max_val), alpha=0.5, label='After', color='red', density=False)
        plt.title(f'Error Distribution Comparison: {col} (Linear)')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'{col}_linear.png'))
        plt.close()

        # Add to summary linear
        ax = axes_lin[i]
        ax.hist(df_before[col], bins=bins, range=(min_val, max_val), alpha=0.5, label='Before', color='blue', density=False)
        ax.hist(df_after[col], bins=bins, range=(min_val, max_val), alpha=0.5, label='After', color='red', density=False)
        ax.set_title(f'{col} (Linear)')
        ax.set_xlabel('Error')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot Log Scale (Y-axis log)
        plt.figure(figsize=(10, 6))
        plt.hist(df_before[col], bins=bins, range=(min_val, max_val), alpha=0.5, label='Before', color='blue', density=False)
        plt.hist(df_after[col], bins=bins, range=(min_val, max_val), alpha=0.5, label='After', color='red', density=False)
        plt.yscale('log')
        plt.title(f'Error Distribution Comparison: {col} (Log Scale)')
        plt.xlabel('Error')
        plt.ylabel('Frequency (Log)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'{col}_log.png'))
        plt.close()

        # Add to summary log
        ax = axes_log[i]
        ax.hist(df_before[col], bins=bins, range=(min_val, max_val), alpha=0.5, label='Before', color='blue', density=False)
        ax.hist(df_after[col], bins=bins, range=(min_val, max_val), alpha=0.5, label='After', color='red', density=False)
        ax.set_yscale('log')
        ax.set_title(f'{col} (Log Scale)')
        ax.set_xlabel('Error')
        ax.set_ylabel('Frequency (Log)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(i + 1, len(axes_lin)):
        axes_lin[j].axis('off')
        axes_log[j].axis('off')

    # Save summary plots
    fig_lin.tight_layout()
    fig_lin.savefig(os.path.join(output_dir, 'summary_linear.png'))
    plt.close(fig_lin)
    
    fig_log.tight_layout()
    fig_log.savefig(os.path.join(output_dir, 'summary_log.png'))
    plt.close(fig_log)

    print("Plotting finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare error distributions between two CSV files.")
    parser.add_argument("--before", type=str, default="/workspace/gnn_benchmark_results_before.csv", help="Path to the 'before' CSV file")
    parser.add_argument("--after", type=str, default="/workspace/benchmark_results_after.csv", help="Path to the 'after' CSV file")
    parser.add_argument("--output", type=str, default="/workspace/plots/comparison", help="Directory to save plots")

    args = parser.parse_args()
    
    plot_comparison(args.before, args.after, args.output)
