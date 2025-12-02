import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_error_distribution(csv_path, output_dir):
    """
    Plots the distribution of errors from the given CSV file.
    
    Args:
        csv_path (str): Path to the CSV file containing error data.
        output_dir (str): Directory to save the plots.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return

    # Target columns
    error_cols = [
        'x_error', 'y_error', 'z_error',
        'Sxx_error', 'Syy_error', 'Szz_error',
        'Sxy_error', 'Syz_error', 'Szx_error'
    ]
    
    # Check if columns exist
    missing_cols = [col for col in error_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: The following columns are missing in the CSV: {missing_cols}")
        # Filter out missing columns
        error_cols = [col for col in error_cols if col in df.columns]
        if not error_cols:
            print("No error columns found to plot.")
            return

    disp_error_cols = [col for col in ['x_error', 'y_error', 'z_error'] if col in error_cols]
    stress_error_cols = [col for col in ['Sxx_error', 'Syy_error', 'Szz_error', 'Sxy_error', 'Syz_error', 'Szx_error'] if col in error_cols]

    # Set style
    sns.set_theme(style="whitegrid")

    print("Plotting overall distributions...")
    
    # 1. Overall Distribution (All times combined)
    
    # 1-1. Boxplot for all errors (Separated by type due to scale)
    if disp_error_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df[disp_error_cols])
        plt.title('Overall Displacement Error Distribution')
        plt.ylabel('Error')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'overall_displacement_error_boxplot.png'))
        plt.close()

    if stress_error_cols:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df[stress_error_cols])
        plt.title('Overall Stress Error Distribution')
        plt.ylabel('Error')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'overall_stress_error_boxplot.png'))
        plt.close()

    # 1-2. Histograms for each error
    for col in error_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Overall Distribution of {col}')
        plt.xlabel('Error')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'overall_{col}_hist.png'))
        plt.close()

    print("Plotting time-series distributions...")

    # 2. Distribution per Time Step
    if 'time' in df.columns:
        unique_times = df['time'].unique()
        unique_times.sort()
        
        for col in error_cols:
            plt.figure(figsize=(15, 8))
            sns.boxplot(x='time', y=col, data=df)
            
            # Adjust x-axis labels if too many
            if len(unique_times) > 20:
                # Show only a subset of labels to avoid overcrowding
                ax = plt.gca()
                # Calculate step size to show roughly 20 labels
                step = max(1, len(unique_times) // 20)
                
                # Get current xticks and labels
                locs, labels = plt.xticks()
                
                # Create new list of visible labels
                new_labels = []
                for i, label in enumerate(labels):
                    if i % step == 0:
                        new_labels.append(label.get_text())
                    else:
                        new_labels.append("")
                
                ax.set_xticklabels(new_labels, rotation=90)
            else:
                plt.xticks(rotation=45)
            
            plt.title(f'{col} Distribution per Time Step')
            plt.xlabel('Time Step')
            plt.ylabel('Error')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'time_series_{col}_boxplot.png'))
            plt.close()
    else:
        print("Warning: 'time' column not found. Skipping time-series plots.")

    print(f"All plots saved to {output_dir}")

if __name__ == "__main__":
    csv_file = 'nbe_peephole_lstm_evaluation_results.csv'
    output_directory = 'plots/error_distribution'
    
    # Check if file exists in current directory, if not try workspace root
    if not os.path.exists(csv_file):
        csv_file = os.path.join('/workspace', csv_file)
        
    plot_error_distribution(csv_file, output_directory)
