import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

def plot_timeseries(files, labels, output_file, y_col, title):
    plt.figure(figsize=(10, 6))
    
    if labels and len(labels) != len(files):
        print("Warning: Number of labels does not match number of files. Using filenames as labels.")
        labels = None

    all_times = set()

    for i, file_path in enumerate(files):
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found.")
            continue
            
        try:
            df = pd.read_csv(file_path)
            if 'time' not in df.columns or y_col not in df.columns:
                print(f"Error: Required columns ('time', '{y_col}') not found in {file_path}")
                continue
                
            label = labels[i] if labels else os.path.basename(file_path)
            plt.plot(df['time'], df[y_col], marker='o', label=label)
            
            # Collect time points for x-axis ticks
            all_times.update(df['time'].dropna().unique())
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Set x-axis ticks to integers with step 1
    if all_times:
        min_time = int(min(all_times))
        max_time = int(max(all_times))
        plt.xticks(np.arange(min_time, max_time + 1, 1))

    plt.xlabel('Time')
    plt.ylabel('MSE')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot time series from CSV files.")
    parser.add_argument("files", nargs='+', help="List of CSV files to plot")
    parser.add_argument("--labels", nargs='+', help="List of labels for the legend")
    parser.add_argument("--output", default="timeseries_plot.png", help="Output plot filename")
    parser.add_argument("--title", default="Mean Squared Error over Time(Displacement exceeding the top 10% only)", help="Title of the plot")
    parser.add_argument("--y_col", default="d_error_mean", help="Column name for Y-axis")
    
    args = parser.parse_args()
    
    plot_timeseries(args.files, args.labels, args.output, args.y_col, args.title)
