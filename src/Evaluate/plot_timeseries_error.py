import pandas as pd
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import os

def main():
    parser = argparse.ArgumentParser(description="Plot time series error from CSV files.")
    parser.add_argument("files", nargs='+', help="List of CSV files to plot")
    parser.add_argument("--columns", nargs='+', required=True, help="List of columns to plot (y-axis)")
    parser.add_argument("--output", type=str, default="timeseries_error_plot.png", help="Output image file name")
    parser.add_argument("--labels", nargs='+', help="Labels for the files (must match number of files)")
    parser.add_argument("--title", type=str, default="Time Series Error", help="Plot title")
    parser.add_argument("--log_scale", action="store_true", help="Use log scale for y-axis")
    parser.add_argument("--no_grid", action="store_false", dest="grid", help="Disable grid")
    parser.set_defaults(grid=True)
    
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.files):
        print("Error: Number of labels must match number of files.")
        return

    plt.figure(figsize=(10, 6))

    # Color cycle or style cycle could be useful if many lines
    # For now, let matplotlib handle colors automatically
    
    for i, file_path in enumerate(args.files):
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
            
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        if 'time' not in df.columns:
            print(f"Warning: 'time' column not found in {file_path}. Skipping.")
            continue

        # Sort by time just in case
        df = df.sort_values('time')

        file_label = args.labels[i] if args.labels else Path(file_path).name

        for col in args.columns:
            if col not in df.columns:
                print(f"Warning: Column '{col}' not found in {file_path}. Skipping this column.")
                continue
            
            # Construct label: "FileLabel - Column" or just "FileLabel" if only one column, or "Column" if only one file
            if len(args.files) > 1 and len(args.columns) > 1:
                label = f"{file_label} - {col}"
            elif len(args.files) > 1:
                label = f"{file_label}"
            else:
                label = f"{col}"

            plt.plot(df['time'], df[col], marker='o', linestyle='-', label=label)

    plt.xlabel("Time")
    plt.ylabel("Error Value")
    plt.title(args.title)
    plt.legend()
    
    if args.grid:
        plt.grid(True, which="both", ls="-", alpha=0.5)
        
    if args.log_scale:
        plt.yscale('log')

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()
