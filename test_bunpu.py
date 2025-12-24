import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import glob

def calculate_percentiles(directory_path):
    dir_path = Path(directory_path)
    if not dir_path.exists():
        print(f"Error: Directory {dir_path} does not exist.")
        return

    feather_files = list(dir_path.glob("*.feather"))
    if not feather_files:
        print(f"No .feather files found in {dir_path}")
        return

    print(f"Found {len(feather_files)} feather files. Loading data...")

    all_d_mags = []

    for f in feather_files:
        try:
            df = pd.read_feather(f)
            
            # Check if required columns exist
            required_cols = {'dx', 'dy', 'dz'}
            if not required_cols.issubset(df.columns):
                print(f"Skipping {f.name}: Missing columns {required_cols - set(df.columns)}")
                continue

            # Calculate displacement magnitude
            d_mag = np.sqrt(df['dx']**2 + df['dy']**2 + df['dz']**2)
            all_d_mags.append(d_mag.values)
            
        except Exception as e:
            print(f"Error reading {f.name}: {e}")

    if not all_d_mags:
        print("No valid data found.")
        return

    # Concatenate all data
    combined_d_mag = np.concatenate(all_d_mags)
    print(f"Total data points: {len(combined_d_mag)}")

    # Calculate percentiles
    # Top X% corresponds to (100 - X)th percentile
    percentages = [10, 15, 20, 25, 30]
    percentiles = [100 - p for p in percentages]
    
    results = np.percentile(combined_d_mag, percentiles)

    print("\n=== Displacement Magnitude (sqrt(dx^2+dy^2+dz^2)) Thresholds ===")
    for p, val in zip(percentages, results):
        print(f"Top {p}% (Percentile {100-p}): {val:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate displacement magnitude percentiles from feather files.")
    parser.add_argument("directory", type=str, help="Directory containing .feather files")
    args = parser.parse_args()

    calculate_percentiles(args.directory)
