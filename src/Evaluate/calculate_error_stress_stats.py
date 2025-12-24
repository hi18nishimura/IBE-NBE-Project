import pandas as pd
import argparse
from pathlib import Path
import numpy as np
import sys

def main():
    parser = argparse.ArgumentParser(description="Calculate stress error statistics (RMSE per component and Esigma/Etau).")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing node evaluation CSVs (node*_eval.csv)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save error metric CSVs")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all result files
    files = list(input_dir.glob("*_results.csv"))
    if not files:
        print(f"No '*_results.csv' files found in {input_dir}")
        return

    print(f"Found {len(files)} files. Loading data...")

    all_data = []
    
    # Stress columns
    sigma_cols = ["Sxx", "Syy", "Szz"]
    tau_cols = ["Sxy", "Syz", "Szx"]
    all_stress_cols = sigma_cols + tau_cols
    
    # Error columns in the provided CSV are like Sxx_error
    required_cols = [f"{c}_error" for c in all_stress_cols]

    for f in files:
        try:
            df = pd.read_csv(f)
            if not all(col in df.columns for col in required_cols):
                print(f"Skipping {f.name}: missing required columns")
                continue
            
            # Filter out time=1 if present (usually initial state with 0 error)
            if 'time' in df.columns:
                df = df[df['time'] != 1]
            
            cols_to_keep = ["time"] + required_cols
            all_data.append(df[cols_to_keep])
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not all_data:
        print("No valid data loaded.")
        return

    print("Concatenating data...")
    combined_df = pd.concat(all_data, ignore_index=True)

    # Calculate squared errors for each component
    # Note: The input columns are signed errors, so we square them.
    for col in all_stress_cols:
        combined_df[f'{col}_sq'] = combined_df[f'{col}_error']**2

    # --- Overall Statistics ---
    overall_stats = {}
    
    # RMSE for each component
    for col in all_stress_cols:
        rmse = np.sqrt(combined_df[f'{col}_sq'].mean())
        overall_stats[f'RMSE_{col}'] = rmse

    # Esigma and Etau
    overall_stats['Esigma'] = (
        overall_stats['RMSE_Sxx'] + 
        overall_stats['RMSE_Syy'] + 
        overall_stats['RMSE_Szz']
    )
    overall_stats['Etau'] = (
        overall_stats['RMSE_Sxy'] + 
        overall_stats['RMSE_Syz'] + 
        overall_stats['RMSE_Szx']
    )

    df_overall = pd.DataFrame([overall_stats])
    # Reorder columns for clarity
    cols_order = [f'RMSE_{c}' for c in all_stress_cols] + ['Esigma', 'Etau']
    df_overall = df_overall[cols_order]
    
    overall_path = output_dir / "overall_stress_error_stats.csv"
    df_overall.to_csv(overall_path, index=False)
    print(f"Saved overall stress stats to {overall_path}")
    print(df_overall)

    # --- Per Time Step Statistics ---
    grouped = combined_df.groupby('time')
    
    time_stats = pd.DataFrame({'time': grouped['time'].first().index}) # Initialize with time index
    
    # RMSE per time for each component
    for col in all_stress_cols:
        # sqrt(mean(sq_error)) per group
        rmse_t = np.sqrt(grouped[f'{col}_sq'].mean())
        time_stats[f'RMSE_{col}'] = rmse_t.values

    # Esigma per time
    time_stats['Esigma'] = (
        time_stats['RMSE_Sxx'] + 
        time_stats['RMSE_Syy'] + 
        time_stats['RMSE_Szz']
    )
    
    # Etau per time
    time_stats['Etau'] = (
        time_stats['RMSE_Sxy'] + 
        time_stats['RMSE_Syz'] + 
        time_stats['RMSE_Szx']
    )

    time_path = output_dir / "time_series_stress_error_stats.csv"
    time_stats.to_csv(time_path, index=False)
    print(f"Saved time series stress stats to {time_path}")

if __name__ == "__main__":
    main()
