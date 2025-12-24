import pandas as pd
import argparse
from pathlib import Path
import numpy as np
import sys

def main():
    parser = argparse.ArgumentParser(description="Calculate error metrics (Ed, Esigma, Etau) from evaluation CSVs.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing node evaluation CSVs (node*_eval.csv)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save error metric CSVs")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all node evaluation files
    files = list(input_dir.glob("node*_eval.csv"))
    if not files:
        print(f"No 'node*_eval.csv' files found in {input_dir}")
        return

    print(f"Found {len(files)} files. Loading data...")

    all_data = []
    
    # Expected columns for error calculation
    # We use the _abs_err columns because (true - pred)^2 == (abs_err)^2
    disp_cols = ["dx_abs_err", "dy_abs_err", "dz_abs_err"]
    sigma_cols = ["Sxx_abs_err", "Syy_abs_err", "Szz_abs_err"]
    tau_cols = ["Sxy_abs_err", "Syz_abs_err", "Szx_abs_err"]
    
    required_cols = disp_cols + sigma_cols + tau_cols

    for f in files:
        try:
            df = pd.read_csv(f)
            
            # Check if required columns exist
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: File {f.name} is missing required columns. Skipping.")
                continue
            
            # Keep only necessary columns to save memory
            cols_to_keep = ["time"] + required_cols
            all_data.append(df[cols_to_keep])
            
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not all_data:
        print("No valid data loaded.")
        return

    print("Concatenating data...")
    combined_df = pd.concat(all_data, ignore_index=True)

    print(f"Total samples: {len(combined_df)}")

    # --- Calculate Squared Errors per sample ---
    
    # Ed = (dx-dx^)^2 + (dy-dy^)^2 + (dz-dz^)^2
    combined_df['Ed_sample'] = (
        combined_df['dx_abs_err']**2 + 
        combined_df['dy_abs_err']**2 + 
        combined_df['dz_abs_err']**2
    )

    # Squared errors for individual components (for Esigma and Etau)
    combined_df['Sxx_sq'] = combined_df['Sxx_abs_err']**2
    combined_df['Syy_sq'] = combined_df['Syy_abs_err']**2
    combined_df['Szz_sq'] = combined_df['Szz_abs_err']**2
    
    combined_df['Sxy_sq'] = combined_df['Sxy_abs_err']**2
    combined_df['Syz_sq'] = combined_df['Syz_abs_err']**2
    combined_df['Szx_sq'] = combined_df['Szx_abs_err']**2

    # --- 1. Overall Average Error ---
    # Ed: RMSE of the displacement vector magnitude
    ed_val = np.sqrt(combined_df['Ed_sample'].mean())

    # Esigma: Sum of RMSEs of each component
    esigma_val = (
        np.sqrt(combined_df['Sxx_sq'].mean()) + 
        np.sqrt(combined_df['Syy_sq'].mean()) + 
        np.sqrt(combined_df['Szz_sq'].mean())
    )

    # Etau: Sum of RMSEs of each component
    etau_val = (
        np.sqrt(combined_df['Sxy_sq'].mean()) + 
        np.sqrt(combined_df['Syz_sq'].mean()) + 
        np.sqrt(combined_df['Szx_sq'].mean())
    )

    overall_metrics = {
        'Ed': ed_val,
        'Esigma': esigma_val,
        'Etau': etau_val
    }
    
    df_overall = pd.DataFrame([overall_metrics])
    overall_path = output_dir / "overall_error_metrics.csv"
    df_overall.to_csv(overall_path, index=False)
    print(f"Saved overall metrics to {overall_path}")
    print(df_overall)

    # --- 2. Average Error per Time Step ---
    # Group by time
    grouped = combined_df.groupby('time')
    
    # Ed: RMSE of vector sum per time
    ed_t = np.sqrt(grouped['Ed_sample'].mean())
    
    # Esigma: Sum of RMSEs per time
    esigma_t = (
        np.sqrt(grouped['Sxx_sq'].mean()) + 
        np.sqrt(grouped['Syy_sq'].mean()) + 
        np.sqrt(grouped['Szz_sq'].mean())
    )
    
    # Etau: Sum of RMSEs per time
    etau_t = (
        np.sqrt(grouped['Sxy_sq'].mean()) + 
        np.sqrt(grouped['Syz_sq'].mean()) + 
        np.sqrt(grouped['Szx_sq'].mean())
    )

    time_metrics = pd.DataFrame({
        'time': ed_t.index,
        'Ed': ed_t.values,
        'Esigma': esigma_t.values,
        'Etau': etau_t.values
    })

    time_path = output_dir / "time_series_error_metrics.csv"
    time_metrics.to_csv(time_path, index=False)
    print(f"Saved time series metrics to {time_path}")

if __name__ == "__main__":
    main()
