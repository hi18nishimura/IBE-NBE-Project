import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_nodal_error_scatter(error_csv_path, distance_csv_path, output_dir):
    """
    Plot scatter charts of nodal error statistics.

    Args:
        error_csv_path (str): Path to nodal_mean_error.csv.
        distance_csv_path (str): Path to node_distances.csv (for coloring points).
        output_dir (str): Directory to save the plots.
    """
    
    # Check if files exist
    if not os.path.exists(error_csv_path):
        print(f"Error: {error_csv_path} not found.")
        return
    if not os.path.exists(distance_csv_path):
        print(f"Error: {distance_csv_path} not found.")
        return

    # Load data
    print(f"Loading error stats from {error_csv_path}...")
    df_error = pd.read_csv(error_csv_path)
    
    print(f"Loading distances from {distance_csv_path}...")
    df_dist = pd.read_csv(distance_csv_path)

    # Merge dataframes
    if 'node_id' not in df_error.columns or 'node_id' not in df_dist.columns:
        print("Error: 'node_id' column missing in input CSVs.")
        return

    df = pd.merge(df_error, df_dist, on='node_id')
    print(f"Merged data shape: {df.shape}")

    # Prepare output directory
    if output_dir is None:
        output_dir = os.path.dirname(error_csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # Set visualization style
    sns.set_theme(style="whitegrid")

    # Define variable groups
    displacement_err_vars = ['dx_error', 'dy_error', 'dz_error']
    position_err_vars = ['x_error', 'y_error', 'z_error']
    normal_stress_err_vars = ['Sxx_error', 'Syy_error', 'Szz_error']
    shear_stress_err_vars = ['Sxy_error', 'Syz_error', 'Szx_error']

    # Helper function for scatter plot
    def create_scatter(x_vars, y_vars, x_label_group, y_label_group, filename_suffix):
        # We want to plot all pairs of (x_var, y_var) but colored by distance
        # To make it manageable, we can plot the MEAN of errors for each category per node, 
        # OR plot all components. Let's plot components against components in a grid or aggregation.
        
        # Strategy: Flatten the dataframe?
        # A simpler approach requested: "Scatter plot with X=Stress, Y=Displacement"
        # Since these are vector components, let's use the composite errors calculated in calculate_error.py
        # Or averages of components if composite not available.
        # Looking at calculate_error.py output (nodal_mean_error.csv), we have:
        # position_euclidean_error, displacement_euclidean_error, stress_diag_rss_error, stress_offdiag_rss_error
        
        # Let's map the user request to these composite metrics first for a summary plot
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Determine X and Y data based on request logic
        # 1. Stress (Normal) vs Displacement
        if x_label_group == 'Normal Stress Error' and y_label_group == 'Displacement Error':
            x_data = df['stress_diag_rss_error']
            y_data = df['displacement_euclidean_error']
            
        # 2. Stress (Shear) vs Displacement
        elif x_label_group == 'Shear Stress Error' and y_label_group == 'Displacement Error':
            x_data = df['stress_offdiag_rss_error']
            y_data = df['displacement_euclidean_error']
            
        # 3. Stress (Normal) vs Position Error
        elif x_label_group == 'Normal Stress Error' and y_label_group == 'Position Error':
            x_data = df['stress_diag_rss_error']
            y_data = df['position_euclidean_error']
            
        # 4. Stress (Shear) vs Position Error
        elif x_label_group == 'Shear Stress Error' and y_label_group == 'Position Error':
            x_data = df['stress_offdiag_rss_error']
            y_data = df['position_euclidean_error']

        # 5. Normal Stress vs Shear Stress
        elif x_label_group == 'Normal Stress Error' and y_label_group == 'Shear Stress Error':
            x_data = df['stress_diag_rss_error']
            y_data = df['stress_offdiag_rss_error']
            
        else:
            print(f"Skipping undefined combination: {x_label_group} vs {y_label_group}")
            plt.close()
            return

        scatter = ax.scatter(
            x_data, 
            y_data, 
            c=df['distance_from_fixed_center'], 
            cmap='viridis', 
            alpha=0.8,
            s=50,
            edgecolors='w', linewidth=0.5
        )
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Distance from Fixed Node Center')
        
        ax.set_xlabel(f'{x_label_group} (RSS/Euclidean)')
        ax.set_ylabel(f'{y_label_group} (RSS/Euclidean)')
        ax.set_title(f'Nodal Error Evaluation: {x_label_group} vs {y_label_group}')
        
        output_path = os.path.join(output_dir, f'scatter_{filename_suffix}.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved plot: {output_path}")

    # Generate the requested plots
    
    # 1. Normal Stress vs Displacement
    create_scatter(normal_stress_err_vars, displacement_err_vars, 
                  'Normal Stress Error', 'Displacement Error', 'normal_stress_vs_disp')

    # 2. Shear Stress vs Displacement
    create_scatter(shear_stress_err_vars, displacement_err_vars, 
                  'Shear Stress Error', 'Displacement Error', 'shear_stress_vs_disp')

    # 3. Normal Stress vs Position
    create_scatter(normal_stress_err_vars, position_err_vars, 
                  'Normal Stress Error', 'Position Error', 'normal_stress_vs_pos')

    # 4. Shear Stress vs Position
    create_scatter(shear_stress_err_vars, position_err_vars, 
                  'Shear Stress Error', 'Position Error', 'shear_stress_vs_pos')

    # 5. Normal Stress vs Shear Stress
    create_scatter(normal_stress_err_vars, shear_stress_err_vars, 
                  'Normal Stress Error', 'Shear Stress Error', 'normal_stress_vs_shear_stress')

def main():
    parser = argparse.ArgumentParser(description='Plot nodal error scatter charts.')
    parser.add_argument('error_csv', type=str, help='Path to nodal_mean_error.csv')
    parser.add_argument('--distance_csv', type=str, 
                        default='dataset/liver_model_info/node_distances.csv',
                        help='Path to node_distances.csv')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Directory to save plots. Defaults to directory of error_csv.')

    args = parser.parse_args()
    
    plot_nodal_error_scatter(args.error_csv, args.distance_csv, args.output_dir)

if __name__ == '__main__':
    main()
