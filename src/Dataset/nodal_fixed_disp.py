import pandas as pd
import numpy as np
import os
import argparse

def calculate_distance_from_fixed_center(coords_path, fixed_nodes_path, output_path):
    """
    Calculate the distance of each node from the centroid of fixed nodes.

    Args:
        coords_path (str): Path to the node coordinates CSV file (liver_coordinates.csv).
        fixed_nodes_path (str): Path to the fixed nodes info CSV file (fixed_nodes.csv).
        output_path (str): Path to save the output CSV.
    """
    print(f"Loading coordinates from {coords_path}...")
    try:
        df_coords = pd.read_csv(coords_path)
    except Exception as e:
        print(f"Error reading coordinates file: {e}")
        return

    print(f"Loading fixed nodes info from {fixed_nodes_path}...")
    try:
        df_fixed = pd.read_csv(fixed_nodes_path)
    except Exception as e:
        print(f"Error reading fixed nodes file: {e}")
        return

    # Merge dataframes on node_id
    # Assuming 'node_id' column exists in both
    if 'node_id' not in df_coords.columns or 'node_id' not in df_fixed.columns:
        print("Error: 'node_id' column missing in one of the input files.")
        return

    df = pd.merge(df_coords, df_fixed, on='node_id')

    # Filter fixed nodes
    # Check if 'is_fixed' column acts as boolean or string
    fixed_nodes = df[df['is_fixed'] == True]
    
    if fixed_nodes.empty:
        # Try checking for string 'True' just in case
        fixed_nodes = df[df['is_fixed'].astype(str).str.lower() == 'true']

    if fixed_nodes.empty:
        print("Error: No fixed nodes found.")
        return
    
    print(f"Found {len(fixed_nodes)} fixed nodes.")

    # Calculate centroid (center of mass) of fixed nodes
    center_x = fixed_nodes['x'].mean()
    center_y = fixed_nodes['y'].mean()
    center_z = fixed_nodes['z'].mean()

    print(f"Centroid of fixed nodes: ({center_x:.4f}, {center_y:.4f}, {center_z:.4f})")

    # Calculate Euclidean distance for all nodes
    df['distance_from_fixed_center'] = np.sqrt(
        (df['x'] - center_x)**2 + 
        (df['y'] - center_y)**2 + 
        (df['z'] - center_z)**2
    )

    # Select columns to save
    output_df = df[['node_id', 'distance_from_fixed_center']]
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Distance data saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Calculate distance from fixed nodes centroid.')
    parser.add_argument('--coords', type=str, 
                        default='dataset/liver_model_info/liver_coordinates.csv',
                        help='Path to liver_coordinates.csv')
    parser.add_argument('--fixed', type=str, 
                        default='dataset/liver_model_info/fixed_nodes.csv',
                        help='Path to fixed_nodes.csv')
    parser.add_argument('--output', type=str, 
                        default='dataset/liver_model_info/node_distances.csv',
                        help='Path to output CSV file')
    
    args = parser.parse_args()
    
    calculate_distance_from_fixed_center(args.coords, args.fixed, args.output)

if __name__ == '__main__':
    main()
