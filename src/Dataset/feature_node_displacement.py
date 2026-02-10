import pandas as pd
import numpy as np
import os
import sys

def calculate_node_displacement(input_file, output_file):
    """
    Calculates the distance between each node and all other nodes in terms of x, y, z axes.
    Normalizes the distances to 0-1 based on the maximum distance for each specified node.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return

    print(f"Reading input from: {input_file}")
    df = pd.read_csv(input_file)
    
    # Ensure required columns exist
    required_cols = ['node_id', 'x', 'y', 'z']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Input file must contain columns: {required_cols}")
        return

    node_ids = df['node_id'].values
    coords = df[['x', 'y', 'z']].values
    num_nodes = len(node_ids)

    # Lists to store result data
    res_node_id = []
    res_target_id = []
    res_x = []
    res_y = []
    res_z = []
    res_x_nor = []
    res_y_nor = []
    res_z_nor = []

    print("Calculating distances...")
    
    # Iterate through each node as the "specified node"
    for i in range(num_nodes):
        current_id = node_ids[i]
        current_pos = coords[i]
        
        # Calculate absolute distances to all target nodes for this specified node
        # shape: (num_nodes, 3)
        diffs = np.abs(coords - current_pos)
        
        # Find max distance in each axis for this specified node
        # to normalize x, y, z to 0~1
        max_vals = np.max(diffs, axis=0)
        
        # Avoid division by zero
        max_vals[max_vals == 0] = 1.0
        
        # Normalize
        norms = diffs / max_vals
        
        # Append to lists
        # We repeat current_id 'num_nodes' times
        res_node_id.extend([current_id] * num_nodes)
        res_target_id.extend(node_ids)
        res_x.extend(diffs[:, 0])
        res_y.extend(diffs[:, 1])
        res_z.extend(diffs[:, 2])
        res_x_nor.extend(norms[:, 0])
        res_y_nor.extend(norms[:, 1])
        res_z_nor.extend(norms[:, 2])

    # Create DataFrame
    output_df = pd.DataFrame({
        'node_id': res_node_id,
        'target_node_id': res_target_id,
        'x': res_x,
        'y': res_y,
        'z': res_z,
        'x_nor': res_x_nor,
        'y_nor': res_y_nor,
        'z_nor': res_z_nor
    })

    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output_df.to_csv(output_file, index=False)
    print(f"Successfully saved calculated features to: {output_file}")
    print(f"Total rows: {len(output_df)}")

if __name__ == "__main__":
    # Default paths based on workspace structure
    INPUT_PATH = '/workspace/dataset/liver_model_info/liver_coordinates.csv'
    OUTPUT_PATH = '/workspace/dataset/liver_model_info/node_displacement_features.csv'
    
    calculate_node_displacement(INPUT_PATH, OUTPUT_PATH)
