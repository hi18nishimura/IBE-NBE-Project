import pandas as pd
from pathlib import Path

def main():
    # Define paths
    base_dir = Path('/workspace/dataset/liver_model_info')
    fixed_nodes_path = base_dir / 'fixed_nodes.csv'
    connections_path = base_dir / 'node_connections.csv'
    output_path = base_dir / 'input_output_size.csv'

    # Load data
    fixed_df = pd.read_csv(fixed_nodes_path)
    connections_df = pd.read_csv(connections_path)

    # Create a lookup for fixed status
    # node_id is int
    fixed_status = dict(zip(fixed_df['node_id'], fixed_df['is_fixed']))

    results = []

    for index, row in connections_df.iterrows():
        node_id = row['node_id']
        neighbors_str = row['neighbors']
        
        if pd.isna(neighbors_str) or neighbors_str == "":
            neighbors = []
        else:
            # neighbors are comma separated
            neighbors = [int(x) for x in str(neighbors_str).split(',')]
        
        # Include the node itself in the list of nodes to consider
        nodes_to_consider = neighbors + [node_id]
        
        total_nodes = len(nodes_to_consider)
        fixed_nodes_count = 0
        
        for nid in nodes_to_consider:
            # Check if node is fixed
            if fixed_status.get(nid, False):
                fixed_nodes_count += 1
        
        # Calculation: (Total Nodes * 9) + (Fixed Nodes * -3)
        input_size = (total_nodes * 9) + (fixed_nodes_count * -3)
        
        # Calculate output_size
        is_fixed = fixed_status.get(node_id, False)
        output_size = 6 if is_fixed else 9

        results.append({
            'node_id': node_id,
            'input_size': input_size,
            'output_size': output_size
        })

    # Create DataFrame and save
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)
    print(f"Saved input/output sizes to {output_path}")
    print(result_df.head())

if __name__ == "__main__":
    main()
