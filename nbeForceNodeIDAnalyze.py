import argparse
import pandas as pd
from pathlib import Path
from typing import List, Set
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
sys.path.append("/workspace")
from src.Dataloader.nbeForceNodeIDDataset import NbeForceNodeIDDataset

def get_neighbors(node_connection_file: str | Path, node_id: int, hops: int = 2) -> List[int]:
    """
    指定したnode_idの隣接ノードを指定されたホップ数まで取得する。
    hops=1: 直接の隣接ノードのみ
    hops=2: 隣接ノードおよびその隣接ノード（2-hop）
    結果は重複のない整数のリストとして返す。
    """
    path = Path(node_connection_file)
    if not path.exists():
        raise FileNotFoundError(f"node_connections.csv not found: {path}")

    df = pd.read_csv(path)
    
    # helper to parse neighbors string "1,2,3" -> [1, 2, 3]
    def parse_neighbors(neighbors_str) -> List[int]:
        if pd.isna(neighbors_str):
            return []
        if isinstance(neighbors_str, str):
            # remove quotes if present and split
            s = neighbors_str.replace('"', '').strip()
            if not s:
                return []
            return [int(x) for x in s.split(',') if x.strip()]
        # already integer or list? likely string in csv
        return []

    # Build adjacency dictionary
    adjacency: dict[int, List[int]] = {}
    for _, row in df.iterrows():
        nid = int(row['node_id'])
        neighbors = parse_neighbors(row['neighbors'])
        adjacency[nid] = neighbors

    if node_id not in adjacency:
        print(f"Node {node_id} not found in connection file.")
        return []

    # 1. Get direct neighbors of target node_id
    direct_neighbors = adjacency[node_id]
    
    # Base set includes direct neighbors
    result_set: Set[int] = set(direct_neighbors)
    
    if hops >= 2:
        # 2. Get neighbors of neighbors
        for neighbor_id in direct_neighbors:
            if neighbor_id in adjacency:
                second_hop = adjacency[neighbor_id]
                result_set.update(second_hop)
            
    # Include original direct neighbors (already in set) 
    #Sort for deterministic output
    return sorted(list(result_set))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze dataset distribution for N-hop neighbors.")
    parser.add_argument("--hops", type=int, default=2, choices=[1, 2], help="Number of hops for neighbors (1 or 2)")
    parser.add_argument("--node_id", type=int, default=1, help="Target Node ID")
    
    args = parser.parse_args()

    # Settings
    CSV_PATH = "/workspace/dataset/liver_model_info/node_connections.csv"
    TARGET_NODE_ID = args.node_id
    HOPS = args.hops
    
    dataset_dir = "/workspace/dataset/bin/toy_all_model/train"
    output_dir = "/workspace/analyze/dataset"
    
    print(f"Analyzing for Node {TARGET_NODE_ID} with {HOPS}-hop neighbors.")

    try:
        result_neighbors = get_neighbors(CSV_PATH, TARGET_NODE_ID, hops=HOPS)
        print(f"Neighbors for node {TARGET_NODE_ID} ({HOPS}-hop):")
        print(result_neighbors)
        print(f"Count: {len(result_neighbors)}")
        
        # Initialize NbeForceNodeIDDataset with result_neighbors as target_force_node_ids
        print("\nInitializing NbeForceNodeIDDataset...")
        dataset = NbeForceNodeIDDataset(
            data_dir=dataset_dir,
            node_id=TARGET_NODE_ID,
            target_force_node_ids=result_neighbors,
            global_normalize=True,
            preload=False
        )
        print(f"Dataset initialized. Found {len(dataset)} files matching the criteria.")
        
        if len(dataset) > 0:
            print("Collecting data for histograms...")
            all_data = [] # For neighbors subset
            target_node_data = [] # For target node only
            
            # Collect data from all files
            for i in tqdm(range(len(dataset)), desc="Loading physical values"):
                # All neighbors subset
                df = dataset.get_physical_values(i)
                all_data.append(df)
                
                # Target node only
                df_target = dataset.get_target_node_physical_values(i)
                target_node_data.append(df_target)
            
            # --- 1. Plot All Neighbors Subset ---
            if all_data:
                full_df = pd.concat(all_data, ignore_index=True)
                
                # Create output directory
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                target_columns = ["dx", "dy", "dz", "Sxx", "Syy", "Szz", "Sxy", "Syz", "Szx"]
                
                print(f"Plotting neighbors subset histograms to {output_path}...")
                for col in target_columns:
                    if col in full_df.columns:
                        # Drop NaNs before plotting
                        data_to_plot = full_df[col].dropna()
                        
                        if len(data_to_plot) > 0:
                            # Original Histogram
                            plt.figure(figsize=(10, 6))
                            plt.hist(data_to_plot, bins=50, edgecolor='black', alpha=0.7)
                            plt.title(f"Histogram of {col} (Node {TARGET_NODE_ID}, {HOPS}-hop neighbors subset)")
                            plt.xlabel(col)
                            plt.ylabel("Count")
                            plt.grid(True, alpha=0.3)
                            
                            save_file = output_path / f"hist_{col}_hop{HOPS}.png"
                            plt.savefig(save_file)
                            print(f"Saved {save_file}")
                            plt.close()
                        else:
                            print(f"No valid data for {col}")

                # Save collected data to CSV
                csv_file = output_path / f"extracted_features_hop{HOPS}.csv"
                full_df.to_csv(csv_file, index=False)
                print(f"Saved extracted features to {csv_file}")
            else:
                print("No data collected for neighbors subset.")

            # --- 2. Plot Target Node Only ---
            if target_node_data:
                full_target_df = pd.concat(target_node_data, ignore_index=True)
                
                # Save collected data to CSV
                csv_file_target = output_path / f"extracted_features_target_only.csv"
                full_target_df.to_csv(csv_file_target, index=False)
                print(f"Saved target node features to {csv_file_target}")
                
                print(f"Plotting target node histograms to {output_path}...")
                for col in target_columns:
                    if col in full_target_df.columns:
                        data_to_plot = full_target_df[col].dropna()
                        
                        if len(data_to_plot) > 0:
                            plt.figure(figsize=(10, 6))
                            plt.hist(data_to_plot, bins=50, edgecolor='black', alpha=0.7, color='green')
                            plt.title(f"Histogram of {col} (Only Node {TARGET_NODE_ID})")
                            plt.xlabel(col)
                            plt.ylabel("Count")
                            plt.grid(True, alpha=0.3)
                            
                            save_file = output_path / f"hist_{col}_target_only_hop{HOPS}.png"
                            plt.savefig(save_file)
                            print(f"Saved {save_file}")
                            plt.close()
                        else:
                            print(f"No valid target node data for {col}")
            else:
                print("No data collected for target node.")

        else:
            print("No files found, skipping plots.")
        
    except Exception as e:
        print(f"Error: {e}")