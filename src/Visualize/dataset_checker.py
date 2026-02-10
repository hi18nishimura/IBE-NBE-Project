import open3d as o3d
import pandas as pd
import argparse
import glob
import os
import numpy as np
from tqdm import tqdm

class DatasetChecker:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.file_list = sorted(glob.glob(os.path.join(input_dir, "*.feather")))
        self.data_list = []
        self.current_index = 0
        
        # モデルの接続情報を読み込む
        self.surface_triangles = []
        self.tetra_path = "/workspace/dataset/liver_model_info/tetra_connectivity.csv"
        # 以前の wireframe 用のエッジ読み込みは mesh 表示にするため削除またはコメントアウトしてもよいが、
        # ユーザー要求は「モデル表面の色がついていると分かりやすい」なので Mesh に切り替える。
        self.load_surface_mesh_connectivity()

        if not self.file_list:
            print(f"No feather files found in {input_dir}")
            return

        print(f"Found {len(self.file_list)} files. Loading data...")
        self.load_all_data()

        self.mesh = o3d.geometry.TriangleMesh()
        if self.surface_triangles:
             self.mesh.triangles = o3d.utility.Vector3iVector(self.surface_triangles)

        # Force node sphere (Red)
        self.force_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=3.0)
        self.force_sphere.paint_uniform_color([1.0, 0.0, 0.0])
        # 初期位置は原点にあるとする (create_sphere のデフォルト中心は (0,0,0))
        self.current_sphere_center = np.array([0.0, 0.0, 0.0])

        if self.data_list:
            self.update_geometry()

    def load_surface_mesh_connectivity(self):
        coord_path = os.path.join(os.path.dirname(self.tetra_path), "liver_coordinates.csv")

        if not os.path.exists(self.tetra_path):
            print(f"Warning: Tetra connectivity file not found at {self.tetra_path}. Mesh surface will not be shown.")
            return

        print(f"Loading tetra connectivity from {self.tetra_path} to extract surface...")
        
        try:
            # 1. Load coordinates for orientation calculation (Initial shape)
            if not os.path.exists(coord_path):
                print(f"Warning: Coordinates file not found at {coord_path}. Cannot calculate correct surface orientation.")
                # We can still try to load without orientation correction if needed, but for now just return or handle partially.
                return

            df_coords = pd.read_csv(coord_path)
            # Ensure 0-based index alignment
            if 'node_id' in df_coords.columns:
                df_coords = df_coords.sort_values('node_id')
            initial_coords = df_coords[['x', 'y', 'z']].to_numpy()

            # 2. Load Tetra Connectivity
            df_tetra = pd.read_csv(self.tetra_path)
            # Check for header columns
            if {"n1", "n2", "n3", "n4"}.issubset(set(df_tetra.columns)):
                tets = df_tetra[["n1", "n2", "n3", "n4"]].to_numpy(dtype=int) - 1
            else:
                # Fallback if names are different or header missing (attempting default structure)
                # Assumes columns 1-4 are node indices if header is present but names differ, or 0-3 if no header?
                # Based on file check, it has headers.
                raise ValueError("Tetra connectivity file must have columns: n1, n2, n3, n4")

            # 3. Identify Surface Faces
            # Map: sorted_face_tuple -> list of (tet_index, face_local_index)
            from collections import defaultdict
            face_to_tets = defaultdict(list)

            # Faces of a tetrahedron indices: [0,1,2], [0,1,3], [0,2,3], [1,2,3]
            tet_faces_local_indices = [
                [0, 1, 2],
                [0, 1, 3],
                [0, 2, 3],
                [1, 2, 3]
            ]

            for t_idx, tet_nodes in enumerate(tets):
                for f_idx, local_indices in enumerate(tet_faces_local_indices):
                    # Nodes of the face
                    face_nodes = [tet_nodes[i] for i in local_indices]
                    # Sort to check uniqueness
                    sorted_face = tuple(sorted(face_nodes))
                    face_to_tets[sorted_face].append((t_idx, f_idx))

            self.surface_triangles = []

            # 4. Extract Surface Faces and Correct Orientation
            for sorted_face, occurrences in face_to_tets.items():
                if len(occurrences) == 1:
                    # Surface face found
                    t_idx, f_idx = occurrences[0]
                    tet_nodes = tets[t_idx] # Indices of nodes in this tet
                    local_indices = tet_faces_local_indices[f_idx]
                    
                    # Original nodes of the face (before sorting)
                    n0 = tet_nodes[local_indices[0]]
                    n1 = tet_nodes[local_indices[1]]
                    n2 = tet_nodes[local_indices[2]]
                    
                    # Coordinates for orientation check
                    v0 = initial_coords[n0]
                    v1 = initial_coords[n1]
                    v2 = initial_coords[n2]
                    
                    # Centroid of the parent tetrahedron
                    tet_coords = initial_coords[tet_nodes]
                    centroid = np.mean(tet_coords, axis=0)
                    
                    # Face geometric center and normal
                    face_center = (v0 + v1 + v2) / 3.0
                    normal = np.cross(v1 - v0, v2 - v0)
                    
                    # Check direction: dot(normal, centroid - face_center)
                    # Vector from face center to centroid is (centroid - face_center).
                    # If normal points roughly towards centroid (> 0), it is pointing INSIDE.
                    # We want normal pointing OUTSIDE.
                    if np.dot(normal, centroid - face_center) > 0:
                        # Flip winding
                        self.surface_triangles.append([n0, n2, n1])
                    else:
                        # Keep winding
                        self.surface_triangles.append([n0, n1, n2])

            print(f"Extracted {len(self.surface_triangles)} surface triangles with corrected orientation.")

        except Exception as e:
            print(f"Error loading tetra/coords for surface extraction: {e}")

    def load_all_data(self):
        self.force_node_positions = []
        # 一度にすべてのファイルの情報を読み込む
        for file_path in tqdm(self.file_list, desc="Loading feather files"):
            try:
                df = pd.read_feather(file_path)
                # time=20 のデータのみ抽出
                if 'time' in df.columns:
                    df_final = df[df['time'] == 20]
                else:
                    print(f"Warning: 'time' column not found in {file_path}. Using all data.")
                    df_final = df
                
                # node_id でソートしてインデックスと対応させる
                if 'node_id' in df_final.columns:
                    df_final = df_final.sort_values('node_id')

                # 座標データの取得
                if {'x', 'y', 'z'}.issubset(df_final.columns):
                    points = df_final[['x', 'y', 'z']].to_numpy()
                    self.data_list.append(points)
                else:
                    print(f"Warning: Coordinate columns (x, y, z) not found in {file_path}. Skipping.")
                    self.data_list.append(None)
                
                # force_node_id の取得と座標特定
                force_pos = None
                if 'force_node_id' in df_final.columns:
                    # 全行同じ値のはずなので先頭を取得、あるいはユニークな値
                    f_ids = df_final['force_node_id'].unique()
                    if len(f_ids) > 0:
                        force_node_id = f_ids[0]
                        # 該当ノードの行を検索
                        target_row = df_final[df_final['node_id'] == force_node_id]
                        if not target_row.empty:
                            force_pos = target_row[['x', 'y', 'z']].iloc[0].to_numpy()
                
                self.force_node_positions.append(force_pos)

            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                self.data_list.append(None)
                self.force_node_positions.append(None)

    def update_geometry(self):
        if not self.data_list:
            return
        
        # Mesh Update
        points = self.data_list[self.current_index]
        if points is not None:
            # Mesh更新
            self.mesh.vertices = o3d.utility.Vector3dVector(points)
            self.mesh.compute_vertex_normals() # ライティングのために法線を計算
            self.mesh.paint_uniform_color([0.2, 0.6, 0.8]) # 見やすい水色っぽい色
        else:
            self.mesh.vertices = o3d.utility.Vector3dVector(np.zeros((1, 3)))
            print(f"No data for file: {self.file_list[self.current_index]}")
        
        # Force Sphere Update
        target_pos = self.force_node_positions[self.current_index]
        if target_pos is not None:
            # 現在の中心位置からの差分で移動 (translate は相対移動)
            diff = target_pos - self.current_sphere_center
            self.force_sphere.translate(diff)
            self.current_sphere_center = target_pos
        else:
            # データがない場合はとりあえず原点へ or その場に留まる
            # 原点に戻す場合:
            diff = np.array([0.0, 0.0, 0.0]) - self.current_sphere_center
            self.force_sphere.translate(diff)
            self.current_sphere_center = np.array([0.0, 0.0, 0.0])

    def run(self):
        if not self.data_list:
            print("No data loaded. Exiting.")
            return

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=f"Dataset Checker - {os.path.basename(self.file_list[self.current_index])}")
        
        # ワイヤーフレーム表示の代わりにメッシュを表示
        vis.add_geometry(self.mesh)
        vis.add_geometry(self.force_sphere)

        def next_file(vis):
            self.current_index = (self.current_index + 1) % len(self.data_list)
            self.update_geometry()
            vis.update_geometry(self.mesh)
            vis.update_geometry(self.force_sphere)
            vis.poll_events()
            vis.update_renderer()
            print(f"Showing file [{self.current_index + 1}/{len(self.data_list)}]: {os.path.basename(self.file_list[self.current_index])}")

        def prev_file(vis):
            self.current_index = (self.current_index - 1 + len(self.data_list)) % len(self.data_list)
            self.update_geometry()
            vis.update_geometry(self.mesh)
            vis.update_geometry(self.force_sphere)
            vis.poll_events()
            vis.update_renderer()
            print(f"Showing file [{self.current_index + 1}/{len(self.data_list)}]: {os.path.basename(self.file_list[self.current_index])}")

        def mark_error_file(vis):
            error_file = "error_file_candidate.csv"
            current_file_path = self.file_list[self.current_index]
            current_file_name = os.path.basename(current_file_path)
            
            # Check if file exists to determine if header is needed
            file_exists = os.path.isfile(error_file)
            
            try:
                import csv
                with open(error_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(['file_index', 'file_name', 'full_path'])
                    
                    writer.writerow([self.current_index + 1, current_file_name, current_file_path])
                
                print(f"Logged error file: [{self.current_index + 1}] {current_file_name}")
            except Exception as e:
                print(f"Error writing to {error_file}: {e}")

        # キーコールバックの登録
        # D: 68, A: 65 (ASCII code)
        vis.register_key_callback(68, next_file) # D
        vis.register_key_callback(65, prev_file) # A
        
        # F: 70
        vis.register_key_callback(70, mark_error_file) # F

        print(f"Viewer started. Press 'D' for next, 'A' for previous.")
        print(f"Press 'F' to mark current file as error candidate.")
        print(f"Showing file [{self.current_index + 1}/{len(self.data_list)}]: {os.path.basename(self.file_list[self.current_index])}")

        vis.run()
        vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description='Visualize dataset deformation at time=20.')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing feather files.')
    args = parser.parse_args()

    checker = DatasetChecker(args.input_dir)
    checker.run()

if __name__ == "__main__":
    main()
