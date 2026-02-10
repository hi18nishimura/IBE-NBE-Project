import open3d as o3d
import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def create_mesh_from_coords(coords: np.ndarray, tet_file: str):
    """
    Creates a TriangleMesh from coordinates and a tetrahedral connectivity file.
    Falls back to PointCloud if tet_file is invalid or not found.
    
    Args:
        coords: (N, 3) coordinates
        tet_file: path to tetrahedral connectivity file
    """
    if tet_file and os.path.exists(tet_file):
        try:
            # Try to read as a normal CSV with header (e.g. element_id,n1,n2,n3,n4)
            try:
                tet_df = pd.read_csv(tet_file)
                # common header names: n1,n2,n3,n4
                if {"n1", "n2", "n3", "n4"}.issubset(set(tet_df.columns)):
                    tets = tet_df[["n1", "n2", "n3", "n4"]].to_numpy(dtype=int) - 1
                else:
                    # not the expected CSV header: raise to fall back to whitespace parse
                    raise ValueError("tet file not in (n1..n4) CSV format")
            except Exception:
                # fallback: whitespace separated file without header (old style)
                tet_df = pd.read_csv(tet_file, sep="\s+", header=None, skiprows=1)
                tets = tet_df.iloc[:, 1:5].to_numpy(dtype=int) - 1

            faces = []
            
            for tet in tets:
                # tet indices -> form 4 triangular faces
                tet_faces_idx = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
                p = coords[tet]
                centroid = np.mean(p, axis=0)
                
                current_tet_faces = []
                for fi in tet_faces_idx:
                    v0, v1, v2 = p[fi]
                    normal = np.cross(v1 - v0, v2 - v0)
                    face_center = (v0 + v1 + v2) / 3.0
                    
                    # Orient face outward
                    if np.dot(normal, centroid - face_center) > 0:
                        face_vids = [tet[fi[0]], tet[fi[2]], tet[fi[1]]]
                    else:
                        face_vids = [tet[fi[0]], tet[fi[1]], tet[fi[2]]]
                    
                    current_tet_faces.append(face_vids)
                
                faces.extend(current_tet_faces)

            # Base Mesh
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(coords)
            mesh.triangles = o3d.utility.Vector3iVector(np.array(faces, dtype=int))
            mesh.compute_vertex_normals()
            mesh.orient_triangles()

            line_set = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
            
            return mesh, line_set, True
            
        except Exception as e:
            print(f"Error loading tet file: {e}; falling back to point cloud.")
            pass

    # fallback: point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    return pcd, None, False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Visualize Liver Model with Distance Coloring")
    parser.add_argument("--animate", action="store_true", help="Rotate the model around Y axis")
    args = parser.parse_args()

    # File paths
    coord_file = "/workspace/dataset/liver_model_info/liver_coordinates.csv"
    dist_file = "/workspace/dataset/liver_model_info/node_distances.csv"
    tet_file = "/workspace/dataset/liver_model_info/tetra_connectivity.csv"

    # Check if files exist
    if not os.path.exists(coord_file):
        print(f"Error: {coord_file} not found.")
        return
    if not os.path.exists(dist_file):
        print(f"Error: {dist_file} not found.")
        return

    # Load coordinates
    print(f"Loading coordinates from {coord_file}...")
    df_coords = pd.read_csv(coord_file)
    # Ensure sorted by node_id to match index if node_ids are 1-based sequential
    df_coords = df_coords.sort_values('node_id')
    
    # Load distances
    print(f"Loading distances from {dist_file}...")
    df_dist = pd.read_csv(dist_file)
    
    # Merge coordinates and distances
    # Left join to df_coords to ensure we have all coordinates even if distance is missing
    df_merged = pd.merge(df_coords, df_dist[['node_id', 'distance_from_fixed_center']], on='node_id', how='left')
    
    coords = df_merged[['x', 'y', 'z']].to_numpy()

    # Move model to origin based on centroid
    centroid_original = np.mean(coords, axis=0)
    print(f"Shifting model centroid {centroid_original} to origin.")
    coords = coords - centroid_original

    # Initial rotation: 90 degrees around Y axis
    print("Applying initial rotation: 90 degrees around Y axis.")
    R_init = o3d.geometry.Geometry3D.get_rotation_matrix_from_xyz((np.radians(90), 0, 0))
    # Apply rotation R to coords (N, 3). Since coords are row vectors, use coords @ R.T
    coords = np.dot(coords, R_init.T)
    
    # Handle missing distances if any (fill with 0 or min/max)
    distances = df_merged['distance_from_fixed_center'].fillna(0).to_numpy()

    # Prepare colors based on distance
    d_min = np.min(distances)
    d_max = np.max(distances)
    print(f"Distance range: {d_min:.2f} to {d_max:.2f}")

    # Use matplotlib for colormap
    # 'jet' is common for heatmaps (blue=low, red=high)
    norm = plt.Normalize(vmin=d_min, vmax=d_max)
    cmap = plt.get_cmap('viridis')
    
    colors = cmap(norm(distances))[:, :3] # Get RGB only (drop Alpha)

    # Create PointCloud for nodes (Primary visualization for "colored nodes")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # List to keep track of geometries
    base_geometries = [pcd]
    highlight_geometries = []

    # Create geometry for structure
    geom, lines, is_mesh = create_mesh_from_coords(coords, tet_file)

    if is_mesh:
        # geom is TriangleMesh
        # Apply the same colors to the mesh vertices so the surface is colored by distance
        geom.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        base_geometries.append(geom)
        
        # Add wireframe
        if lines:
            lines.paint_uniform_color([0.0, 0.0, 0.0]) # Black lines
            base_geometries.append(lines)
        
        print("Visualizing Mesh (Distance Colored)...")
    else:
        # geom is PointCloud (fallback)
        print("Visualizing PointCloud (Distance Colored)...")

    # Visualization
    try:
        # Save to file
        output_ply = "liver_model_dist.ply"
        o3d.io.write_point_cloud(output_ply, pcd)
        print(f"Point cloud saved to {output_ply}")
        
        print("Attempting to open visualization window...")
        
        # Use Visualizer for simple view
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Liver Model Distance", width=800, height=600)
        
        # Add all geometries initially
        all_geometries = base_geometries
        for g in all_geometries:
            vis.add_geometry(g)
            
        # Increase point size
        opt = vis.get_render_option()
        opt.point_size = 5.0
        
        if args.animate:
            # Calculate centroid of the coordinates
            centroid = np.mean(coords, axis=0)
            print(f"Animation enabled. Rotating around centroid: {centroid}")
            
            # 1 degree per frame
            angle = 2*np.radians(1.0) 
            #R = o3d.geometry.Geometry3D.get_rotation_matrix_from_xyz((angle, 0, 0))
            R = o3d.geometry.Geometry3D.get_rotation_matrix_from_xyz((0, angle, 0))
            #R = o3d.geometry.Geometry3D.get_rotation_matrix_from_xyz((0, 0, angle))
            
            while True:
                for g in all_geometries:
                    g.rotate(R, center=centroid)
                    vis.update_geometry(g)
                
                if not vis.poll_events():
                    break
                vis.update_renderer()
        else:
            vis.run()
            
        vis.destroy_window()
        
    except Exception as e:
        print(f"Error during visualization: {e}")

if __name__ == "__main__":
    main()
