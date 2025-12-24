import open3d as o3d
import pandas as pd
import numpy as np
import os

def create_mesh_from_coords(coords: np.ndarray, tet_file: str):
    """
    Creates a TriangleMesh from coordinates and a tetrahedral connectivity file.
    Falls back to PointCloud if tet_file is invalid or not found.
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
                for fi in tet_faces_idx:
                    v0, v1, v2 = p[fi]
                    normal = np.cross(v1 - v0, v2 - v0)
                    face_center = (v0 + v1 + v2) / 3.0
                    if np.dot(normal, centroid - face_center) > 0:
                        face_vids = [tet[fi[0]], tet[fi[2]], tet[fi[1]]]
                    else:
                        face_vids = [tet[fi[0]], tet[fi[1]], tet[fi[2]]]
                    faces.append(face_vids)

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
    # File paths
    coord_file = "/workspace/dataset/liver_model_info/liver_coordinates.csv"
    fixed_file = "/workspace/dataset/liver_model_info/fixed_nodes.csv"
    tet_file = "/workspace/dataset/liver_model_info/tetra_connectivity.csv"

    # Check if files exist
    if not os.path.exists(coord_file):
        print(f"Error: {coord_file} not found.")
        return
    if not os.path.exists(fixed_file):
        print(f"Error: {fixed_file} not found.")
        return

    # Load coordinates
    print(f"Loading coordinates from {coord_file}...")
    df_coords = pd.read_csv(coord_file)
    # Ensure sorted by node_id to match index if node_ids are 1-based sequential
    df_coords = df_coords.sort_values('node_id')
    coords = df_coords[['x', 'y', 'z']].to_numpy()
    
    # Load fixed nodes
    print(f"Loading fixed nodes from {fixed_file}...")
    df_fixed = pd.read_csv(fixed_file)

    # Create a set of fixed node IDs
    if df_fixed['is_fixed'].dtype == object:
        fixed_node_ids = set(df_fixed[df_fixed['is_fixed'].astype(str) == 'True']['node_id'])
    else:
        fixed_node_ids = set(df_fixed[df_fixed['is_fixed'] == True]['node_id'])

    print(f"Found {len(fixed_node_ids)} fixed nodes.")

    # Prepare colors
    colors = []
    color_normal = [0.7, 0.7, 0.7]  # Gray
    color_fixed = [1.0, 0.0, 0.0]   # Red

    for node_id in df_coords['node_id']:
        if node_id in fixed_node_ids:
            colors.append(color_fixed)
        else:
            colors.append(color_normal)
    
    colors = np.array(colors)

    # Create PointCloud for nodes (Primary visualization for "colored nodes")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    geometries = [pcd]

    # Create geometry for structure
    geom, lines, is_mesh = create_mesh_from_coords(coords, tet_file)

    if is_mesh:
        # geom is TriangleMesh
        # Make the mesh surface neutral (e.g. white/light gray) so nodes stand out
        geom.paint_uniform_color([0.9, 0.9, 0.9])
        geometries.append(geom)
        
        # Add wireframe
        if lines:
            lines.paint_uniform_color([0.0, 0.0, 0.0]) # Black lines
            geometries.append(lines)
            
        print("Visualizing Mesh (Neutral) with Fixed Nodes (Red Points)...")
    else:
        # geom is PointCloud (fallback)
        # We already have pcd, so we don't need geom.
        print("Visualizing PointCloud with Fixed Nodes (Red)...")

    # Visualization
    try:
        # Save to file
        output_ply = "liver_model.ply"
        o3d.io.write_point_cloud(output_ply, pcd)
        print(f"Point cloud saved to {output_ply}")
        
        print("Attempting to open visualization window...")
        
        # Use Visualizer to control render options (point size)
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Liver Model 3D", width=800, height=600)
        
        for g in geometries:
            vis.add_geometry(g)
            
        # Increase point size
        opt = vis.get_render_option()
        opt.point_size = 20.0
        
        vis.run()
        vis.destroy_window()
        
    except Exception as e:
        print(f"Error during visualization: {e}")

if __name__ == "__main__":
    main()
