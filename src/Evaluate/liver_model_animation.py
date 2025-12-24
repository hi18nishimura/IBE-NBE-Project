import open3d as o3d
import pandas as pd
import numpy as np
import os
import time
import argparse

def load_tets(tet_file):
    """
    Loads tetrahedral connectivity.
    Returns numpy array of shape (N_tets, 4) with 0-based indices.
    """
    if not os.path.exists(tet_file):
        return None

    try:
        # Try to read as a normal CSV with header (e.g. element_id,n1,n2,n3,n4)
        try:
            tet_df = pd.read_csv(tet_file)
            if {"n1", "n2", "n3", "n4"}.issubset(set(tet_df.columns)):
                tets = tet_df[["n1", "n2", "n3", "n4"]].to_numpy(dtype=int) - 1
                return tets
            else:
                raise ValueError("tet file not in (n1..n4) CSV format")
        except Exception:
            # fallback: whitespace separated file without header
            tet_df = pd.read_csv(tet_file, sep="\s+", header=None, skiprows=1)
            tets = tet_df.iloc[:, 1:5].to_numpy(dtype=int) - 1
            return tets
    except Exception as e:
        print(f"Error loading tet file: {e}")
        return None

def build_triangles_from_tets(tets, coords):
    """
    Builds triangle faces from tetrahedrons, ensuring correct orientation based on centroids.
    """
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
            # Check orientation relative to centroid
            if np.dot(normal, centroid - face_center) > 0:
                face_vids = [tet[fi[0]], tet[fi[2]], tet[fi[1]]]
            else:
                face_vids = [tet[fi[0]], tet[fi[1]], tet[fi[2]]]
            faces.append(face_vids)
    return np.array(faces, dtype=int)

def main():
    parser = argparse.ArgumentParser(description="3D Animation of Liver Model from Feather")
    parser.add_argument("input_file", type=str, help="Path to the time-series Feather file")
    parser.add_argument("--tet_file", type=str, default="/workspace/dataset/liver_model_info/tetra_connectivity.csv", help="Path to tetra connectivity file")
    parser.add_argument("--fps", type=float, default=10.0, help="Frames per second")
    parser.add_argument("--save_video", type=str, default=None, help="Path to save video (e.g. animation.mp4). Requires ffmpeg.")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} not found.")
        return

    print(f"Loading data from {args.input_file}...")
    df = pd.read_feather(args.input_file)
    
    # Check columns
    required_cols = {'time', 'node_id', 'x', 'y', 'z'}
    if not required_cols.issubset(df.columns):
        print(f"Error: CSV must contain columns: {required_cols}")
        return

    # Get unique time steps
    times = sorted(df['time'].unique())
    print(f"Found {len(times)} time steps.")

    # Load topology using the first time step
    print("Building mesh topology...")
    df_t1 = df[df['time'] == times[0]].sort_values('node_id')
    coords_t1 = df_t1[['x', 'y', 'z']].to_numpy()
    
    tets = load_tets(args.tet_file)
    
    mesh = o3d.geometry.TriangleMesh()
    use_mesh = False
    
    if tets is not None:
        try:
            triangles = build_triangles_from_tets(tets, coords_t1)
            mesh.vertices = o3d.utility.Vector3dVector(coords_t1)
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([0.6, 0.6, 0.6])
            use_mesh = True
            
            # Create wireframe
            line_set = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
            line_set.paint_uniform_color([0.0, 0.0, 0.0])
        except Exception as e:
            print(f"Error building mesh: {e}")
            use_mesh = False
    
    if not use_mesh:
        print("Falling back to PointCloud visualization.")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords_t1)
        # Color mapping could be added here if needed
    
    # Visualization setup
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Liver Animation", width=800, height=600)
    
    if use_mesh:
        vis.add_geometry(mesh)
        vis.add_geometry(line_set)
    else:
        vis.add_geometry(pcd)

    # Optional: Set view control if needed
    # ctr = vis.get_view_control()
    # ctr.set_zoom(0.8)

    # Animation loop
    print("Starting animation...")
    
    # Pre-group data by time for faster access
    grouped = df.groupby('time')
    
    # Capture frames for video if requested
    frames = []

    try:
        while True: # Loop animation
            for t in times:
                start_time = time.time()
                
                # Get coordinates for current time step
                # Ensure sorting by node_id matches the topology
                df_t = grouped.get_group(t).sort_values('node_id')
                current_coords = df_t[['x', 'y', 'z']].to_numpy()
                
                if use_mesh:
                    mesh.vertices = o3d.utility.Vector3dVector(current_coords)
                    mesh.compute_vertex_normals()
                    # Update wireframe lines
                    line_set.points = o3d.utility.Vector3dVector(current_coords)
                    
                    vis.update_geometry(mesh)
                    vis.update_geometry(line_set)
                else:
                    pcd.points = o3d.utility.Vector3dVector(current_coords)
                    vis.update_geometry(pcd)
                    
                vis.poll_events()
                vis.update_renderer()
                
                if args.save_video and len(frames) < len(times):
                    # Capture image
                    image = vis.capture_screen_float_buffer(False)
                    frames.append(np.asarray(image))

                # Frame rate control
                elapsed = time.time() - start_time
                wait_time = max(0, (1.0 / args.fps) - elapsed)
                time.sleep(wait_time)
                
                # Pause at the last frame
                if t == times[-1]:
                    time.sleep(1.0)

                if not vis.poll_events():
                    break
            
            if not vis.poll_events():
                break
            
            # If saving video, we might only want to run once or handle it differently
            if args.save_video:
                break

    except KeyboardInterrupt:
        pass
    
    vis.destroy_window()

    if args.save_video and frames:
        print(f"Saving video to {args.save_video}...")
        try:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis('off')
            
            # frames are float [0,1], convert to uint8 for some writers or keep as is
            # matplotlib imshow handles float [0,1]
            
            ims = []
            for frame in frames:
                im = ax.imshow(frame, animated=True)
                ims.append([im])
            
            ani = animation.ArtistAnimation(fig, ims, interval=1000/args.fps, blit=True)
            ani.save(args.save_video, writer='ffmpeg')
            print("Video saved.")
        except ImportError:
            print("Error: matplotlib is required to save video.")
        except Exception as e:
            print(f"Error saving video: {e}")

if __name__ == "__main__":
    main()
