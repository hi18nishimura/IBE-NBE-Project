import argparse
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from typing import List, Tuple, Dict, Any

# --- ヘルパー関数 ---

def load_data_and_prepare_frames_for_comparison(input_csv: str, cmap_name: str = 'viridis') -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int], float, float]:
    """
    CSVファイルを読み込み、比較アニメーションのためのフレームを準備する。
    - 予測モデルのフレーム (frames_pred)
    - 正解モデルのフレーム (frames_true)
    - 共通の頂点カラー (colors)
    """
    df = pd.read_csv(input_csv)
    all_times = sorted(df['time'].unique())

    # 誤差マグニチュードの計算 (色付けのため、x_mae, y_mae, z_mae を使用)
    df['mag'] = np.sqrt(df['x_mae']**2 + df['y_mae']**2 + df['z_mae']**2)
    
    # カラーマップとスケールの決定
    vmin, vmax = df['mag'].min(), df['mag'].max()
    cmap = plt.get_cmap(cmap_name)
    
    frames_pred = []
    frames_true = []
    colors = []
    
    for time_step in all_times:
        df_time = df[df['time'] == time_step].sort_values(by='node_id').reset_index(drop=True)
        
        # 予測モデルの頂点座標
        positions_pred = df_time[['x', 'y', 'z']].to_numpy()
        frames_pred.append(positions_pred)

        # 正解モデルの頂点座標 (true_x, true_y, true_z)
        positions_true = df_time[['true_x', 'true_y', 'true_z']].to_numpy()
        frames_true.append(positions_true)

        # 頂点カラー (誤差マグニチュードに基づく)
        mags = df_time['mag'].to_numpy()
        if vmax > vmin:
            norm_mags = (mags - vmin) / (vmax - vmin)
        else:
            norm_mags = np.zeros_like(mags)
        
        vertex_colors = cmap(norm_mags)[:, :3] 
        colors.append(vertex_colors)
        
    return frames_pred, frames_true, colors, all_times, vmin, vmax

def create_geometry_from_frames(frames: List[np.ndarray], colors: List[np.ndarray], tet_file: str) -> Tuple[Any, Any, bool]:
    """file_path = f'node_coordinates_by_time_finetune.csv'
    フレームデータからメッシュまたは点群ジオメトリを作成する。
    """
    is_mesh = False
    try:
        if not tet_file or not os.path.exists(tet_file):
            raise FileNotFoundError

        tet_df = pd.read_csv(tet_file, sep='\s+', header=None, skiprows=1)
        tets = tet_df.iloc[:, 1:5].to_numpy() - 1
        
        initial_positions = frames[0]
        
        # 四面体から外面の三角形を抽出
        faces = []
        for tet_indices in tets:
            # 四面体の4頂点の座標を取得
            p = initial_positions[tet_indices]
            
            # 四面体の重心を計算
            centroid = np.mean(p, axis=0)

            # 四面体の4つの面を定義
            # 各面は [v0, v1, v2] の頂点インデックスで構成
            tet_faces_indices = [
                [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]
            ]

            for face_indices in tet_faces_indices:
                # 面の3頂点の座標を取得
                v0, v1, v2 = p[face_indices]
                
                # 面の法線ベクトルを計算 (v1-v0) x (v2-v0)
                normal = np.cross(v1 - v0, v2 - v0)
                
                # 面の中心から四面体の重心へのベクトル
                face_center = (v0 + v1 + v2) / 3.0
                center_to_centroid = centroid - face_center
                
                # 法線が内側を向いているかチェック (内積が正)
                if np.dot(normal, center_to_centroid) > 0:
                    # 法線が内側を向いている場合、頂点の順序を反転 (v1とv2を入れ替え)
                    # これにより、法線が外側を向く（反時計回り）
                    face_vertex_ids = [tet_indices[face_indices[0]], tet_indices[face_indices[2]], tet_indices[face_indices[1]]]
                else:
                    # 法線が外側を向いている場合、そのままの順序
                    face_vertex_ids = [tet_indices[face_indices[0]], tet_indices[face_indices[1]], tet_indices[face_indices[2]]]
                
                faces.append(face_vertex_ids)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(initial_positions)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors[0])
        mesh.orient_triangles()
        mesh.compute_vertex_normals()
        
        line_set = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        edge_color = [0.5, 0.5, 0.5]
        line_set.colors = o3d.utility.Vector3dVector([edge_color for _ in np.asarray(line_set.lines)])

        return mesh, line_set, True

    except (FileNotFoundError, IndexError, ValueError) as e:
        print(f"\n警告: ジオメトリファイルの問題 ({e})。点群として描画します。")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(frames[0])
        pcd.colors = o3d.utility.Vector3dVector(colors[0])
        return pcd, None, False

# --- メインのアニメーション関数 ---

def animate_comparison(input_csv: str,
                       tet_file: str = "",
                       cmap_name: str = 'viridis',
                       fps: float = 8.0,
                       no_shade: bool = False,
                       overlay: bool = False,
                       show_true: bool = True):
    """
    予測モデルと正解モデルを並べてアニメーション表示する。
    """
    
    # 1. データ準備
    frames_pred, frames_true, colors, all_times, vmin, vmax = load_data_and_prepare_frames_for_comparison(input_csv, cmap_name)
    
    # 2. ジオメトリの初期化
    # 予測モデル (左側)
    geom_pred, lines_pred, is_mesh_pred = create_geometry_from_frames(frames_pred, colors, tet_file)
    
    # 正解モデル (右側)
    geom_true = None
    lines_true = None
    is_mesh_true = False
    if show_true:
        geom_true, lines_true, is_mesh_true = create_geometry_from_frames(frames_true, colors, tet_file)

    # 3. 正解モデルを横にずらす
    offset = 0.0
    if show_true and geom_pred.has_vertices():
        bbox = geom_pred.get_axis_aligned_bounding_box()
        if overlay:
            offset = 0.0
        else:
            offset = bbox.get_max_bound()[0] - bbox.get_min_bound()[0] + 10  # X軸方向のオフセット
        geom_true.translate([offset, 0, 0])
        if lines_true:
            lines_true.translate([offset, 0, 0])

    # 4. 可視化ウィンドウの設定
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f'Prediction (Left) vs. Ground Truth (Right) - Time: {all_times[0]}')
    
    vis.add_geometry(geom_pred)
    if is_mesh_pred and lines_pred:
        vis.add_geometry(lines_pred)

    if show_true and geom_true is not None:
        vis.add_geometry(geom_true)
        if is_mesh_true and lines_true:
            vis.add_geometry(lines_true)

    # シェーディングを無効化
    if no_shade:
        render_option = vis.get_render_option()
        render_option.light_on = False

    # --- 5. アニメーションコールバック ---
    delay = 1.0 / fps
    idx = 0
    
    def animation_callback(vis):
        nonlocal idx
        
        # 頂点と色情報を更新
        current_colors = colors[idx]
        
        # オーバーレイモードの場合は色を薄くする（透明度効果）
        if overlay:
            alpha = 0.6  # 透明度係数 (0.0=透明, 1.0=不透明)
            # 色を白に近づけることで薄く見せる
            current_colors_pred = current_colors * alpha + np.array([1.0, 1.0, 1.0]) * (1 - alpha)
        else:
            current_colors_pred = current_colors
        
        current_colors_vector = o3d.utility.Vector3dVector(current_colors_pred)
        
        # 予測モデルの更新
        current_points_pred = o3d.utility.Vector3dVector(frames_pred[idx])
        if is_mesh_pred:
            geom_pred.vertices = current_points_pred
            geom_pred.vertex_colors = current_colors_vector
            geom_pred.compute_vertex_normals()
            vis.update_geometry(geom_pred)
            lines_pred.points = geom_pred.vertices
            vis.update_geometry(lines_pred)
        else: # PointCloud
            geom_pred.points = current_points_pred
            geom_pred.colors = current_colors_vector
            vis.update_geometry(geom_pred)

        if show_true and geom_true is not None:
            # 正解モデルの更新
            translated_points_true = frames_true[idx] + np.array([offset, 0, 0])
            current_points_true = o3d.utility.Vector3dVector(translated_points_true)

            # 正解モデルの色を灰色に設定
            gray_color = [0.8, 0.8, 0.8]
            
            # オーバーレイモードの場合は灰色も薄くする
            if overlay:
                alpha_true = 0.6  # 透明度係数
                gray_color_with_alpha = np.array(gray_color) * alpha_true + np.array([1.0, 1.0, 1.0]) * (1 - alpha_true)
                gray_color = gray_color_with_alpha.tolist()
            
            num_vertices = len(frames_true[idx])
            gray_colors_vector = o3d.utility.Vector3dVector(np.tile(gray_color, (num_vertices, 1)))

            if is_mesh_true:
                geom_true.vertices = current_points_true
                geom_true.vertex_colors = gray_colors_vector
                geom_true.compute_vertex_normals()
                vis.update_geometry(geom_true)
                if lines_true:
                    lines_true.points = geom_true.vertices
                    vis.update_geometry(lines_true)
            else: # PointCloud
                geom_true.points = current_points_true
                geom_true.colors = gray_colors_vector
                vis.update_geometry(geom_true)
        
        # ウィンドウタイトルの更新
        current_time = all_times[idx]
        vis.get_render_option().point_size = 5.0 # 点のサイズを調整
        # vis.update_renderer() # タイトル更新は register_animation_callback では直接できない
        
        if idx == 0:
            time.sleep(1)  # 最後のフレームで1秒停止
        else:
            time.sleep(delay)
        idx = (idx + 1) % len(frames_pred)
        return True

    # 6. アニメーション実行
    vis.register_animation_callback(animation_callback)
    vis.run()
    vis.destroy_window()

# --- 7. 実行ブロック ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare Predicted and Ground Truth Models")
    parser.add_argument("input_csv", type=str, help="ノード座標と誤差データを含むCSVファイルパス")
    parser.add_argument("--tet_file", type=str, default="/workspace/src/dataset/info/tetra_elements.txt", help="ジオメトリ（四面体要素）定義ファイルパス")
    parser.add_argument("--cmap", type=str, default="viridis", help="使用するmatplotlibカラーマップ名 (例: viridis, jet)")
    parser.add_argument("--fps", type=float, default=10.0, help="フレームレート (Frames Per Second)")
    parser.add_argument("--no_shade", action="store_true", help="シェーディングを無効にする")
    parser.add_argument("--overlay", action="store_true", help="モデルを重ねるかどうか")
    parser.add_argument("--hide_true", action="store_true", help="正解モデルを表示しない")
    
    args = parser.parse_args()
    
    animate_comparison(
        args.input_csv,
        args.tet_file,
        args.cmap,
        args.fps,
        args.no_shade,
        args.overlay,
        not args.hide_true
    )
