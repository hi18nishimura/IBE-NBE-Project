import pandas as pd
from pathlib import Path


def read_coordinates_from_dat(file_path: str = None):
    """
    liver.datファイルからCOORDINATESセクションを読み込み、節点座標データを抽出する関数
    
    Parameters
    ----------
    file_path : str
        読み込むdatファイルのパス（デフォルト: liver.datファイルのパス）
    
    Returns
    -------
    pd.DataFrame or dict
        節点番号、X座標、Y座標、Z座標を含むデータフレーム
        カラム名: ['node_id', 'x', 'y', 'z']
    """
    # デフォルト: リポジトリの dataset/liver_model/liver.dat を使う
    if file_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        default_path = repo_root / 'dataset' / 'liver_model' / 'liver.dat'
        file_path = default_path

    # ファイルパスをPathオブジェクトに変換
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"ファイルが見つかりません: {file_path}\n"
            f"デフォルトの想定場所は: {repo_root / 'dataset' / 'liver_model' / 'liver.dat'} です。"
            " 必要なら正しいファイルパスを引数 'file_path' に渡してください。"
        )
    
    # ファイルを読み込む
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # COORDINATESの行を探す
    coordinates_line_index = None
    for i, line in enumerate(lines):
        if 'COORDINATES' in line.strip():
            coordinates_line_index = i
            break
    
    if coordinates_line_index is None:
        raise ValueError("COORDINATESセクションが見つかりません")
    
    # COORDINATESの次の行から節点数を取得
    # フォーマット: "    3,  140,    5,    0,"
    # 左から2番目の数字が節点数
    info_line = lines[coordinates_line_index + 1].strip()
    info_parts = [part.strip() for part in info_line.split(',') if part.strip()]
    num_nodes = int(info_parts[1])  # 左から2番目の数字
    
    print(f"節点数: {num_nodes}")
    
    # 座標データを格納するリスト
    coordinates_data = []
    
    # COORDINATESの2行後から節点データを読み込む
    start_line = coordinates_line_index + 2
    for i in range(start_line, start_line + num_nodes):
        line = lines[i].strip()
        if not line or line.startswith('$'):  # 空行やコメント行をスキップ
            continue
        
        # データを分割
        parts = [part.strip() for part in line.split(',') if part.strip()]
        
        if len(parts) >= 4:
            node_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            coordinates_data.append([node_id, x, y, z])
    
    # データフレームに変換
    df = pd.DataFrame(coordinates_data, columns=['node_id', 'x', 'y', 'z'])
    
    return df


def save_coordinates_to_csv(df: pd.DataFrame, output_path: str = None):
    """
    座標データをCSVファイルとして保存する関数
    
    Parameters
    ----------
    df : pd.DataFrame
        保存する座標データのデータフレーム
    output_path : str, optional
        保存先のパス（デフォルト: liver.datと同じディレクトリにliver_coordinates.csvとして保存）
    """
    if output_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        output_path = repo_root / 'dataset' / 'liver_model_info' / 'liver_coordinates.csv'

    output_path = Path(output_path)
    
    # ディレクトリが存在しない場合は作成
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # CSVファイルとして保存
    df.to_csv(output_path, index=False)
    print(f"座標データを保存しました: {output_path}")
    
    return output_path


def read_coordinates(file_path: str = None, return_type: str = 'dataframe'):
    """
    指定した .dat ファイルから節点座標を読み込み、DataFrame / リスト辞書 / node_id をキーにした辞書で返すユーティリティ関数。

    Parameters
    ----------
    file_path : str, optional
        読み込む dat ファイルのパス。None の場合はリポジトリ内の
        `dataset/liver_model/liver.dat` を既定として使用します。
    return_type : str, optional
        賞味期限:
        - 'dataframe' : pandas.DataFrame を返します（既定）。
        - 'dict' : list[dict]（pandas.DataFrame.to_dict(orient='records')）を返します。
        - 'by_id' or 'dict_by_id' : node_id をキーにした辞書を返します。形式は {node_id: {'x':..., 'y':..., 'z':...}}。

    Returns
    -------
    pd.DataFrame or list[dict] or dict
        指定された形式で節点座標を返します。

    Raises
    ------
    ValueError
        `return_type` がサポート外の値の場合に発生します。
    """
    # 内部の既存関数を利用して DataFrame を取得
    df = read_coordinates_from_dat(file_path=file_path)

    if return_type == 'dataframe':
        return df
    elif return_type == 'dict':
        return df.to_dict(orient='records')
    elif return_type in ('by_id', 'dict_by_id'):
        # node_id をキーにした辞書にする
        by_id = {
            int(row['node_id']): {'x': float(row['x']), 'y': float(row['y']), 'z': float(row['z'])}
            for _, row in df.iterrows()
        }
        return by_id
    else:
        raise ValueError("return_type must be one of: 'dataframe', 'dict', 'by_id' (or 'dict_by_id')")


def main():
    """
    メイン関数: liver.datファイルから座標データを読み込み、CSVファイルとして保存
    """
    # 座標データを読み込む
    df = read_coordinates_from_dat()
    
    # データフレームの情報を表示
    print(f"\n読み込んだデータの形状: {df.shape}")
    print(f"\nデータの先頭5行:")
    print(df.head())
    print(f"\nデータの末尾5行:")
    print(df.tail())
    
    # CSVファイルとして保存
    save_coordinates_to_csv(df)
    
    return df


if __name__ == "__main__":
    main()
