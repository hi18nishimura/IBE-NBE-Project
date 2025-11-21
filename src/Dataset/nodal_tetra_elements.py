import pandas as pd
from pathlib import Path


def read_tetra_connectivity(file_path: str = None):
    """
    liver.dat の CONNECTIVITY セクションを読み込み、テトラメッシュ情報を DataFrame で返す。

    期待フォーマットの例:
        CONNECTIVITY
            140,  ...   # 次の行の一番最初の整数が要素数
            1,  157,   36,   64,   59,   62,  141,

    上記の例では最初の整数が element_id、3~6番目の整数が節点ID (4つ) となるため
    CSV は columns=['element_id','n1','n2','n3','n4'] で出力される。

    Parameters
    ----------
    file_path : str, optional
        .dat ファイルのパス。None の場合はリポジトリ内の
        dataset/liver_model/liver.dat を使用します。

    Returns
    -------
    pd.DataFrame
        columns=['element_id','n1','n2','n3','n4']
    """
    if file_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        file_path = repo_root / 'dataset' / 'liver_model' / 'liver.dat'

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")

    with open(file_path, 'r') as f:
        lines = f.readlines()

    conn_idx = None
    for i, line in enumerate(lines):
        if 'CONNECTIVITY' in line.strip():
            conn_idx = i
            break

    if conn_idx is None:
        raise ValueError('CONNECTIVITY セクションが見つかりません')

    # CONNECTIVITY の次の行に最初の整数として要素数があると仮定
    info_line = lines[conn_idx + 1].strip()
    info_parts = [p.strip() for p in info_line.split(',') if p.strip()]
    try:
        num_elements = int(info_parts[0])
    except Exception:
        raise ValueError('CONNECTIVITY セクションの要素数を解析できませんでした')

    rows = []
    start_line = conn_idx + 2
    read_count = 0
    i = start_line
    # 要素数分、非空行・非コメント行を読み取る
    while read_count < num_elements and i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith('$'):
            continue

        parts = [p.strip() for p in line.split(',') if p.strip()]
        # 少なくとも 6 個以上の数字があれば処理（例に倣う）
        if len(parts) >= 6:
            try:
                element_id = int(parts[0])
                # 3~6番目の整数 (1-based) -> parts[2:6]
                n1 = int(parts[2])
                n2 = int(parts[3])
                n3 = int(parts[4])
                n4 = int(parts[5])
            except Exception:
                # 形式違いならスキップ
                continue
            rows.append([element_id, n1, n2, n3, n4])
            read_count += 1
        else:
            # 行が分割されている等の可能性があるが、ここでは簡潔にスキップ
            continue

    df = pd.DataFrame(rows, columns=['element_id', 'n1', 'n2', 'n3', 'n4'])
    return df


def save_tetra_connectivity_to_csv(df: pd.DataFrame, output_path: str = None):
    """
    テトラ接続情報 DataFrame を CSV として保存する。デフォルトは
    dataset/liver_model_info/tetra_connectivity.csv
    """
    if output_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        output_path = repo_root / 'dataset' / 'liver_model_info' / 'tetra_connectivity.csv'

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"テトラ接続情報を保存しました: {output_path}")
    return output_path


if __name__ == '__main__':
    df = read_tetra_connectivity()
    print(f"読み込んだテトラ数: {len(df)}")
    print(df.head())
    save_tetra_connectivity_to_csv(df)


def read_tetra(file_path: str = None, return_type: str = 'dataframe'):
    """
    読み込みの便利ラッパー。ファイルパスを受け取り、DataFrame または辞書を返す。

    Parameters
    ----------
    file_path : str, optional
        .dat ファイルのパス。None の場合はリポジトリの
        dataset/liver_model/liver.dat を使用します（既定）。
    return_type : str, optional
        - 'dataframe' : pandas.DataFrame を返す（既定）。
        - 'dict_by_node' or 'by_node' : node_id をキーにした辞書を返す。
            形式は {node_id: [element_id1, element_id2, ...], ...}。
        - 'dict_by_mesh' or 'by_mesh' : element_id をキーにした辞書を返す。
            形式は {element_id: [n1,n2,n3,n4], ...}。

    Returns
    -------
    pd.DataFrame or dict
    """
    df = read_tetra_connectivity(file_path=file_path)

    if return_type == 'dataframe':
        return df
    elif return_type in ('dict_by_node', 'by_node'):
        # node_id -> list of element_ids
        node_map = {}
        for _, row in df.iterrows():
            element_id = int(row['element_id'])
            for col in ['n1', 'n2', 'n3', 'n4']:
                nid = int(row[col])
                node_map.setdefault(nid, []).append(element_id)
        return node_map
    elif return_type in ('dict_by_mesh', 'by_mesh'):
        mesh_map = {int(row['element_id']): [int(row['n1']), int(row['n2']), int(row['n3']), int(row['n4'])]
                    for _, row in df.iterrows()}
        return mesh_map
    else:
        raise ValueError("return_type must be one of: 'dataframe', 'dict_by_node' (or 'by_node'), 'dict_by_mesh' (or 'by_mesh')")
