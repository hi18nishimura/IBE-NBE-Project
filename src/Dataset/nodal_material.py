import re
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from Dataset.nodal_tetra_elements import read_tetra


def read_material_sections(file_path: str = None) -> List[Dict[str, Any]]:
    """
    liver.dat 内の Isotropic/material セクションを抽出して各マテリアルの物性値と対応メッシュIDを返す。

    Returns a list of dicts with keys: 'name', 'E', 'nu', 'element_ids'
    """
    if file_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        file_path = repo_root / 'dataset' / 'liver_model' / 'liver.dat'

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")

    text = file_path.read_text()
    lines = text.splitlines()

    sections = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # detect header lines like: $Isotropic from material named "tumor_material"
        if 'Isotropic' in line and 'material named' in line:
            # extract material name in quotes
            m = re.search(r'"([^"]+)"', line)
            name = m.group(1) if m else None

            # properties expected two lines later per user's description
            prop_line_idx = i + 2
            E = None
            nu = None
            if prop_line_idx < len(lines):
                prop_line = lines[prop_line_idx]
                # extract floats from the line
                floats = [float(x) for x in re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', prop_line)]
                if len(floats) >= 2:
                    E = floats[0]
                    nu = floats[1]

            # find subsequent lines that likely contain mesh ids
            # collect up to a small window of lines and extract integers robustly
            element_ids = []
            j = prop_line_idx + 1
            collected = []
            max_lookahead = 12
            look = 0
            while j < len(lines) and look < max_lookahead:
                l = lines[j].strip()
                # stop if we hit another material header or a comment block
                if 'Isotropic' in l or l.startswith('$'):
                    break
                # if line contains any digit, collect it
                if re.search(r'\d', l):
                    collected.append(l)
                    # also advance i to this line so outer loop skips ahead
                    i = j
                # if an empty line and we've already collected something, stop
                if not l and collected:
                    break
                j += 1
                look += 1

            if collected:
                combined = ' '.join(collected)

                def extract_element_ids_from_text(s: str) -> List[int]:
                    # split on commas/spaces
                    toks = re.split(r'[,\s]+', s.strip())
                    out: List[int] = []
                    i = 0
                    while i < len(toks):
                        tok = toks[i]
                        if not tok:
                            i += 1
                            continue
                        if re.fullmatch(r'\d+', tok):
                            num = int(tok)
                            # handle range syntax: NUM TO NUM
                            if i + 2 < len(toks) and toks[i+1].upper() == 'TO' and re.fullmatch(r'\d+', toks[i+2]):
                                num2 = int(toks[i+2])
                                start, end = (num, num2) if num <= num2 else (num2, num)
                                out.extend(list(range(start, end + 1)))
                                i += 3
                                continue
                            else:
                                out.append(num)
                                i += 1
                                continue
                        else:
                            # skip non-numeric tokens (like 'TO' or material names)
                            i += 1
                    return out

                element_ids = [m for m in extract_element_ids_from_text(combined) if m != 0]

            sections.append({'name': name, 'E': E, 'nu': nu, 'element_ids': element_ids})
        i += 1

    return sections


def map_materials_to_nodes(file_path: str = None, output_csv: str = None) -> Path:
    """
    マテリアル定義（メッシュ単位）を読み取り、各節点に対応する物性値を割り当てて CSV に保存する。

    CSV のカラム: node_id, E, nu, material, element_ids

    Returns: Path to saved CSV
    """
    # read material sections
    sections = read_material_sections(file_path=file_path)

    # build element_id -> material mapping
    mesh_material = {}
    for sec in sections:
        name = sec.get('name')
        E = sec.get('E')
        nu = sec.get('nu')
        for mid in sec.get('element_ids', []):
            mesh_material[int(mid)] = {'material': name, 'E': E, 'nu': nu}

    # read tetra connectivity as mesh->nodes
    mesh_map = read_tetra(file_path=file_path, return_type='by_mesh')

    # node_id -> list of assignments
    node_assignments: Dict[int, List[Dict[str, Any]]] = {}
    for element_id, nodes in mesh_map.items():
        mat = mesh_material.get(int(element_id))
        if mat is None:
            # no material assigned to this mesh
            continue
        for nid in nodes:
            node_assignments.setdefault(int(nid), []).append({'element_id': int(element_id),
                                                               'material': mat.get('material'),
                                                               'E': mat.get('E'),
                                                               'nu': mat.get('nu')})

    # Prepare rows for CSV - if multiple assignments, we record element_ids list and pick first material
    rows = []
    for nid, assigns in sorted(node_assignments.items()):
        element_ids = [str(a['element_id']) for a in assigns]
        # pick first assignment as canonical (deterministic)
        first = assigns[0]
        rows.append({'node_id': nid,
                     'E': first.get('E'),
                     'nu': first.get('nu'),
                     'material': first.get('material'),
                     'element_ids': ';'.join(element_ids)})

    df = pd.DataFrame(rows, columns=['node_id', 'E', 'nu', 'material', 'element_ids'])

    # output path default
    if output_csv is None:
        repo_root = Path(__file__).resolve().parents[2]
        output_csv = repo_root / 'dataset' / 'liver_model_info' / 'node_materials.csv'

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"節点ごとの物性値を保存しました: {output_csv}")
    return output_csv


if __name__ == '__main__':
    map_materials_to_nodes()
