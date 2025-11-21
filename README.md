# IBE-NBE-Project
IBEとNBEのデータセット・ネットワーク・可視化を行うプログラムを作成する。

## データセットの開発
1. liver.datファイルを準備してIBE-NBE-Project\dataset\liver_modelに保存する。
2. liver.datファイルを編集して、実験条件（作用点・変位）を変更したliver.datの複製ファイルを作成する
3. 複製されたファイルをMarcで解析する。ディレクトリに含まれるdatファイルをMarcに渡すbatファイルで解析する。（大体8割程度は解析に成功する）
4. Marcの解析ファイル(.out)をパースして、各時刻・各節点の変位と応力を取り出す。ただし、応力はメッシュ単位での応力になっているため、節点ごとの応力に計算しなおす必要がある。
5. 4で各時刻・各節点の応力をもとに特徴量を新しく計算する。現在は、その節点と作用点までの距離ベクトル・作用点の変位ベクトル・最も近い固定節点までの距離ベクトルの計算が実装されている。
6. 注目節点の座標系に変換したデータセット、データセット全体の最大値、注目節点と隣接節点ごとの最大値などを計算する。（最大値は正規化で使用する）

### Marc解析結果の一括パース

`dataset_make.py` に `marc_parse` サブコマンドを追加しており、指定ディレクトリ内の Marc `.out` ファイルをまとめて処理できます。内部では `feature_disp_nodal.py`・`feature_stress_elements.py`・`feature_stress_nodal.py` を順番に実行し、以下のディレクトリへCSVを保存します。

- `disp_nodal/`: `*_disp_with_coords.csv`
- `stress_elements/`: `*_stress_elements.csv`
- `stress_nodal/`: `*_nodal.csv`, `*_nodal_average.csv`
- `all_feature/`: 変位+応力を結合した `*_nodal_features.csv`, `*_nodal_average_features.csv`

`marc_parse` も Hydra ベースになっているため、設定は `config/dataset/marc_parse.yaml` を基準に Override します。デフォルトの `input_dir` は `dataset/marc/sample` に向いています。

基本コマンド例:

```
python dataset_make.py marc_parse input_dir=dataset/marc/sample output_dir=parsed_outputs glob="*.out"
```

よく使う Override 例:

- `glob="**/*.out"` : サブディレクトリも含めて探索
- `skip_existing=true` : すべての出力が揃っているファイルはスキップ
- `connectivity=dataset/liver_model_info/tetra_connectivity.csv` : 接続情報を切り替え
- `force_condition=liver_forced` / `fixed_csv=dataset/.../fixed_nodes.csv` : feature_disp_nodal 用の設定差し替え
- `merge_how=left` : `all_feature` 生成時の結合方法を `left` に変更

## NBEの学習
現在、NBEは節点ごとにネットワークを作成している。