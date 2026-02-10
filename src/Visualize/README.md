# 臓器モデルの可視化
## dataset_checker.py
データセットの全データの最終時刻の変形結果を可視化するプログラム。指定されたディレクトリのfeather形式ファイルを読み込んで、最終時刻（time=20）の各節点の座標を取得する。
取得された情報を基に3Dモデルをプロットする。Dボタンで次のファイルのプレビュー、Aボタンで前のファイルのプレビューを表示する。3DモデルはOpen3Dで実装する。
操作ですぐに切り替えられるように、一度にすべてのファイルの情報を読み込む処理を行う。
### 実行方法
`python3 /workspace/src/Visualize/dataset_checker.py \
--input_dir /workspace/dataset/bin/toy_all_model/train`