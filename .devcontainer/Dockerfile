# Python 3.10ベースイメージを使用
# FROM python:3.10-slim
# CUDAとcuDNNを含むNVIDIAの公式イメージを使用
# ここではCUDA 12.6とcuDNN 9を使用（必要に応じてバージョンを調整してください）
# GPUを使うために調整
# FROM nvidia/cuda:12.1-cudnn-runtime-ubuntu22.04
# 73サーバに合わせたCUDAとcuDNNのバージョン
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# 作業ディレクトリを設定
WORKDIR /workspace

# システムパッケージのアップデートと必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1 \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# requirements.txtをコピー
COPY requirements.txt .

# 基本的なPythonパッケージのインストール
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# PyTorch CUDA版パッケージのインストール（PyTorchの専用インデックスを使用）
RUN pip install --no-cache-dir \
    torch==2.7.1 \
    torchaudio==2.7.1 \
    torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu126

# 開発用の追加設定
RUN pip install --no-cache-dir jupyter jupyterlab

# ポート9126を公開（Jupyter用）
EXPOSE 9129
EXPOSE 9130
EXPOSE 9131

# デフォルトコマンド
CMD ["bash"]