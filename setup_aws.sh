#!/bin/bash

# CUDA 및 NVIDIA 드라이버 설치
sudo apt-get update
sudo apt-get install -y nvidia-cuda-toolkit

# Python 패키지 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install sentence-transformers
pip install datasets
pip install scikit-learn
pip install numpy 