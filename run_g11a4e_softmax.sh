#!/usr/bin/env bash
set -euo pipefail

export PATH=/opt/cmake-3.22.6-linux-x86_64/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
hash -r

cd /workspace/notebooks/llama.cpp-ph2/build-cuda8-parent

cmake .. \
  -DGGML_CUDA8=ON \
  -DGGML_CUDA=OFF \
  -DBUILD_SHARED_LIBS=OFF

make -j1 ggml-cuda8-ggml-buffer-device-softmax-smoke
./bin/ggml-cuda8-ggml-buffer-device-softmax-smoke
