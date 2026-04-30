#!/usr/bin/env bash
set -euo pipefail

export PATH=/opt/cmake-3.22.6-linux-x86_64/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
hash -r

ROOT="/workspace/notebooks/llama.cpp-ph2"
BUILD="$ROOT/build-cuda8-parent"

cd "$ROOT"
python3 ./G16A_fix_cmake_link_ggml_cxx17_v6.py

cd "$BUILD"
cmake .. -DGGML_CUDA8=ON -DGGML_CUDA=OFF -DBUILD_SHARED_LIBS=OFF
make -j1 ggml-cuda8-ggml-backend-graph-builder-add-smoke
./bin/ggml-cuda8-ggml-backend-graph-builder-add-smoke
