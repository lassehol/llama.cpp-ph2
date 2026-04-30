#!/usr/bin/env bash
set -euo pipefail

export PATH=/opt/cmake-3.22.6-linux-x86_64/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
hash -r

ROOT="/workspace/notebooks/llama.cpp-ph2"
BUILD="$ROOT/build-cuda8-parent"

cd "$ROOT"

python3 ./G17A2_patch_mul_mat_graph_compute.py

grep -n "GGML_OP_MUL_MAT\|MUL_MAT_Q8_0xF32_VEC\|GGML_CUDA8_OP_MUL_MAT_Q8_0_F32_VEC" \
    ggml/src/ggml-cuda8/ggml-cuda8-ggml-backend.cpp

cd "$BUILD"

cmake .. -DGGML_CUDA8=ON -DGGML_CUDA=OFF -DBUILD_SHARED_LIBS=OFF

make -j1 ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke

./bin/ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke
