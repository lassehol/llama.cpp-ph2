#!/usr/bin/env bash
set -euo pipefail

export PATH=/opt/cmake-3.22.6-linux-x86_64/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
hash -r

ROOT="/workspace/notebooks/llama.cpp-ph2"
BUILD="$ROOT/build-cuda8-parent"

cd "$ROOT"

python3 ./G18A_writer.py

echo
echo "---- G18A source markers ----"
grep -n "mul_mat\|ggml_add\|Q8_0xF32_VEC->ADD_F32\|bias residency\|expected 2 graph" \
    ggml/src/ggml-cuda8/ggml-cuda8-ggml-backend-graph-builder-q8_0-add-smoke.cpp || true

cd "$BUILD"

cmake .. -DGGML_CUDA8=ON -DGGML_CUDA=OFF -DBUILD_SHARED_LIBS=OFF

make -j1 ggml-cuda8-ggml-backend-graph-builder-q8_0-add-smoke

./bin/ggml-cuda8-ggml-backend-graph-builder-q8_0-add-smoke
