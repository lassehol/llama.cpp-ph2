#!/usr/bin/env bash
set -euo pipefail

export PATH=/opt/cmake-3.22.6-linux-x86_64/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
hash -r

ROOT="/workspace/notebooks/llama.cpp-ph2"
BUILD="$ROOT/build-cuda8-parent"

cd "$ROOT"

python3 ./G21A_writer.py

echo
echo "---- G21A source markers ----"
grep -n "mul_mat\|ggml_add\|residual residency\|residual branch isolation\|RESIDUAL_ADD_F32\|expected 2 graph" \
    ggml/src/ggml-cuda8/ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-smoke.cpp || true

cd "$BUILD"

cmake .. -DGGML_CUDA8=ON -DGGML_CUDA=OFF -DBUILD_SHARED_LIBS=OFF

make -j1 ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-smoke

./bin/ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-smoke
