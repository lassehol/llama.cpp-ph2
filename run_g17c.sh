#!/usr/bin/env bash
set -euo pipefail

export PATH=/opt/cmake-3.22.6-linux-x86_64/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
hash -r

ROOT="/workspace/notebooks/llama.cpp-ph2"
BUILD="$ROOT/build-cuda8-parent"

cd "$ROOT"

python3 ./G17C_patch_packed_q8_ref.py

echo
echo "---- packed Q8_0 markers ----"
grep -n "fill_f32_matrix\|pack_q8_0\|packed Q8_0" \
    ggml/src/ggml-cuda8/ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke.cpp

cd "$BUILD"

cmake .. -DGGML_CUDA8=ON -DGGML_CUDA=OFF -DBUILD_SHARED_LIBS=OFF

make -j1 ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke

./bin/ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke
