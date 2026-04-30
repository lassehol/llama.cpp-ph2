#!/usr/bin/env bash
set -euo pipefail

export PATH=/opt/cmake-3.22.6-linux-x86_64/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
hash -r

BUILD="/workspace/notebooks/llama.cpp-ph2/build-cuda8-parent"

cd "$BUILD"

cmake .. \
  -DGGML_CUDA8=ON \
  -DGGML_CUDA=OFF \
  -DBUILD_SHARED_LIBS=OFF

make -j1 ggml-cuda8-ggml-buffer-device-softmax-graph-smoke
./bin/ggml-cuda8-ggml-buffer-device-softmax-graph-smoke

make -j1 ggml-cuda8-ggml-buffer-device-softmax-smoke
./bin/ggml-cuda8-ggml-buffer-device-softmax-smoke

make -j1 ggml-cuda8-ggml-buffer-device-graph-smoke
./bin/ggml-cuda8-ggml-buffer-device-graph-smoke

make -j1 ggml-cuda8-ggml-buffer-device-scalar-smoke
./bin/ggml-cuda8-ggml-buffer-device-scalar-smoke

make -j1 ggml-cuda8-ggml-buffer-device-add-smoke
./bin/ggml-cuda8-ggml-buffer-device-add-smoke

make -j1 ggml-cuda8-dispatch-all-smoke
./bin/ggml-cuda8-dispatch-all-smoke

echo
echo "G11B-2 regression SUCCESS"