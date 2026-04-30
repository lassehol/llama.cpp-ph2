#!/usr/bin/env bash
set -euo pipefail
export PATH=/opt/cmake-3.22.6-linux-x86_64/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
hash -r
BUILD="/workspace/notebooks/llama.cpp-ph2/build-cuda8-parent"
cd "$BUILD"
cmake .. -DGGML_CUDA8=ON -DGGML_CUDA=OFF -DBUILD_SHARED_LIBS=OFF
run_target() {
    target="$1"; exe="$2"
    echo; echo "---- build: $target ----"; make -j1 "$target"
    echo "---- run: ./bin/$exe ----"; "./bin/$exe"
}
run_target ggml-cuda8-ggml-backend-graph-builder-add-smoke ggml-cuda8-ggml-backend-graph-builder-add-smoke
run_target ggml-cuda8-ggml-backend-graph-builder-add-mul-smoke ggml-cuda8-ggml-backend-graph-builder-add-mul-smoke
run_target ggml-cuda8-ggml-backend-graph-builder-softmax-smoke ggml-cuda8-ggml-backend-graph-builder-softmax-smoke
run_target ggml-cuda8-ggml-backend-graph-compute-softmax-smoke ggml-cuda8-ggml-backend-graph-compute-softmax-smoke
run_target ggml-cuda8-ggml-backend-graph-compute-attnlike-smoke ggml-cuda8-ggml-backend-graph-compute-attnlike-smoke
run_target ggml-cuda8-dispatch-all-smoke ggml-cuda8-dispatch-all-smoke
echo; echo "G16C graph-builder softmax regression SUCCESS"
