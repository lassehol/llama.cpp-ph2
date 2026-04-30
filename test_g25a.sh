#!/usr/bin/env bash
set -euo pipefail

export PATH=/opt/cmake-3.22.6-linux-x86_64/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
hash -r

BUILD="/workspace/notebooks/llama.cpp-ph2/build-cuda8-parent"
cd "$BUILD"

cmake .. -DGGML_CUDA8=ON -DGGML_CUDA=OFF -DBUILD_SHARED_LIBS=OFF

run_target() {
    target="$1"
    exe="$2"
    echo
    echo "---- build: $target ----"
    make -j1 "$target"
    echo "---- run: ./bin/$exe ----"
    "./bin/$exe"
}

run_target ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-isolation-smoke ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-isolation-smoke
run_target ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-smoke ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-smoke
run_target ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-sumrows-smoke ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-sumrows-smoke
run_target ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-softmax-sumrows-smoke ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-softmax-sumrows-smoke
run_target ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke
run_target ggml-cuda8-dispatch-all-smoke ggml-cuda8-dispatch-all-smoke

echo
echo "G25A Q8_0 MUL_MAT -> MUL_SCALAR -> residual ADD -> SOFTMAX -> SUM_ROWS input-isolation regression SUCCESS"
