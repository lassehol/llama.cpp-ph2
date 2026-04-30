#!/usr/bin/env bash
set -euo pipefail

export PATH=/opt/cmake-3.22.6-linux-x86_64/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
hash -r

ROOT="/workspace/notebooks/llama.cpp-ph2"
cd "$ROOT"

python3 ./G23C_precommit_sanity.py
bash -n ./run_g11_regression.sh

if [ -x ./test_g23a.sh ]; then
    ./test_g23a.sh
else
    echo "test_g23a.sh missing; running G23A target directly"
    cd "$ROOT/build-cuda8-parent"
    cmake .. -DGGML_CUDA8=ON -DGGML_CUDA=OFF -DBUILD_SHARED_LIBS=OFF
    make -j1 ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-sumrows-smoke
    ./bin/ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-sumrows-smoke
fi

echo
echo "G23C pre-commit sanity + focused G23A regression SUCCESS"
