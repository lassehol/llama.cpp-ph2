#!/usr/bin/env bash
set -euo pipefail

export PATH=/opt/cmake-3.22.6-linux-x86_64/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
hash -r

ROOT="/workspace/notebooks/llama.cpp-ph2"
cd "$ROOT"

python3 ./G16E_update_regression_readme.py
bash -n ./run_g11_regression.sh

grep -q "ggml-cuda8-ggml-backend-graph-builder-add-smoke"      ./run_g11_regression.sh
grep -q "ggml-cuda8-ggml-backend-graph-builder-add-mul-smoke"  ./run_g11_regression.sh
grep -q "ggml-cuda8-ggml-backend-graph-builder-softmax-smoke"  ./run_g11_regression.sh
grep -q "ggml-cuda8-ggml-backend-graph-builder-attnlike-smoke" ./run_g11_regression.sh

grep -q "G16 status: real GGML graph-builder graph_compute coverage" ggml/src/ggml-cuda8/README.md

echo "G16E metadata checks PASS"

if [ -x ./test_g16d_v2.sh ]; then
    ./test_g16d_v2.sh
else
    echo "test_g16d_v2.sh not found/executable; running main regression instead"
    chmod +x ./run_g11_regression.sh
    ./run_g11_regression.sh
fi

echo "G16E regression + README status SUCCESS"
