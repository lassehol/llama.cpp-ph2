#!/usr/bin/env bash
set -euo pipefail

export PATH=/opt/cmake-3.22.6-linux-x86_64/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
hash -r

ROOT="/workspace/notebooks/llama.cpp-ph2"
cd "$ROOT"

python3 ./G17B_update_regression_readme.py

bash -n ./run_g11_regression.sh

grep -q "ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke" ./run_g11_regression.sh
grep -q "G17 status: real GGML graph-builder Q8_0 MUL_MAT coverage" ggml/src/ggml-cuda8/README.md
grep -q "GGML_OP_MUL_MAT" ggml/src/ggml-cuda8/README.md
grep -q "GGML_CUDA8_OP_MUL_MAT_Q8_0_F32_VEC" ggml/src/ggml-cuda8/README.md

echo "G17B metadata checks PASS"

if [ -x ./test_g17a2_v2.sh ]; then
    ./test_g17a2_v2.sh
elif [ -x ./test_g17a.sh ]; then
    ./test_g17a.sh
else
    echo "No focused G17A test script found; running main regression instead"
    chmod +x ./run_g11_regression.sh
    ./run_g11_regression.sh
fi

echo
echo "G17B regression + README status SUCCESS"
