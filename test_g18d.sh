#!/usr/bin/env bash
set -euo pipefail

export PATH=/opt/cmake-3.22.6-linux-x86_64/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
hash -r

ROOT="/workspace/notebooks/llama.cpp-ph2"
cd "$ROOT"

python3 ./G18D_update_regression_readme.py

bash -n ./run_g11_regression.sh

grep -q "ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke" ./run_g11_regression.sh
grep -q "ggml-cuda8-ggml-backend-graph-builder-q8_0-add-smoke" ./run_g11_regression.sh
grep -q "ggml-cuda8-ggml-backend-graph-builder-q8_0-mul-add-smoke" ./run_g11_regression.sh
grep -q "CUDA8 G18D regression SUCCESS" ./run_g11_regression.sh

grep -q "G18 status: real GGML graph-builder quantized pipeline coverage" ggml/src/ggml-cuda8/README.md
grep -q "G18C" ggml/src/ggml-cuda8/README.md
grep -q "G18D" ggml/src/ggml-cuda8/README.md
grep -q "GGML_OP_MUL_MAT" ggml/src/ggml-cuda8/README.md
grep -q "GGML_OP_MUL" ggml/src/ggml-cuda8/README.md
grep -q "GGML_OP_ADD" ggml/src/ggml-cuda8/README.md
grep -q "GGML_CUDA8_OP_MUL_MAT_Q8_0_F32_VEC" ggml/src/ggml-cuda8/README.md
grep -q "GGML_CUDA8_OP_MUL_SCALAR_F32" ggml/src/ggml-cuda8/README.md
grep -q "GGML_CUDA8_OP_ADD_F32" ggml/src/ggml-cuda8/README.md

echo "G18D metadata checks PASS"

if [ -x ./test_g18c.sh ]; then
    ./test_g18c.sh
else
    echo "test_g18c.sh not found; running main regression instead"
    chmod +x ./run_g11_regression.sh
    ./run_g11_regression.sh
fi

echo
echo "G18D regression + README status SUCCESS"
