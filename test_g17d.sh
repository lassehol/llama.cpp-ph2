#!/usr/bin/env bash
set -euo pipefail

export PATH=/opt/cmake-3.22.6-linux-x86_64/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
hash -r

ROOT="/workspace/notebooks/llama.cpp-ph2"
cd "$ROOT"

python3 ./G17D_update_regression_readme.py

bash -n ./run_g11_regression.sh

grep -q "ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke" ./run_g11_regression.sh
grep -q "CUDA8 G17D regression SUCCESS" ./run_g11_regression.sh
grep -q "G17 status: real GGML graph-builder Q8_0 MUL_MAT coverage" ggml/src/ggml-cuda8/README.md
grep -q "G17C" ggml/src/ggml-cuda8/README.md
grep -q "host-packed Q8_0" ggml/src/ggml-cuda8/README.md
grep -q "GGML_CUDA8_OP_MUL_MAT_Q8_0_F32_VEC" ggml/src/ggml-cuda8/README.md

echo "G17D metadata checks PASS"

if [ -x ./test_g17c.sh ]; then
    ./test_g17c.sh
elif [ -x ./test_g17a2_v2.sh ]; then
    ./test_g17a2_v2.sh
else
    echo "No focused G17C/G17A2 test script found; running main regression instead"
    chmod +x ./run_g11_regression.sh
    ./run_g11_regression.sh
fi

echo
echo "G17D regression + README status SUCCESS"
