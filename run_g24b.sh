#!/usr/bin/env bash
set -euo pipefail

export PATH=/opt/cmake-3.22.6-linux-x86_64/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
hash -r

ROOT="/workspace/notebooks/llama.cpp-ph2"
cd "$ROOT"

python3 ./G24B_update_regression_readme.py

bash -n ./run_g11_regression.sh
chmod +x ./run_g11_regression.sh

echo
echo "---- G17/G18/G19/G20/G21/G22/G23/G24 entries in main regression ----"
grep -n "G17 real\|G18 real\|G19 real\|G20 real\|G21 real\|G22 real\|G23 real\|G24 real\|q8_0-mmv\|q8_0-add\|q8_0-mul-add\|q8_0-mul-add-softmax\|q8_0-mul-add-softmax-sumrows\|q8_0-residual-add\|q8_0-residual-add-softmax\|q8_0-residual-add-softmax-sumrows\|q8_0-residual-scale-add-softmax-sumrows\|CUDA8 G24B" ./run_g11_regression.sh || true

echo
echo "---- README G24 status markers ----"
grep -n "G24 status\|G24A\|G24B\|scaled residual-softmax-sumrows\|MUL_SCALAR\|REDUCE_SUM_ROWS" ggml/src/ggml-cuda8/README.md || true

echo
echo "---- running main regression ----"
./run_g11_regression.sh
