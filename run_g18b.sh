#!/usr/bin/env bash
set -euo pipefail

export PATH=/opt/cmake-3.22.6-linux-x86_64/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
hash -r

ROOT="/workspace/notebooks/llama.cpp-ph2"
cd "$ROOT"

python3 ./G18B_update_regression_readme.py

bash -n ./run_g11_regression.sh
chmod +x ./run_g11_regression.sh

echo
echo "---- G17/G18 entries in main regression ----"
grep -n "G17 real\|G18 real\|q8_0-mmv\|q8_0-add\|CUDA8 G18B" ./run_g11_regression.sh || true

echo
echo "---- README G18 status markers ----"
grep -n "G18 status\|G18A\|G18B\|MUL_MAT -> ADD" ggml/src/ggml-cuda8/README.md || true

echo
echo "---- running main regression ----"
./run_g11_regression.sh
