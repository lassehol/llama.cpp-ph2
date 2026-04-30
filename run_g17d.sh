#!/usr/bin/env bash
set -euo pipefail

export PATH=/opt/cmake-3.22.6-linux-x86_64/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
hash -r

ROOT="/workspace/notebooks/llama.cpp-ph2"
cd "$ROOT"

python3 ./G17D_update_regression_readme.py

bash -n ./run_g11_regression.sh
chmod +x ./run_g11_regression.sh

echo
echo "---- G17 entries in main regression ----"
grep -n "G17\|graph-builder-q8_0-mmv\|CUDA8 G17D" ./run_g11_regression.sh || true

echo
echo "---- README G17 status markers ----"
grep -n "G17 status\|G17C\|host-packed Q8_0" ggml/src/ggml-cuda8/README.md || true

echo
echo "---- running main regression ----"
./run_g11_regression.sh
