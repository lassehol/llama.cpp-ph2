#!/usr/bin/env bash
set -euo pipefail

export PATH=/opt/cmake-3.22.6-linux-x86_64/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
hash -r

ROOT="/workspace/notebooks/llama.cpp-ph2"
cd "$ROOT"

python3 ./G16E_update_regression_readme.py
bash -n ./run_g11_regression.sh
chmod +x ./run_g11_regression.sh

./run_g11_regression.sh
