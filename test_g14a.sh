#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/notebooks/llama.cpp-ph2"
cd "$ROOT"

python3 -m py_compile ./G14A_inspect_backend_api.py
bash -n ./run_g14a.sh
./run_g14a.sh

test -s ggml/src/ggml-cuda8/G14A_backend_api_report.md
test -s ggml/src/ggml-cuda8/G14A_backend_api_snapshot.txt

grep -q "G14A Backend API Inspection Report" ggml/src/ggml-cuda8/G14A_backend_api_report.md

echo
echo "G14A regression SUCCESS"
