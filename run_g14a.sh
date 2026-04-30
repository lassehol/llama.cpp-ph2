#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/notebooks/llama.cpp-ph2"
cd "$ROOT"

python3 ./G14A_inspect_backend_api.py

echo
echo "---- G14A report summary ----"
grep -n "ggml_backend_i\|Graph/compute\|No graph/compute\|G14A interpretation" ggml/src/ggml-cuda8/G14A_backend_api_report.md || true

echo
echo "G14A inspection SUCCESS"
