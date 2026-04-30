#!/usr/bin/env bash
set -euo pipefail

export PATH=/opt/cmake-3.22.6-linux-x86_64/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
hash -r

ROOT="/workspace/notebooks/llama.cpp-ph2"
cd "$ROOT"

python3 ./G23C_precommit_sanity.py
bash -n ./run_g11_regression.sh

echo
echo "---- G23C git status snapshot ----"
git status --short || true

echo
echo "---- G23C backup files summary ----"
find . -type f \( -name '*.g23b-backup-*' -o -name '*.g23b-repair-backup-*' -o -name '*.g23a-backup-*' \) | sort || true

echo
echo "G23C pre-commit sanity SUCCESS"
