#!/usr/bin/env bash
set -euo pipefail

export PATH=/opt/cmake-3.22.6-linux-x86_64/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
hash -r

ROOT="/workspace/notebooks/llama.cpp-ph2"
cd "$ROOT"

echo "== G23D: pre-stage validation =="
./test_g23c.sh

echo
echo "== G23D: reset staged area =="
git reset

echo
echo "== G23D: stage core root files =="
git add \
  run_g11_regression.sh \
  ggml/src/CMakeLists.txt \
  ggml/src/ggml-backend-meta.cpp

echo
echo "== G23D: stage clean ggml-cuda8 backend tree =="
find ggml/src/ggml-cuda8 -type f \
  ! -name '*backup*' \
  ! -name '*.zip' \
  ! -name '*:Zone.Identifier' \
  ! -name '*.bad*' \
  ! -name '*.badtail-*' \
  -print0 | xargs -0 git add --

echo
echo "== G23D: stage G17-G23 checkpoint writers/scripts =="

git add \
  G17C_patch_packed_q8_ref.py \
  G17D_update_regression_readme.py \
  G18A_writer.py \
  G18B_update_regression_readme.py \
  G18C_writer.py \
  G18D_update_regression_readme.py \
  G19A_writer.py \
  G19B_update_regression_readme.py \
  G20A_writer.py \
  G20B_update_regression_readme.py \
  G21A_writer.py \
  G21B_update_regression_readme.py \
  G22A_writer.py \
  G22B_update_regression_readme.py \
  G23A_writer.py \
  G23B_update_regression_readme.py \
  G23C_precommit_sanity.py \
  run_g17c.sh test_g17c.sh \
  run_g17d.sh test_g17d.sh \
  run_g18a.sh test_g18a.sh \
  run_g18b.sh test_g18b.sh \
  run_g18c.sh test_g18c.sh \
  run_g18d.sh test_g18d.sh \
  run_g19a.sh test_g19a.sh \
  run_g19b.sh test_g19b.sh \
  run_g20a.sh test_g20a.sh \
  run_g20b.sh test_g20b.sh \
  run_g21a.sh test_g21a.sh \
  run_g21b.sh test_g21b.sh \
  run_g22a.sh test_g22a.sh \
  run_g22b.sh test_g22b.sh \
  run_g23a.sh test_g23a.sh \
  run_g23b.sh test_g23b.sh \
  run_g23c.sh test_g23c.sh

echo
echo "== G23D: reject unwanted staged artifacts =="
BAD="$(git diff --cached --name-only | grep -E '(:Zone\.Identifier$|\.zip$|backup-|\.bad|\.badtail-|^patch_g|nodecount-|remaining-regex-backup|tensor-regex-backup|source-token-fix-backup)' || true)"

if [ -n "$BAD" ]; then
    echo "Unexpected staged artifacts:"
    echo "$BAD"
    exit 1
fi

echo
echo "== G23D: staged diff summary =="
git diff --cached --stat

echo
echo "== G23D: staged files =="
git diff --cached --name-only | sort

echo
echo "G23D staged checkpoint SUCCESS"
