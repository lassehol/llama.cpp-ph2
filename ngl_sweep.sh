#!/bin/bash

set -euo pipefail

MODEL="models/Qwen3-0.6B-Q4_K_M.gguf"
PROMPT="Say exactly: hello"

OUTDIR="ngl_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

NGLS=(
    0
    2
    6
    10
    14
    18
)

for NGL in "${NGLS[@]}"; do
    echo "=================================================="
    echo "Running -ngl $NGL"
    echo "=================================================="

    LOG="$OUTDIR/ngl_${NGL}.log"

    GGML_CUDA8_DEBUG_OPS=1 \
    ./build-host/bin/llama-cli \
        -m "$MODEL" \
        -ngl "$NGL" \
        -fa off \
        --cache-type-k f32 \
        --cache-type-v f32 \
        -n 32 \
        -p "$PROMPT"  | sed -u -E \
        -e 's/ggml-cuda8\/backend:.*$//' \
        -e '/^[[:space:]]*ggml-cuda8:/d' \
        -e '/^[[:space:]]*$/d' \
        >"$LOG" 2>&1 || true

    echo "Saved $LOG"
done

echo
echo "=== SUMMARY ==="

for NGL in "${NGLS[@]}"; do
    LOG="$OUTDIR/ngl_${NGL}.log"

    echo
    echo "---- ngl=$NGL ----"

    grep -A5 -B5 "Generation:" "$LOG" || true

    grep "ops refused by supports_op" -A5 "$LOG" || true
done

echo
echo "Logs written to:"
echo "  $OUTDIR/"
