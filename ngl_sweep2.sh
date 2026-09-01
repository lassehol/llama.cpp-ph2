#!/usr/bin/env bash

set -u

MODEL="models/Qwen3-0.6B-Q4_K_M.gguf"
OUTDIR="ngl_boundary_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

for NGL in 2 3 4 5 6; do
    for KV_MODE in gpu host; do
        LOG="$OUTDIR/ngl_${NGL}_${KV_MODE}_kv.log"

        EXTRA_ARGS=()
        if [ "$KV_MODE" = host ]; then
            EXTRA_ARGS+=("-nkvo")
        fi

        echo "=== ngl=$NGL kv=$KV_MODE ==="

        ./build-host/bin/llama-cli \
          -m "$MODEL" \
          -ngl "$NGL" \
          "${EXTRA_ARGS[@]}" \
          -fa off \
          --cache-type-k f32 \
          --cache-type-v f32 \
          --seed 1234 \
          --temp 0 \
          -n 32 \
          -p "Say exactly: hello" \
          >"$LOG" 2>&1 || true

        CLEAN="$OUTDIR/ngl_${NGL}_${KV_MODE}_kv.clean.txt"

        sed -E \
          -e 's/ggml-cuda8\/backend:.*$//' \
          -e '/^[[:space:]]*ggml-cuda8:/d' \
          -e '/^[[:space:]]*$/d' \
          "$LOG" >"$CLEAN"

        echo "Saved:"
        echo "  $LOG"
        echo "  $CLEAN"
    done
done

echo
echo "Results: $OUTDIR"
