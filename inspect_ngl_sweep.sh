#!/bin/bash

DIR="ngl_sweep_20260901_081519"

for f in "$DIR"/ngl_*.log; do
    echo
    echo "================================================="
    echo "$f"
    echo "================================================="

    awk '
        /^> Say exactly: hello/ {show=1; next}
        /^\[ Prompt:/ {show=0}
        show {print}
    ' "$f"
done
