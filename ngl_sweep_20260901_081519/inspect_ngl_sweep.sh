#!/bin/bash

DIR="$1"

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
