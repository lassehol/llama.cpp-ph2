#!/usr/bin/env bash
#
# fix_cuda8_synchronize.sh
#
# Patches ggml-cuda8-ggml-backend.cpp to add a real cudaDeviceSynchronize()
# call in cuda8_backend_synchronize(), which is currently a no-op stub.
#
# Root cause: the synchronize backend-interface hook never called any CUDA
# synchronization primitive, so consecutive GPU-executed transformer layers
# could race — layer N+1 reading layer N's output buffer before the GPU had
# actually finished writing it. Confirmed via bisection (2026-08-14):
# 1 GPU layer = correct output, 2+ GPU layers = degenerate repeating-token
# output, CPU-only always correct.
#
# This script:
#   1. Locates the target file
#   2. Backs it up with a timestamped name matching the project's existing
#      backup convention (.g<tag>-backup-<epoch>)
#   3. Replaces the no-op synchronize function with a real
#      cudaDeviceSynchronize() call + error checking
#   4. Verifies the patch applied exactly once
#   5. Prints a diff for review — does NOT rebuild automatically
#
# Usage:
#   ./fix_cuda8_synchronize.sh [path-to-ggml-cuda8-ggml-backend.cpp]
#
# If no path is given, defaults to the known location on Athlon.

set -euo pipefail

TARGET="${1:-/mnt/shared/caffe/examples/llama.cpp-ph2/ggml/src/ggml-cuda8/ggml-cuda8-ggml-backend.cpp}"
TAG="g17a3-sync-fix"
TIMESTAMP="$(date +%s)"
BACKUP="${TARGET}.${TAG}-backup-${TIMESTAMP}"

if [[ ! -f "$TARGET" ]]; then
    echo "ERROR: target file not found: $TARGET" >&2
    exit 1
fi

echo "== Target file: $TARGET"
echo "== Backup will be written to: $BACKUP"

# --- Step 1: confirm the exact no-op function is present, unmodified ---
# This is the known-bad function body we expect to find, byte for byte.
# If it doesn't match exactly, the file may have already been patched or
# modified in an incompatible way — abort rather than guess.

read -r -d '' OLD_BLOCK <<'EOF' || true
static void cuda8_backend_synchronize(ggml_backend_t backend) {
    (void) backend;
    std::printf("ggml-cuda8/backend: synchronize PASS\n");
}
EOF

if ! grep -qF "$OLD_BLOCK" "$TARGET"; then
    echo "ERROR: expected no-op synchronize() block not found verbatim in $TARGET" >&2
    echo "The file may already be patched, or its formatting differs from what" >&2
    echo "this script expects. Aborting without making changes." >&2
    echo "" >&2
    echo "Expected block:" >&2
    echo "$OLD_BLOCK" >&2
    exit 1
fi

MATCH_COUNT=$(grep -zoF "$OLD_BLOCK" "$TARGET" | grep -zoc "$OLD_BLOCK" || true)
echo "== Found the target function block in the file (verbatim match confirmed)."

# --- Step 1b: confirm <cuda_runtime.h> is not already included ---
# Confirmed 2026-08-14: neither this file nor its three project-local headers
# (ggml-cuda8-ggml-backend.h, ggml-cuda8-ggml-buffer.h, ggml-cuda8-dispatch.h)
# include cuda_runtime.h anywhere. Without it, cudaError_t/cudaDeviceSynchronize/
# cudaGetErrorString are undeclared and the patched code will fail to compile
# (a normal host-compiler error, unrelated to nvcc/CUDA8 kernel compilation).
NEEDS_INCLUDE=1
if grep -qE '#include\s*[<"]cuda_runtime' "$TARGET"; then
    echo "== <cuda_runtime.h> already included in this file — skipping include insertion."
    NEEDS_INCLUDE=0
else
    echo "== <cuda_runtime.h> not found in this file — will insert it after the existing"
    echo "   <cstring> include (defensive: not currently pulled in transitively either,"
    echo "   per direct check of the three project-local headers)."
fi

# --- Step 2: back up before touching anything ---
cp -p "$TARGET" "$BACKUP"
echo "== Backup written: $BACKUP"

# --- Step 3: apply the patch ---
# We replace the whole function body with a real synchronization call plus
# error checking, matching the existing code's style (std::printf,
# (void) backend already present, similar error-reporting pattern used
# elsewhere in this file via std::fprintf(stderr, ...)). We also insert
# #include <cuda_runtime.h> if it wasn't already present (Step 1b).

python3 - "$TARGET" "$NEEDS_INCLUDE" <<'PYEOF'
import sys

target = sys.argv[1]
needs_include = sys.argv[2] == "1"

old_block = '''static void cuda8_backend_synchronize(ggml_backend_t backend) {
    (void) backend;
    std::printf("ggml-cuda8/backend: synchronize PASS\\n");
}'''

new_block = '''static void cuda8_backend_synchronize(ggml_backend_t backend) {
    (void) backend;
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "ggml-cuda8/backend: synchronize FAILED: %s\\n", cudaGetErrorString(err));
    } else {
        std::printf("ggml-cuda8/backend: synchronize PASS\\n");
    }
}'''

old_includes = '''#include <cstdio>
#include <cstdlib>
#include <cstring>'''

new_includes = '''#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>'''

with open(target, 'r') as f:
    content = f.read()

count = content.count(old_block)
if count != 1:
    print(f"ERROR: expected exactly 1 occurrence of the synchronize() block, found {count}", file=sys.stderr)
    sys.exit(1)

content = content.replace(old_block, new_block, 1)

if needs_include:
    inc_count = content.count(old_includes)
    if inc_count != 1:
        print(f"ERROR: expected exactly 1 occurrence of the include block, found {inc_count}", file=sys.stderr)
        print("The synchronize() function body WAS patched, but the include", file=sys.stderr)
        print("insertion point was not found as expected. You will need to add", file=sys.stderr)
        print("'#include <cuda_runtime.h>' near the top of the file by hand", file=sys.stderr)
        print("before rebuilding.", file=sys.stderr)
        with open(target, 'w') as f:
            f.write(content)
        sys.exit(2)
    content = content.replace(old_includes, new_includes, 1)
    print("== <cuda_runtime.h> include inserted after existing <cstring> include.")

with open(target, 'w') as f:
    f.write(content)

print("== Patch applied via Python (exact single-occurrence replace).")
PYEOF
PATCH_STATUS=$?
if [[ $PATCH_STATUS -eq 2 ]]; then
    echo "== WARNING: synchronize() was patched but include insertion needs manual follow-up (see message above)."
elif [[ $PATCH_STATUS -ne 0 ]]; then
    echo "ERROR: patch step failed (exit $PATCH_STATUS). Restoring from backup." >&2
    cp -p "$BACKUP" "$TARGET"
    exit 1
fi

# --- Step 4: verify ---
VERIFY_OK=1
if grep -qF "cudaDeviceSynchronize()" "$TARGET"; then
    echo "== Verification PASSED: cudaDeviceSynchronize() now present in $TARGET"
else
    echo "ERROR: verification FAILED — cudaDeviceSynchronize() not found after patching." >&2
    VERIFY_OK=0
fi

if grep -qE '#include\s*[<"]cuda_runtime' "$TARGET"; then
    echo "== Verification PASSED: <cuda_runtime.h> include present in $TARGET"
else
    echo "ERROR: verification FAILED — <cuda_runtime.h> include not found after patching." >&2
    echo "The build will fail with 'cudaDeviceSynchronize was not declared' or similar" >&2
    echo "until this include is added manually." >&2
    VERIFY_OK=0
fi

if [[ "$VERIFY_OK" -eq 0 ]]; then
    echo "Restoring from backup due to failed verification." >&2
    cp -p "$BACKUP" "$TARGET"
    exit 1
fi

# --- Step 5: also add a per-node sync safety net inside graph_compute ---
# The top-level synchronize() hook may only be called by llama.cpp's scheduler
# at coarse granularity (e.g. once per llama_decode() call, not once per
# layer). If the multi-layer bug persists after rebuilding with the fix
# above, the more targeted fix is a sync call after each dispatched node
# inside cuda8_backend_graph_compute's loop. This script does NOT apply that
# automatically since it requires locating the exact dispatch loop structure
# per node — flagging it here so you know it's the next lever if step 3
# alone doesn't fully resolve the bug.
echo ""
echo "== NOTE: if the multi-layer bug persists after rebuilding, the next"
echo "   lever is adding cudaDeviceSynchronize() after each node dispatch"
echo "   inside cuda8_backend_graph_compute() (around line 357-367), not"
echo "   just at the top-level synchronize() hook. Not applied by this"
echo "   script — review and apply manually if needed."

# --- Step 6: show the diff for review ---
echo ""
echo "== Diff (backup vs. patched file):"
diff -u "$BACKUP" "$TARGET" || true

echo ""
echo "== Done. File patched, backup saved at:"
echo "     $BACKUP"
echo ""
echo "== Note: this is a plain .cpp file compiled by the host C++ compiler"
echo "   (g++), not a .cu file — no nvcc invocation, no Docker step, and no"
echo "   CUDA-8-toolkit involvement needed for this rebuild. It links against"
echo "   the already-built libcudart and the pre-built CUDA8 kernel library"
echo "   exactly as before; only the dispatch/orchestration code changed."
echo ""
echo "Next steps (not run automatically):"
echo "  cd /mnt/shared/caffe/examples/llama.cpp-ph2/build-host-cuda8"
echo "  cmake --build . --target llama-cli -j\$(nproc)"
echo ""
echo "Then re-run the -ngl bisection sweep (1/2/3/4/8/12) to confirm the fix."
