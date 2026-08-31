#!/usr/bin/env bash
#
# CUDA8/Fermi backend regression runner.
#
# Run inside the CUDA 8 container (Ubuntu 16.04 base, CUDA 8.0.61, C++11 tools),
# on a machine with the GTX 560 visible - every target below touches the GPU.
#
#   ./ggml/src/ggml-cuda8/run-regression.sh            # G37 set (default)
#   ./ggml/src/ggml-cuda8/run-regression.sh --all      # every smoke target
#   ./ggml/src/ggml-cuda8/run-regression.sh <target>.. # explicit targets
#
# Env:
#   BUILD_DIR   build directory (default: <repo>/build-cuda8-parent)
#   NPROC       parallel build jobs (default: nproc)

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build-cuda8-parent}"
NPROC="${NPROC:-$(nproc 2>/dev/null || echo 4)}"
BIN_DIR="${BUILD_DIR}/ggml/src/ggml-cuda8"

# -- target sets --------------------------------------------------------------

# Changed by G37: the softmax guard and the two corrected synthetic fixtures.
G37_PRIMARY=(
    ggml-cuda8-ggml-backend-supports-op-smoke
    ggml-cuda8-ggml-backend-graph-compute-softmax-smoke
    ggml-cuda8-ggml-backend-graph-compute-attnlike-smoke
)

# Must be unaffected by G37: these build real graphs with ggml_soft_max(),
# which emits src[1]=NULL and op_params={1.0f, 0.0f} - still supported.
G37_REGRESSION=(
    ggml-cuda8-softmax-smoke
    ggml-cuda8-dispatch-all-smoke
    ggml-cuda8-ggml-backend-graph-builder-softmax-smoke
    ggml-cuda8-ggml-backend-graph-builder-attnlike-smoke
    ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-smoke
    ggml-cuda8-ggml-backend-graph-builder-transformer-block-smoke
    ggml-cuda8-ggml-backend-graph-builder-attention-smoke
    ggml-cuda8-ggml-backend-graph-builder-e2e-smoke
)

if [ "$#" -gt 0 ] && [ "$1" = "--all" ]; then
    mapfile -t TARGETS < <(
        grep -o 'cuda_add_executable([a-zA-Z0-9_-]*' "${SCRIPT_DIR}/CMakeLists.txt" |
        sed 's/cuda_add_executable(//' | sort -u
    )
elif [ "$#" -gt 0 ]; then
    TARGETS=("$@")
else
    TARGETS=("${G37_PRIMARY[@]}" "${G37_REGRESSION[@]}")
fi

# -- configure ----------------------------------------------------------------

if [ ! -f "${BUILD_DIR}/CMakeCache.txt" ]; then
    echo "== configuring ${BUILD_DIR}"
    cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" \
        -DGGML_CUDA8=ON -DGGML_CUDA=OFF || exit 1
fi

# -- build --------------------------------------------------------------------

echo "== building ${#TARGETS[@]} target(s) with -j${NPROC}"
for t in "${TARGETS[@]}"; do
    cmake --build "${BUILD_DIR}" --target "$t" -j "${NPROC}" > /tmp/build-$t.log 2>&1
    if [ $? -ne 0 ]; then
        echo "BUILD FAILED: $t   (see /tmp/build-$t.log)"
        tail -30 /tmp/build-$t.log
        exit 1
    fi
done
echo "   build ok"

# -- run ----------------------------------------------------------------------

echo
pass=0
fail=0
failed_targets=()

for t in "${TARGETS[@]}"; do
    if [ ! -x "${BIN_DIR}/$t" ]; then
        echo "MISSING  $t   (not at ${BIN_DIR})"
        fail=$((fail + 1)); failed_targets+=("$t")
        continue
    fi

    if "${BIN_DIR}/$t" > "/tmp/run-$t.log" 2>&1; then
        printf 'PASS     %s\n' "$t"
        pass=$((pass + 1))
    else
        printf 'FAIL     %s   (see /tmp/run-%s.log)\n' "$t" "$t"
        fail=$((fail + 1)); failed_targets+=("$t")
    fi
done

# -- summary ------------------------------------------------------------------

echo
echo "== ${pass} passed, ${fail} failed"
if [ "${fail}" -ne 0 ]; then
    echo "failed:"
    for t in "${failed_targets[@]}"; do echo "  $t"; done
    exit 1
fi
