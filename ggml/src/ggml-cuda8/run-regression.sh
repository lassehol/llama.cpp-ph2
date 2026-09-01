#!/usr/bin/env bash
#
# CUDA8/Fermi backend regression runner.
#
# Run inside the CUDA 8 container (Ubuntu 16.04 base, CUDA 8.0.61, C++11 tools),
# on a machine with the GTX 560 visible - every target below touches the GPU.
#
# Written against the container's old cmake: no -S/-B, no "--build -j".
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

# The root CMakeLists sets CMAKE_RUNTIME_OUTPUT_DIRECTORY to <build>/bin, so
# executables do not land in the target's source-mirror directory. Resolve
# rather than assume - some targets also set RUNTIME_OUTPUT_DIRECTORY directly.
find_binary() {
    local t="$1" cand hit
    for cand in "${BUILD_DIR}/bin/$t" "${BUILD_DIR}/ggml/src/ggml-cuda8/$t"; do
        if [ -x "$cand" ]; then printf '%s' "$cand"; return 0; fi
    done
    hit="$(find "${BUILD_DIR}" -type f -name "$t" -perm -u+x 2>/dev/null | head -1)"
    if [ -n "$hit" ]; then printf '%s' "$hit"; return 0; fi
    return 1
}

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

echo "== cmake $(cmake --version | head -1 | awk '{print $3}')"

# The CUDA 8 container ships an old cmake: no -S/-B (needs 3.13) and no
# --build -j (needs 3.12). Use the portable forms - they work on every version.
if [ ! -f "${BUILD_DIR}/CMakeCache.txt" ]; then
    echo "== configuring ${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}" || exit 1
    ( cd "${BUILD_DIR}" && cmake "${REPO_ROOT}" -DGGML_CUDA8=ON -DGGML_CUDA=OFF ) || exit 1
fi

# -- build --------------------------------------------------------------------

echo "== building ${#TARGETS[@]} target(s) with -j${NPROC}"
for t in "${TARGETS[@]}"; do
    # -j goes after "--" so it reaches make/ninja rather than cmake itself.
    cmake --build "${BUILD_DIR}" --target "$t" -- -j"${NPROC}" > /tmp/build-$t.log 2>&1
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
    bin="$(find_binary "$t")"
    if [ -z "$bin" ]; then
        echo "MISSING  $t   (not found anywhere under ${BUILD_DIR})"
        fail=$((fail + 1)); failed_targets+=("$t")
        continue
    fi

    if "$bin" > "/tmp/run-$t.log" 2>&1; then
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
