#!/usr/bin/env bash
#
# CUDA8/Fermi backend regression runner.
#
# Run inside the CUDA 8 container (Ubuntu 16.04 base, CUDA 8.0.61, C++11 tools),
# on a machine with the GTX 560 visible - every target below touches the GPU.
#
#   ./ggml/src/ggml-cuda8/run-regression.sh            # default set
#   ./ggml/src/ggml-cuda8/run-regression.sh --all      # every smoke target
#   ./ggml/src/ggml-cuda8/run-regression.sh <target>.. # explicit targets
#
# Env:
#   BUILD_DIR   build directory (default: <repo>/build-cuda8-kernels)
#   CMAKE_BIN   cmake to use    (default: cmake on PATH)
#   NPROC       parallel jobs   (default: nproc)
#
# Why this configures ggml/src/ggml-cuda8 and not the repo root:
#
#   The root CMakeLists.txt (and ggml/CMakeLists.txt) require cmake >= 3.14.
#   The CUDA 8 container ships cmake 3.5.1, so it cannot configure them - it can
#   only drive "cmake --build" on a tree someone else generated. But
#   ggml/src/ggml-cuda8/CMakeLists.txt is a standalone project needing only
#   cmake 3.5, and it pulls in ../ggml.c, ../ggml-quants.c and
#   ../ggml-threading.cpp by relative path. So the kernels and every smoke can
#   be configured and built here without the parent project existing at all.
#
#   An older build-cuda8-parent tree may still be around, configured against the
#   repo root back when it accepted cmake 3.5. That tree can no longer be
#   regenerated in this container, so adding a target to CMakeLists.txt breaks it
#   with "No rule to make target". Point BUILD_DIR at it only for builds that do
#   not add targets.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

SOURCE_DIR="${SCRIPT_DIR}"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build-cuda8-kernels}"
CMAKE_BIN="${CMAKE_BIN:-$(command -v cmake)}"
NPROC="${NPROC:-$(nproc 2>/dev/null || echo 4)}"

# Executables land in <build>/bin for targets that set RUNTIME_OUTPUT_DIRECTORY
# and in <build> for those that do not, so resolve rather than assume.
find_binary() {
    local t="$1" cand hit
    for cand in "${BUILD_DIR}/bin/$t" "${BUILD_DIR}/$t"; do
        if [ -x "$cand" ]; then printf '%s' "$cand"; return 0; fi
    done
    hit="$(find "${BUILD_DIR}" -type f -name "$t" -perm -u+x 2>/dev/null | head -1)"
    if [ -n "$hit" ]; then printf '%s' "$hit"; return 0; fi
    return 1
}

# -- target sets --------------------------------------------------------------

# Targets exercising the most recent changes.
#   G37: the SOFT_MAX guard and the two corrected synthetic fixtures.
#   G38: grid clamps across the older kernels, plus the oversized-tensor smoke
#        that drives them past the 65535-block limit.
PRIMARY=(
    ggml-cuda8-oversized-smoke
    ggml-cuda8-ggml-backend-supports-op-smoke
    ggml-cuda8-ggml-backend-graph-compute-softmax-smoke
    ggml-cuda8-ggml-backend-graph-compute-attnlike-smoke
	ggml-cuda8-poison-smoke
)

# Standalone kernel smokes for every kernel G38 rewrote. These use small
# shapes, so they check that the stride loops did not break the ordinary path.
KERNEL_REGRESSION=(
    ggml-cuda8-add-smoke
    ggml-cuda8-mul-smoke
    ggml-cuda8-scalar-smoke
    ggml-cuda8-reduce-smoke
    ggml-cuda8-reduce-max-smoke
    ggml-cuda8-softmax-smoke
    ggml-cuda8-rms-norm-smoke
    ggml-cuda8-rope-smoke
    ggml-cuda8-diagmask-smoke
    ggml-cuda8-getrows-smoke
    ggml-cuda8-swiglu-smoke
    ggml-cuda8-set-rows-smoke
	ggml-cuda8-softmax-ext-smoke
	ggml-cuda8-mulmat-f32-smoke
	ggml-cuda8-cont-smoke
	ggml-cuda8-set-rows-f16-smoke
	ggml-cuda8-mulmat-f16-smoke
)

# End-to-end graphs. Unaffected in principle by both G37 and G38.
GRAPH_REGRESSION=(
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
        grep -o 'cuda_add_executable([a-zA-Z0-9_-]*' "${SOURCE_DIR}/CMakeLists.txt" |
        sed 's/cuda_add_executable(//' | sort -u
    )
elif [ "$#" -gt 0 ]; then
    TARGETS=("$@")
else
    TARGETS=("${PRIMARY[@]}" "${KERNEL_REGRESSION[@]}" "${GRAPH_REGRESSION[@]}")
fi

# -- configure ----------------------------------------------------------------

echo "== cmake $("${CMAKE_BIN}" --version | head -1 | awk '{print $3}')  (${CMAKE_BIN})"
echo "== source ${SOURCE_DIR}"
echo "== build  ${BUILD_DIR}"

if [ ! -f "${BUILD_DIR}/CMakeCache.txt" ]; then
    echo "== configuring"
    mkdir -p "${BUILD_DIR}" || exit 1
    # No -S/-B: cmake 3.5 does not have them.
    ( cd "${BUILD_DIR}" && "${CMAKE_BIN}" "${SOURCE_DIR}" ) || exit 1
elif [ "${SOURCE_DIR}/CMakeLists.txt" -nt "${BUILD_DIR}/CMakeCache.txt" ]; then
    # make only re-runs cmake as a prerequisite of targets it already knows
    # about, so a target added since the last configure would otherwise fail
    # with "No rule to make target" and never trigger a regenerate.
    echo "== regenerating (CMakeLists.txt is newer than the cache)"
    ( cd "${BUILD_DIR}" && "${CMAKE_BIN}" . > /tmp/cmake-regen.log 2>&1 ) || {
        echo "cmake regeneration FAILED (see /tmp/cmake-regen.log)"
        tail -20 /tmp/cmake-regen.log
        exit 1
    }
fi

# -- build --------------------------------------------------------------------

echo "== building ${#TARGETS[@]} target(s) with -j${NPROC}"
for t in "${TARGETS[@]}"; do
    # -j after "--" so it reaches make, not cmake (cmake 3.5 has no --build -j).
    "${CMAKE_BIN}" --build "${BUILD_DIR}" --target "$t" -- -j"${NPROC}" \
        > "/tmp/build-$t.log" 2>&1
    if [ $? -ne 0 ]; then
        echo "BUILD FAILED: $t   (see /tmp/build-$t.log)"
        tail -30 "/tmp/build-$t.log"
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
