#!/usr/bin/env bash
#
# G39: stage the CUDA 8 artifacts the host build needs.
#
# Run INSIDE the CUDA 8 container, after run-regression.sh (or any build of the
# ggml-cuda8-kernels target) has produced the archive.
#
# ggml-cuda8-host.cmake expects, under GGML_CUDA8_LIB_DIR:
#
#     libggml-cuda8-kernels.a          <- built here
#     cuda8-libs/libcudart_static.a    <- from the container's CUDA 8
#     cuda8-libs/libcudadevrt.a
#     cuda8-libs/libculibos.a
#     cuda8-headers/                   <- the CUDA 8 headers
#
# The CUDA 8 toolkit only exists in the container, so the last two have to be
# copied out into the build directory, which is on the shared volume.
#
#   ./ggml/src/ggml-cuda8/stage-host-artifacts.sh
#
# Env:
#   BUILD_DIR   default <repo>/build-cuda8-kernels
#   CUDA_HOME   default /usr/local/cuda

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build-cuda8-kernels}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

fail() { echo "error: $*" >&2; exit 1; }

# -- sanity -------------------------------------------------------------------

ARCHIVE="${BUILD_DIR}/libggml-cuda8-kernels.a"
[ -f "${ARCHIVE}" ] || ARCHIVE="${BUILD_DIR}/ggml/src/ggml-cuda8/libggml-cuda8-kernels.a"
[ -f "${ARCHIVE}" ] || fail "kernel archive not found under ${BUILD_DIR}
       build it first:  ./ggml/src/ggml-cuda8/run-regression.sh"

[ -d "${CUDA_HOME}" ] || fail "CUDA toolkit not found at ${CUDA_HOME}
       this script must run inside the CUDA 8 container"

CUDA_VER="$(sed -n 's/.*CUDA Version \([0-9.]*\).*/\1/p' "${CUDA_HOME}/version.txt" 2>/dev/null)"
case "${CUDA_VER}" in
    8.*) ;;
    "")  echo "warning: could not determine CUDA version at ${CUDA_HOME}" ;;
    *)   fail "expected CUDA 8, found ${CUDA_VER} at ${CUDA_HOME}.
       Fermi (sm_21) support was removed in CUDA 9 - staging a newer runtime
       here would produce a host binary that cannot talk to the kernels." ;;
esac

echo "== archive   ${ARCHIVE}"
echo "== cuda      ${CUDA_HOME} (${CUDA_VER:-unknown})"
echo "== staging   ${BUILD_DIR}"

# -- static libs --------------------------------------------------------------

mkdir -p "${BUILD_DIR}/cuda8-libs" || exit 1

for lib in libcudart_static.a libcudadevrt.a libculibos.a; do
    src=""
    for d in "${CUDA_HOME}/lib64" "${CUDA_HOME}/lib"; do
        if [ -f "$d/$lib" ]; then src="$d/$lib"; break; fi
    done
    [ -n "$src" ] || fail "$lib not found under ${CUDA_HOME}/lib64 or ${CUDA_HOME}/lib"
    cp -f "$src" "${BUILD_DIR}/cuda8-libs/" || exit 1
    printf '   %-22s %s\n' "$lib" "$(du -h "${BUILD_DIR}/cuda8-libs/$lib" | cut -f1)"
done

# -- headers ------------------------------------------------------------------

[ -d "${CUDA_HOME}/include" ] || fail "no include/ under ${CUDA_HOME}"
rm -rf "${BUILD_DIR}/cuda8-headers"
mkdir -p "${BUILD_DIR}/cuda8-headers" || exit 1
cp -r "${CUDA_HOME}/include/." "${BUILD_DIR}/cuda8-headers/" || exit 1
printf '   %-22s %s\n' "cuda8-headers/" "$(find "${BUILD_DIR}/cuda8-headers" -type f | wc -l) files"

# -- flatten the archive if it is in the nested layout -------------------------

if [ "${ARCHIVE}" != "${BUILD_DIR}/libggml-cuda8-kernels.a" ]; then
    cp -f "${ARCHIVE}" "${BUILD_DIR}/libggml-cuda8-kernels.a" || exit 1
    echo "   copied archive to the flat layout"
fi

# -- next steps ---------------------------------------------------------------

BUILD_SUBDIR="${BUILD_DIR#${REPO_ROOT}/}"

cat <<EOF

Staged into ${BUILD_DIR}
  (that is this container's path - the host mounts the same volume elsewhere,
   e.g. /mnt/shared/caffe/examples/llama.cpp-ph2, so use \$PWD below rather than
   copying the path above)

On the Ubuntu 22.04 host, from the repo root:

  cmake -S . -B build-host \\
        -DGGML_CUDA=OFF \\
        -DGGML_CUDA8_HOST=ON \\
        -DGGML_CUDA8_LIB_DIR="\$PWD/${BUILD_SUBDIR}"
  cmake --build build-host -j\$(nproc) --target llama-cli llama-server

Then, to see the CPU/GPU split:

  GGML_CUDA8_DEBUG_OPS=1 GGML_SCHED_DEBUG=2 \\
    ./build-host/bin/llama-cli -m ./models/Qwen3-0.6B-Q4_K_M.gguf \\
    -ngl 99 -p "hello" -n 8 --cache-type-k f32 --cache-type-v f32 \\
    2> split.log

  --cache-type-k/v f32 is needed until G49: the KV cache defaults to F16 and
  Fermi has no F16 arithmetic.

  GGML_SCHED_DEBUG=2   - where each node ran, and the split boundaries
  GGML_CUDA8_DEBUG_OPS - which ops CUDA8 refused, summarised by frequency at exit

Note: CUDA 8 headers reject GCC > 5.3 in crt/host_config.h. If the host build
stops there, that check is the cause - see the G39 notes in
ggml/src/ggml-cuda8/README.md.
EOF
