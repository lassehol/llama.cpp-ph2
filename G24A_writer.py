#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G24A_writer.py  -  RMS_NORM kernel + dispatch + smoke test
   Target: GTX 560 / Fermi / CUDA 8 / compute 2.1
   Python 3.5+
"""
import os, sys, glob, re

ROOT = "/workspace/notebooks/llama.cpp-ph2"
BASE = os.path.join(ROOT, "ggml/src/ggml-cuda8")
ENC = {"encoding": "utf-8"}

def read_file(path):
    with open(path, **ENC) as fh:
        return fh.read()

def write_file(path, content):
    with open(path, "w", **ENC) as fh:
        fh.write(content)

def find_cu(pattern_in_content):
    """Return first .cu in BASE whose content contains pattern_in_content."""
    for f in sorted(glob.glob(os.path.join(BASE, "*.cu"))):
        if pattern_in_content in read_file(f):
            return f
    return None

def find_h(pattern_in_content):
    """Return first .h in BASE whose content contains pattern_in_content."""
    for f in sorted(glob.glob(os.path.join(BASE, "*.h"))):
        if pattern_in_content in read_file(f):
            return f
    return None

# ============================================================================
# 1.  New file: ggml-cuda8-rms-norm.cu  (Fermi-safe, shared-mem reduction)
# ============================================================================

RMS_NORM_CU = r"""// ggml-cuda8-rms-norm.cu  -  G24A: RMS_NORM kernel (Fermi-safe)
#include <cuda_runtime.h>

__global__ void kernel_rms_norm_f32(
        const float * __restrict__ x,
        float       * __restrict__ y,
        const int ncols,
        const float eps) {

    const int row = blockIdx.x;
    const float * x_row = x + (size_t)row * ncols;
    float       * y_row = y + (size_t)row * ncols;

    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;

    // partial sum-of-squares
    float sum_sq = 0.0f;
    for (int col = tid; col < ncols; col += bs) {
        float v = x_row[col];
        sum_sq += v * v;
    }
    sdata[tid] = sum_sq;
    __syncthreads();

    // tree reduction in shared memory (no warp shuffle on Fermi)
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // normalize:  y_i = x_i * rsqrt( mean(x^2) + eps )
    const float scale = rsqrtf(sdata[0] / (float)ncols + eps);
    for (int col = tid; col < ncols; col += bs) {
        y_row[col] = x_row[col] * scale;
    }
}

void ggml_cuda8_rms_norm_f32(
        const float * x, float * y,
        int nrows, int ncols, float eps,
        cudaStream_t stream) {
    const int block_size = 256;
    const size_t smem = block_size * sizeof(float);
    kernel_rms_norm_f32<<<nrows, block_size, smem, stream>>>(x, y, ncols, eps);
}
"""

rms_norm_path = os.path.join(BASE, "ggml-cuda8-rms-norm.cu")
if not os.path.exists(rms_norm_path):
    write_file(rms_norm_path, RMS_NORM_CU)
    print("[G24A] Created %s" % rms_norm_path)
else:
    print("[G24A] %s already exists, skip" % rms_norm_path)

# ============================================================================
# 2.  Patch header  -  add RMS_NORM declaration
# ============================================================================

RMS_NORM_DECL = """
// G24A: RMS_NORM
void ggml_cuda8_rms_norm_f32(
        const float * x, float * y,
        int nrows, int ncols, float eps,
        cudaStream_t stream);
"""

# Try to find the main header that declares the other kernel wrappers
hdr_file = find_h("ggml_cuda8_softmax") or find_h("ggml_cuda8_add") or find_h("ggml_cuda8")
if hdr_file is None:
    # fallback: list all .h files for diagnostics
    hdr_files = glob.glob(os.path.join(BASE, "*.h"))
    if hdr_files:
        # just pick the first one
        hdr_file = hdr_files[0]
        print("[G24A] WARNING: guessing header = %s" % hdr_file)
    else:
        print("[G24A] WARNING: no .h files found in %s" % BASE)

if hdr_file:
    hsrc = read_file(hdr_file)
    if "ggml_cuda8_rms_norm_f32" not in hsrc:
        if "#endif" in hsrc:
            last_endif = hsrc.rfind("#endif")
            hsrc = hsrc[:last_endif] + RMS_NORM_DECL + "\n" + hsrc[last_endif:]
        else:
            hsrc = hsrc.rstrip() + "\n" + RMS_NORM_DECL + "\n"
        write_file(hdr_file, hsrc)
        print("[G24A] Patched %s  +RMS_NORM decl" % hdr_file)
    else:
        print("[G24A] %s already has RMS_NORM decl, skip" % hdr_file)

# ============================================================================
# 3.  Add GGML_OP_RMS_NORM to graph_compute dispatch
# ============================================================================

RMS_NORM_DISPATCH = r"""
            // -- G24A: RMS_NORM -------------------------------------------
            case GGML_OP_RMS_NORM: {
                float eps;
                memcpy(&eps, node->op_params, sizeof(float));
                const int64_t ncols = node->src[0]->ne[0];
                const int64_t nrows = ggml_nrows(node->src[0]);
                ggml_cuda8_rms_norm_f32(
                    (const float *) node->src[0]->data,
                    (float *)       node->data,
                    (int) nrows, (int) ncols, eps, stream);
                break;
            }"""

# The dispatch switch is most likely in ggml-cuda8.cu (the main backend file)
disp_file = find_cu("GGML_OP_SOFTMAX") or find_cu("GGML_OP_SUM_ROWS") or find_cu("GGML_OP_ADD")
if disp_file is None:
    # Last resort: try the main backend file by name
    candidate = os.path.join(BASE, "ggml-cuda8.cu")
    if os.path.exists(candidate):
        disp_file = candidate
        print("[G24A] WARNING: using %s as dispatch file (no GGML_OP_ tokens found)" % disp_file)

if disp_file is None:
    print("[G24A] ERROR: cannot find dispatch file")
    print("       .cu files: %s" % glob.glob(os.path.join(BASE, "*.cu")))
    sys.exit(1)

dsrc = read_file(disp_file)

if "GGML_OP_RMS_NORM" not in dsrc:
    ok = False
    if "default:" in dsrc:
        idx = dsrc.rfind("default:")
        dsrc = dsrc[:idx] + RMS_NORM_DISPATCH + "\n\n            " + dsrc[idx:]
        ok = True
    else:
        # Try inserting after last break; } block
        pattern = re.compile(r"break;\s*\}", re.MULTILINE)
        matches = list(pattern.finditer(dsrc))
        if matches:
            m = matches[-1]
            pos = m.end()
            dsrc = dsrc[:pos] + RMS_NORM_DISPATCH + dsrc[pos:]
            ok = True

    if ok:
        write_file(disp_file, dsrc)
        print("[G24A] Patched %s  +GGML_OP_RMS_NORM dispatch" % disp_file)
    else:
        print("[G24A] WARNING: could not find insertion point in %s" % disp_file)
        print("       You may need to add the RMS_NORM dispatch case manually.")
        print("       Snippet:\n%s" % RMS_NORM_DISPATCH)
else:
    print("[G24A] %s already has GGML_OP_RMS_NORM, skip" % disp_file)

# ============================================================================
# 4.  RMS_NORM standalone smoke test
# ============================================================================

SMOKE_FILE = os.path.join(BASE, "ggml-cuda8-rms-norm-smoke.cu")

# Figure out what the smoke tests actually #include
# Check an existing smoke test for the include pattern
existing_smoke = find_cu("CHECK_CUDA") or find_cu("cudaMalloc")
inc_line = '#include "ggml-cuda8-kernels.h"'
if existing_smoke:
    esrc = read_file(existing_smoke)
    m = re.search(r'#include\s+"([^"]+\.h)"', esrc)
    if m:
        inc_line = '#include "%s"' % m.group(1)
        print("[G24A] Smoke test will use: %s  (from %s)" % (inc_line, os.path.basename(existing_smoke)))

SMOKE_SRC = r"""// ggml-cuda8-rms-norm-smoke.cu  -  G24A: RMS_NORM standalone smoke test
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

// Forward-declare the kernel wrapper directly (no header dependency)
void ggml_cuda8_rms_norm_f32(
        const float * x, float * y,
        int nrows, int ncols, float eps,
        cudaStream_t stream);

#define CHECK_CUDA(x) do { \
    cudaError_t err_ = (x); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d  %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err_)); exit(1); } \
} while (0)

// -- CPU reference ------------------------------------------------------------
static void rms_norm_ref(const float *x, float *y,
                          int nrows, int ncols, float eps) {
    for (int r = 0; r < nrows; r++) {
        const float *xr = x + r * ncols;
        float       *yr = y + r * ncols;
        float ss = 0.0f;
        for (int c = 0; c < ncols; c++) ss += xr[c] * xr[c];
        float scale = 1.0f / sqrtf(ss / (float)ncols + eps);
        for (int c = 0; c < ncols; c++) yr[c] = xr[c] * scale;
    }
}

int main() {
    printf("ggml-cuda8-rms-norm-smoke: starting\n");

    const int nrows  = 4;
    const int ncols  = 128;
    const float eps  = 1e-5f;
    const size_t nb  = (size_t)nrows * ncols * sizeof(float);

    float *h_x   = (float *)malloc(nb);
    float *h_y   = (float *)malloc(nb);
    float *h_ref = (float *)malloc(nb);

    // deterministic pseudo-random input
    srand(42);
    for (int i = 0; i < nrows * ncols; i++)
        h_x[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;

    rms_norm_ref(h_x, h_ref, nrows, ncols, eps);

    float *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_x, nb));
    CHECK_CUDA(cudaMalloc(&d_y, nb));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, nb, cudaMemcpyHostToDevice));

    ggml_cuda8_rms_norm_f32(d_x, d_y, nrows, ncols, eps, 0);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_y, d_y, nb, cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < nrows * ncols; i++) {
        float e = fabsf(h_y[i] - h_ref[i]);
        if (e > max_err) max_err = e;
    }

    printf("  rows=%d  cols=%d  eps=%.1e  max_err=%.6e\n",
           nrows, ncols, eps, max_err);

    int pass = (max_err < 1e-4f);
    printf("ggml-cuda8-rms-norm-smoke: %s\n", pass ? "PASS" : "FAIL");

    cudaFree(d_x); cudaFree(d_y);
    free(h_x); free(h_y); free(h_ref);
    return pass ? 0 : 1;
}
"""

write_file(SMOKE_FILE, SMOKE_SRC)
print("[G24A] Created %s" % SMOKE_FILE)

# ============================================================================
# 5.  Patch CMakeLists.txt  -  add rms-norm.cu to backend lib + smoke target
# ============================================================================

cmake_path = os.path.join(BASE, "CMakeLists.txt")
cmake = read_file(cmake_path)

# 5a. Add ggml-cuda8-rms-norm.cu to the backend library source list
#     Look for an existing .cu in a cuda_add_library or set(SOURCES ...) block
if "ggml-cuda8-rms-norm.cu" not in cmake:
    # Strategy: find last occurrence of a .cu file being added to sources
    # and insert after it
    pattern = re.compile(r'(ggml-cuda8-(?:softmax|add|scalar|reduce|mulmat)\.cu)')
    matches = list(pattern.finditer(cmake))
    if matches:
        m = matches[-1]
        insert_pos = m.end()
        # Check if it's on its own line; add after that line
        line_end = cmake.find("\n", insert_pos)
        if line_end == -1:
            line_end = len(cmake)
        cmake = cmake[:line_end] + "\n    ggml-cuda8-rms-norm.cu" + cmake[line_end:]
        print("[G24A] Added ggml-cuda8-rms-norm.cu to library sources in CMakeLists.txt")
    else:
        print("[G24A] WARNING: could not find source list in CMakeLists.txt")
        print("       Add  ggml-cuda8-rms-norm.cu  to your backend library sources manually.")
else:
    print("[G24A] ggml-cuda8-rms-norm.cu already in CMakeLists.txt sources, skip")

# 5b. Add smoke test target
CMAKE_SMOKE = """
# -- G24A: RMS_NORM smoke test ------------------------------------------------
cuda_add_executable(ggml-cuda8-rms-norm-smoke
    ggml-cuda8-rms-norm-smoke.cu
    ggml-cuda8-rms-norm.cu)
target_link_libraries(ggml-cuda8-rms-norm-smoke ${CUDA_LIBRARIES})
set_target_properties(ggml-cuda8-rms-norm-smoke PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
"""

if "ggml-cuda8-rms-norm-smoke" not in cmake:
    cmake = cmake.rstrip() + "\n" + CMAKE_SMOKE + "\n"
    print("[G24A] Added rms-norm-smoke target to CMakeLists.txt")
else:
    print("[G24A] rms-norm-smoke target already in CMakeLists.txt, skip")

write_file(cmake_path, cmake)

# ============================================================================
# Done
# ============================================================================
print("""
[G24A] All patches applied.

Build & run:
  cd /workspace/notebooks/llama.cpp-ph2/build-cuda8-parent
  cmake .. && make -j$(nproc) ggml-cuda8-rms-norm-smoke
  ./bin/ggml-cuda8-rms-norm-smoke

Expected output:
  ggml-cuda8-rms-norm-smoke: starting
    rows=4  cols=128  eps=1.0e-05  max_err=<small>
  ggml-cuda8-rms-norm-smoke: PASS
""")
