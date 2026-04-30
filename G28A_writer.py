#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G28A_writer.py - ROPE kernel + dispatch wiring + standalone smoke test
   Basic mode only (mode=0, no YaRN, no mrope). Fermi-safe.
   Python 3.5+
"""
import os, sys, re

REPO = "/workspace/notebooks/llama.cpp-ph2"
BASE = os.path.join(REPO, "ggml/src/ggml-cuda8")
ENC  = {"encoding": "utf-8"}

ok_all = True

def read_file(path):
    with open(path, **ENC) as fh:
        return fh.read()

def write_file(path, content):
    with open(path, "w", **ENC) as fh:
        fh.write(content)

# ============================================================================
# 1. ggml-cuda8-rope.cu  -  ROPE kernel (Fermi-safe)
# ============================================================================

ROPE_CU = (
    '// ggml-cuda8-rope.cu  -  G28A: Rotary Positional Embedding kernel\n'
    '// Basic ROPE (mode=0, no YaRN, no mrope). Fermi-safe.\n'
    '#include <cuda_runtime.h>\n'
    '#include <cstdio>\n'
    '#include <math.h>\n'
    '\n'
    '// One thread per pair of elements across all heads/positions.\n'
    '// Layout: x[i0 + i1*ne0 + i2*ne0*ne1 + i3*ne0*ne1*ne2]\n'
    '//   ne0 = head_dim, ne1 = n_heads, ne2 = seq_len, ne3 = batch\n'
    '//   pos[i2] = position id for sequence index i2\n'
    '//   Pairs at i0 < n_dims are rotated; i0 >= n_dims pass through.\n'
    'static __global__ void kernel_rope_f32(\n'
    '        const float * __restrict__ x,\n'
    '        float       * __restrict__ dst,\n'
    '        const int   * __restrict__ pos,\n'
    '        const int ne0,\n'
    '        const int ne1,\n'
    '        const int ne2,\n'
    '        const int ne3,\n'
    '        const int n_dims,\n'
    '        const float theta_scale,\n'
    '        const float freq_scale) {\n'
    '\n'
    '    const int idx = blockIdx.x * blockDim.x + threadIdx.x;\n'
    '    const int pairs_per_row = ne0 / 2;\n'
    '    const int rows = ne1 * ne2 * ne3;\n'
    '    const int total_pairs = pairs_per_row * rows;\n'
    '    if (idx >= total_pairs) return;\n'
    '\n'
    '    const int pair = idx % pairs_per_row;\n'
    '    const int row  = idx / pairs_per_row;\n'
    '\n'
    '    const int i1 = row % ne1;\n'
    '    const int i2 = (row / ne1) % ne2;\n'
    '    const int i3 = row / (ne1 * ne2);\n'
    '\n'
    '    const int i0 = pair * 2;\n'
    '    const int offset = i0 + i1 * ne0\n'
    '                          + i2 * ne0 * ne1\n'
    '                          + i3 * ne0 * ne1 * ne2;\n'
    '\n'
    '    if (i0 >= n_dims) {\n'
    '        // Beyond rotary dims - pass through unchanged\n'
    '        dst[offset]     = x[offset];\n'
    '        dst[offset + 1] = x[offset + 1];\n'
    '        return;\n'
    '    }\n'
    '\n'
    '    const int p = pos[i2];\n'
    '    float theta = (float)p * powf(theta_scale, (float)pair) * freq_scale;\n'
    '    float cos_t = cosf(theta);\n'
    '    float sin_t = sinf(theta);\n'
    '\n'
    '    float x0 = x[offset];\n'
    '    float x1 = x[offset + 1];\n'
    '\n'
    '    dst[offset]     = x0 * cos_t - x1 * sin_t;\n'
    '    dst[offset + 1] = x0 * sin_t + x1 * cos_t;\n'
    '}\n'
    '\n'
    'extern "C" int ggml_cuda8_op_rope_f32(\n'
    '        const float * x,\n'
    '        float * dst,\n'
    '        const int * pos,\n'
    '        int ne0, int ne1, int ne2, int ne3,\n'
    '        int n_dims,\n'
    '        float freq_base,\n'
    '        float freq_scale) {\n'
    '\n'
    '    if (x == NULL || dst == NULL || pos == NULL || ne0 < 2) {\n'
    '        std::fprintf(stderr, "ggml-cuda8/rope: invalid args\\n");\n'
    '        return -1;\n'
    '    }\n'
    '\n'
    '    const float theta_scale = powf(freq_base, -2.0f / (float)n_dims);\n'
    '    const int total_pairs = (ne0 / 2) * ne1 * ne2 * ne3;\n'
    '    const int block = 256;\n'
    '    const int grid  = (total_pairs + block - 1) / block;\n'
    '\n'
    '    kernel_rope_f32<<<grid, block>>>(\n'
    '        x, dst, pos, ne0, ne1, ne2, ne3,\n'
    '        n_dims, theta_scale, freq_scale);\n'
    '\n'
    '    cudaError_t err = cudaGetLastError();\n'
    '    if (err != cudaSuccess) {\n'
    '        std::fprintf(stderr, "ggml-cuda8/rope: launch failed: %s\\n",\n'
    '            cudaGetErrorString(err));\n'
    '        return -1;\n'
    '    }\n'
    '    err = cudaDeviceSynchronize();\n'
    '    if (err != cudaSuccess) {\n'
    '        std::fprintf(stderr, "ggml-cuda8/rope: sync failed: %s\\n",\n'
    '            cudaGetErrorString(err));\n'
    '        return -1;\n'
    '    }\n'
    '    return 0;\n'
    '}\n'
)

rope_cu = os.path.join(BASE, "ggml-cuda8-rope.cu")
if not os.path.exists(rope_cu):
    write_file(rope_cu, ROPE_CU)
    print("[G28A] Created %s" % rope_cu)
else:
    print("[G28A] %s already exists, skip" % rope_cu)

# ============================================================================
# 2. ggml-cuda8-rope-smoke.cu  -  standalone smoke test
# ============================================================================

SMOKE_CU = (
    '// ggml-cuda8-rope-smoke.cu  -  G28A: ROPE standalone smoke test\n'
    '// Tests basic ROPE (mode=0) with head_dim=64, n_heads=4, seq_len=8\n'
    '#include <cstdio>\n'
    '#include <cstdlib>\n'
    '#include <cstring>\n'
    '#include <cmath>\n'
    '#include <cuda_runtime.h>\n'
    '\n'
    'extern "C" int ggml_cuda8_op_rope_f32(\n'
    '        const float * x, float * dst, const int * pos,\n'
    '        int ne0, int ne1, int ne2, int ne3,\n'
    '        int n_dims, float freq_base, float freq_scale);\n'
    '\n'
    '#define CHECK_CUDA(x) do { \\\n'
    '    cudaError_t err_ = (x); \\\n'
    '    if (err_ != cudaSuccess) { \\\n'
    '        fprintf(stderr, "CUDA %s:%d  %s\\n", __FILE__, __LINE__, \\\n'
    '                cudaGetErrorString(err_)); exit(1); } \\\n'
    '} while (0)\n'
    '\n'
    '// CPU reference: basic ROPE (mode=0, no YaRN)\n'
    'static void rope_ref(const float *x, float *dst, const int *pos,\n'
    '                      int ne0, int ne1, int ne2, int ne3,\n'
    '                      int n_dims, float freq_base, float freq_scale) {\n'
    '    float theta_scale = powf(freq_base, -2.0f / (float)n_dims);\n'
    '    for (int i3 = 0; i3 < ne3; i3++) {\n'
    '        for (int i2 = 0; i2 < ne2; i2++) {\n'
    '            for (int i1 = 0; i1 < ne1; i1++) {\n'
    '                for (int pair = 0; pair < ne0/2; pair++) {\n'
    '                    int i0 = pair * 2;\n'
    '                    int idx = i0 + i1*ne0 + i2*ne0*ne1 + i3*ne0*ne1*ne2;\n'
    '                    if (i0 >= n_dims) {\n'
    '                        dst[idx]   = x[idx];\n'
    '                        dst[idx+1] = x[idx+1];\n'
    '                    } else {\n'
    '                        float theta = (float)pos[i2] * powf(theta_scale, (float)pair) * freq_scale;\n'
    '                        float c = cosf(theta);\n'
    '                        float s = sinf(theta);\n'
    '                        dst[idx]   = x[idx]*c - x[idx+1]*s;\n'
    '                        dst[idx+1] = x[idx]*s + x[idx+1]*c;\n'
    '                    }\n'
    '                }\n'
    '            }\n'
    '        }\n'
    '    }\n'
    '}\n'
    '\n'
    'int main() {\n'
    '    printf("ggml-cuda8-rope-smoke: starting\\n");\n'
    '\n'
    '    const int ne0 = 64;    // head_dim\n'
    '    const int ne1 = 4;     // n_heads\n'
    '    const int ne2 = 8;     // seq_len\n'
    '    const int ne3 = 1;     // batch\n'
    '    const int n_dims = 64; // full rotary\n'
    '    const float freq_base  = 10000.0f;\n'
    '    const float freq_scale = 1.0f;\n'
    '\n'
    '    const int n_floats = ne0 * ne1 * ne2 * ne3;\n'
    '    const size_t data_bytes = (size_t)n_floats * sizeof(float);\n'
    '    const size_t pos_bytes  = (size_t)ne2 * sizeof(int);\n'
    '\n'
    '    float *h_x   = (float *)malloc(data_bytes);\n'
    '    float *h_y   = (float *)malloc(data_bytes);\n'
    '    float *h_ref = (float *)malloc(data_bytes);\n'
    '    int   *h_pos = (int *)malloc(pos_bytes);\n'
    '\n'
    '    srand(42);\n'
    '    for (int i = 0; i < n_floats; i++)\n'
    '        h_x[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;\n'
    '    for (int i = 0; i < ne2; i++)\n'
    '        h_pos[i] = i;  // positions 0,1,2,...,7\n'
    '\n'
    '    rope_ref(h_x, h_ref, h_pos, ne0, ne1, ne2, ne3,\n'
    '             n_dims, freq_base, freq_scale);\n'
    '\n'
    '    float *d_x, *d_y;\n'
    '    int   *d_pos;\n'
    '    CHECK_CUDA(cudaMalloc(&d_x,   data_bytes));\n'
    '    CHECK_CUDA(cudaMalloc(&d_y,   data_bytes));\n'
    '    CHECK_CUDA(cudaMalloc(&d_pos, pos_bytes));\n'
    '    CHECK_CUDA(cudaMemcpy(d_x,   h_x,   data_bytes, cudaMemcpyHostToDevice));\n'
    '    CHECK_CUDA(cudaMemcpy(d_pos, h_pos, pos_bytes,  cudaMemcpyHostToDevice));\n'
    '\n'
    '    int rc = ggml_cuda8_op_rope_f32(d_x, d_y, d_pos,\n'
    '                ne0, ne1, ne2, ne3, n_dims, freq_base, freq_scale);\n'
    '    if (rc != 0) {\n'
    '        printf("ggml-cuda8-rope-smoke: launch FAILED rc=%d\\n", rc);\n'
    '        return 1;\n'
    '    }\n'
    '\n'
    '    CHECK_CUDA(cudaMemcpy(h_y, d_y, data_bytes, cudaMemcpyDeviceToHost));\n'
    '\n'
    '    float max_err = 0.0f;\n'
    '    for (int i = 0; i < n_floats; i++) {\n'
    '        float e = fabsf(h_y[i] - h_ref[i]);\n'
    '        if (e > max_err) max_err = e;\n'
    '    }\n'
    '\n'
    '    printf("  head_dim=%d  n_heads=%d  seq_len=%d  n_dims=%d\\n",\n'
    '           ne0, ne1, ne2, n_dims);\n'
    '    printf("  freq_base=%.1f  freq_scale=%.1f\\n", freq_base, freq_scale);\n'
    '    printf("  max_err=%.6e\\n", max_err);\n'
    '\n'
    '    int pass = (max_err < 1e-4f);\n'
    '    printf("ggml-cuda8-rope-smoke: %s\\n", pass ? "PASS" : "FAIL");\n'
    '\n'
    '    cudaFree(d_x); cudaFree(d_y); cudaFree(d_pos);\n'
    '    free(h_x); free(h_y); free(h_ref); free(h_pos);\n'
    '    return pass ? 0 : 1;\n'
    '}\n'
)

smoke_path = os.path.join(BASE, "ggml-cuda8-rope-smoke.cu")
if not os.path.exists(smoke_path):
    write_file(smoke_path, SMOKE_CU)
    print("[G28A] Created %s" % smoke_path)
else:
    print("[G28A] %s already exists, skip" % smoke_path)

# ============================================================================
# 3. Patch dispatch.h  -  add GGML_CUDA8_OP_ROPE_F32 enum
# ============================================================================

dispatch_h = os.path.join(BASE, "ggml-cuda8-dispatch.h")
src = read_file(dispatch_h)

if "GGML_CUDA8_OP_ROPE_F32" not in src:
    anchor = "GGML_CUDA8_OP_MUL_F32,"
    if anchor in src:
        idx = src.index(anchor) + len(anchor)
        src = src[:idx] + "\n\n    GGML_CUDA8_OP_ROPE_F32," + src[idx:]
    else:
        idx = src.rfind("};")
        src = src[:idx] + "    GGML_CUDA8_OP_ROPE_F32,\n" + src[idx:]
    write_file(dispatch_h, src)
    print("[G28A] Patched %s  +ROPE_F32 enum" % dispatch_h)
else:
    print("[G28A] %s already has ROPE_F32 enum, skip" % dispatch_h)

# ============================================================================
# 4. Patch dispatch.cpp  -  add helpers + switch cases
# ============================================================================

dispatch_cpp = os.path.join(BASE, "ggml-cuda8-dispatch.cpp")
src = read_file(dispatch_cpp)

if "GGML_CUDA8_OP_ROPE_F32" not in src:

    # 4a. Insert extern decl + static helpers before dispatch_supported
    HELPERS = (
        '\n'
        '// -- G28A: ROPE_F32 helpers ---------------------------------------------------\n'
        'extern "C" int ggml_cuda8_op_rope_f32(\n'
        '        const float * x, float * dst, const int * pos,\n'
        '        int ne0, int ne1, int ne2, int ne3,\n'
        '        int n_dims, float freq_base, float freq_scale);\n'
        '\n'
        'static int ggml_cuda8_supported_rope_f32(\n'
        '        const struct ggml_cuda8_context * ctx,\n'
        '        const struct ggml_tensor * src0,\n'
        '        const struct ggml_tensor * src1,\n'
        '        const struct ggml_tensor * dst) {\n'
        '    (void) ctx;\n'
        '    if (src0 == NULL || src1 == NULL || dst == NULL) return 0;\n'
        '    if (src0->type != GGML_TYPE_F32) return 0;\n'
        '    if (src1->type != GGML_TYPE_I32) return 0;\n'
        '    if (dst->type  != GGML_TYPE_F32) return 0;\n'
        '    return 1;\n'
        '}\n'
        '\n'
        'static int ggml_cuda8_exec_rope_f32(\n'
        '        struct ggml_cuda8_context * ctx,\n'
        '        const struct ggml_tensor * src0,\n'
        '        const struct ggml_tensor * src1,\n'
        '        struct ggml_tensor * dst) {\n'
        '    (void) ctx;\n'
        '\n'
        '    // Extract op_params from dst (the graph node)\n'
        '    int32_t op_params[15];\n'
        '    std::memcpy(op_params, dst->op_params, sizeof(op_params));\n'
        '\n'
        '    int n_dims = op_params[1];\n'
        '    int mode   = op_params[2];\n'
        '\n'
        '    float freq_base, freq_scale, ext_factor;\n'
        '    std::memcpy(&freq_base,   &op_params[5], sizeof(float));\n'
        '    std::memcpy(&freq_scale,  &op_params[6], sizeof(float));\n'
        '    std::memcpy(&ext_factor,  &op_params[7], sizeof(float));\n'
        '\n'
        '    if (mode != 0 || ext_factor != 0.0f) {\n'
        '        std::fprintf(stderr, "ggml-cuda8/rope: unsupported mode=%d ext=%.1f\\n",\n'
        '                     mode, (double)ext_factor);\n'
        '        return -1;\n'
        '    }\n'
        '\n'
        '    int ne0 = (int) src0->ne[0];\n'
        '    int ne1 = (int) src0->ne[1];\n'
        '    int ne2 = (int) src0->ne[2];\n'
        '    int ne3 = (int) src0->ne[3];\n'
        '\n'
        '    return ggml_cuda8_op_rope_f32(\n'
        '        (const float *) src0->data,\n'
        '        (float *)       dst->data,\n'
        '        (const int *)   src1->data,\n'
        '        ne0, ne1, ne2, ne3,\n'
        '        n_dims, freq_base, freq_scale);\n'
        '}\n'
        '\n'
    )

    anchor_func = "int ggml_cuda8_dispatch_supported("
    if anchor_func in src:
        idx = src.index(anchor_func)
        src = src[:idx] + HELPERS + src[idx:]
        print("[G28A] Inserted ROPE_F32 helper functions")
    else:
        print("[G28A] WARNING: cannot find dispatch_supported anchor")
        ok_all = False

    # 4b. Insert op_name case after MUL_F32 line
    m = re.search(
        r'(case GGML_CUDA8_OP_MUL_F32:\s+return "MUL_F32";)',
        src)
    if m:
        ins = '\n        case GGML_CUDA8_OP_ROPE_F32:             return "ROPE_F32";'
        src = src[:m.end()] + ins + src[m.end():]
        print("[G28A] Inserted op_name case")
    else:
        print("[G28A] WARNING: cannot find MUL_F32 op_name line")
        ok_all = False

    # 4c. Insert supported case after MUL_F32 supported case
    m = re.search(
        r'(case GGML_CUDA8_OP_MUL_F32:\s*\n\s*return ggml_cuda8_supported_mul_f32\(ctx, src0, src1, dst\);)',
        src)
    if m:
        ins = ("\n\n        case GGML_CUDA8_OP_ROPE_F32:\n"
               "            return ggml_cuda8_supported_rope_f32(ctx, src0, src1, dst);")
        src = src[:m.end()] + ins + src[m.end():]
        print("[G28A] Inserted supported case")
    else:
        print("[G28A] WARNING: cannot find MUL_F32 supported case")
        ok_all = False

    # 4d. Insert execute case after MUL_F32 execute case
    m = re.search(
        r'(case GGML_CUDA8_OP_MUL_F32:\s*\n\s*return ggml_cuda8_exec_mul_f32\(ctx, src0, src1, dst\);)',
        src)
    if m:
        ins = ("\n\n        case GGML_CUDA8_OP_ROPE_F32:\n"
               "            return ggml_cuda8_exec_rope_f32(ctx, src0, src1, dst);")
        src = src[:m.end()] + ins + src[m.end():]
        print("[G28A] Inserted execute case")
    else:
        print("[G28A] WARNING: cannot find MUL_F32 execute case")
        ok_all = False

    write_file(dispatch_cpp, src)
    print("[G28A] Patched %s" % dispatch_cpp)
else:
    print("[G28A] %s already has ROPE_F32, skip" % dispatch_cpp)

# ============================================================================
# 5. Patch backend.cpp  -  add GGML_OP_ROPE case
# ============================================================================

backend_cpp = os.path.join(BASE, "ggml-cuda8-ggml-backend.cpp")
src = read_file(backend_cpp)

if "GGML_OP_ROPE" not in src:
    CASE = (
        '\n'
        '            case GGML_OP_ROPE: {\n'
        '                if (node->type != GGML_TYPE_F32 ||\n'
        '                    src0 == NULL || src0->type != GGML_TYPE_F32 ||\n'
        '                    src1 == NULL || src1->type != GGML_TYPE_I32) {\n'
        '                    std::fprintf(stderr,\n'
        '                        "ggml-cuda8/backend graph_compute: ROPE node %d unsupported types\\n", i);\n'
        '                    ggml_cuda8_context_destroy(ctx);\n'
        '                    return (enum ggml_status) -1;\n'
        '                }\n'
        '\n'
        '                // Do NOT flatten: ROPE needs multi-dim shape + op_params\n'
        '                dispatch_src0 = src0;\n'
        '                dispatch_src1 = src1;\n'
        '                dispatch_dst  = node;\n'
        '\n'
        '                cuda8_op = GGML_CUDA8_OP_ROPE_F32;\n'
        '                opname = "ROPE_F32";\n'
        '            } break;\n\n'
    )

    if "default:" in src:
        idx = src.rfind("default:")
        src = src[:idx] + CASE + "            " + src[idx:]
        write_file(backend_cpp, src)
        print("[G28A] Patched %s  +GGML_OP_ROPE" % backend_cpp)
    else:
        print("[G28A] WARNING: no default: in backend.cpp")
        ok_all = False
else:
    print("[G28A] %s already has GGML_OP_ROPE, skip" % backend_cpp)

# ============================================================================
# 6. Patch CMakeLists.txt
# ============================================================================

cmake_path = os.path.join(BASE, "CMakeLists.txt")
cmake = read_file(cmake_path)

# 6a. Add ggml-cuda8-rope.cu to library sources
if "ggml-cuda8-rope.cu" not in cmake:
    anchor = "ggml-cuda8-mul.cu"
    if anchor in cmake:
        idx = cmake.index(anchor) + len(anchor)
        line_end = cmake.find("\n", idx)
        if line_end == -1:
            line_end = len(cmake)
        cmake = cmake[:line_end] + "\n    ggml-cuda8-rope.cu" + cmake[line_end:]
        print("[G28A] Added ggml-cuda8-rope.cu to library sources")
    else:
        print("[G28A] WARNING: cannot find mul.cu anchor in CMakeLists.txt")
        ok_all = False
else:
    print("[G28A] ggml-cuda8-rope.cu already in CMakeLists.txt, skip")

# 6b. Add smoke test target
CMAKE_SMOKE = (
    '\n'
    '# -- G28A: ROPE smoke test ----------------------------------------------------\n'
    'cuda_add_executable(ggml-cuda8-rope-smoke\n'
    '    ggml-cuda8-rope-smoke.cu\n'
    '    ggml-cuda8-rope.cu)\n'
    'target_link_libraries(ggml-cuda8-rope-smoke ${CUDA_LIBRARIES})\n'
    'set_target_properties(ggml-cuda8-rope-smoke PROPERTIES\n'
    '    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")\n'
)

if "ggml-cuda8-rope-smoke" not in cmake:
    cmake = cmake.rstrip() + "\n" + CMAKE_SMOKE + "\n"
    print("[G28A] Added rope-smoke target")
else:
    print("[G28A] rope-smoke target already in CMakeLists.txt, skip")

write_file(cmake_path, cmake)

# ============================================================================
# Done
# ============================================================================
if ok_all:
    print("""
[G28A] All patches applied.

Build & test:
  cd /workspace/notebooks/llama.cpp-ph2/build-cuda8-parent
  cmake .. && make -j$(nproc) ggml-cuda8-rope-smoke
  ./bin/ggml-cuda8-rope-smoke

Expected output:
  ggml-cuda8-rope-smoke: starting
    head_dim=64  n_heads=4  seq_len=8  n_dims=64
    freq_base=10000.0  freq_scale=1.0
    max_err=<small>
  ggml-cuda8-rope-smoke: PASS
""")
else:
    print("\n[G28A] Some patches had warnings -- review output above.")
