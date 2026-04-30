#!/usr/bin/env python3
# G24A_writer.py
#
# Create G24A real graph-builder quantized residual scaled-softmax-sumrows pipeline:
#
#   h       = mul_mat(Q8_0, x)
#   scaled  = h * scale
#   biased  = scaled + residual
#   prob    = softmax(biased)
#   row_sum = sum_rows(prob)
#
# Derives from validated G23A:
#   Q8_0 MUL_MAT -> residual ADD -> SOFTMAX -> SUM_ROWS
#
# Python 3.5-compatible.

import os
import re
import time

ROOT = "/workspace/notebooks/llama.cpp-ph2"
BASE = os.path.join(ROOT, "ggml/src/ggml-cuda8")

SRC_IN = os.path.join(BASE, "ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-sumrows-smoke.cpp")
SRC_OUT = os.path.join(BASE, "ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-smoke.cpp")
CMAKE = os.path.join(BASE, "CMakeLists.txt")

TARGET = "ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-smoke"

OLD_NAME = "ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-sumrows-smoke"
NEW_NAME = "ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-smoke"

CMAKE_BLOCK = r'''
# ---------------------------------------------------------------------------
# G24A real GGML graph-builder Q8_0 MUL_MAT -> MUL_SCALAR -> residual ADD -> SOFTMAX -> SUM_ROWS smoke
# Uses standalone-GC GGML graph-builder subset.
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-smoke
    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-smoke.cpp
)

target_include_directories(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_compile_definitions(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-smoke PRIVATE
    GGML_VERSION="${GGML_VERSION}"
    GGML_COMMIT="${GGML_COMMIT}"
)

target_compile_options(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-smoke PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:C>:-fdata-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-fdata-sections>
)

set_property(TARGET ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-smoke APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--gc-sections")

target_link_libraries(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)
'''


def read_file(path):
    with open(path, "r") as f:
        return f.read()


def write_file(path, data):
    with open(path, "w") as f:
        f.write(data)


def backup(path, tag):
    if os.path.exists(path):
        data = read_file(path)
        b = path + "." + tag + "-backup-" + str(int(time.time()))
        write_file(b, data)
        print("backup", b)
        return data
    return ""


def sub_once(s, pattern, repl, label):
    s_new, n = re.subn(pattern, repl, s, count=1, flags=re.S)
    if n != 1:
        raise RuntimeError("could not patch: " + label)
    return s_new


s = read_file(SRC_IN)

s = s.replace(OLD_NAME, NEW_NAME)

s = s.replace(
    "G23A: real GGML graph-builder Q8_0 MUL_MAT -> residual ADD -> SOFTMAX -> SUM_ROWS branch through CUDA8 graph_compute.",
    "G24A: real GGML graph-builder Q8_0 MUL_MAT -> MUL_SCALAR -> residual ADD -> SOFTMAX -> SUM_ROWS branch through CUDA8 graph_compute."
)

# Add CPU scalar reference helper if absent.
helper_insert = r'''
static void mul_scalar_ref(
    const std::vector<float> & a,
    float scale,
    std::vector<float> & y
) {
    for (size_t i = 0; i < y.size(); ++i) {
        y[i] = a[i] * scale;
    }
}

'''

if "static void mul_scalar_ref(" not in s:
    anchor = "static void cpu_ref_q8_mmv("
    idx = s.find(anchor)
    if idx < 0:
        raise RuntimeError("could not find cpu_ref_q8_mmv anchor")
    s = s[:idx] + helper_insert + s[idx:]

# Tensor creation: insert scale/scaled between h and residual.
s = sub_once(
    s,
    r'(    ggml_tensor \* Aq\s*=\s*ggml_new_tensor_2d\(gctx, GGML_TYPE_Q8_0, cols, rows\);\n'
    r'    ggml_tensor \* x\s*=\s*ggml_new_tensor_1d\(gctx, GGML_TYPE_F32, cols\);\n'
    r'    ggml_tensor \* h\s*=\s*ggml_mul_mat\(gctx, Aq, x\);\n)'
    r'    ggml_tensor \* residual\s*=\s*ggml_new_tensor_1d\(gctx, GGML_TYPE_F32, rows\);\n'
    r'    ggml_tensor \* biased\s*=\s*ggml_add\(gctx, h, residual\);\n'
    r'    ggml_tensor \* prob\s*=\s*ggml_soft_max\(gctx, biased\);\n'
    r'    ggml_tensor \* y\s*=\s*ggml_sum_rows\(gctx, prob\);\n\n'
    r'    if \(Aq == NULL \|\| x == NULL \|\| h == NULL \|\| residual == NULL \|\| biased == NULL \|\| prob == NULL \|\| y == NULL\) \{',
    r'\1'
    '    ggml_tensor * scale    = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 1);\n'
    '    ggml_tensor * scaled   = ggml_mul(gctx, h, scale);\n'
    '    ggml_tensor * residual = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, rows);\n'
    '    ggml_tensor * biased   = ggml_add(gctx, scaled, residual);\n'
    '    ggml_tensor * prob     = ggml_soft_max(gctx, biased);\n'
    '    ggml_tensor * y        = ggml_sum_rows(gctx, prob);\n\n'
    '    if (Aq == NULL || x == NULL || h == NULL || scale == NULL || scaled == NULL || residual == NULL || biased == NULL || prob == NULL || y == NULL) {',
    "tensor creation"
)

s = s.replace("graph->n_nodes != 4", "graph->n_nodes != 5", 1)
s = s.replace("expected 4 graph nodes", "expected 5 graph nodes", 1)

# Offsets: add scale scalar and scaled vector.
s = sub_once(
    s,
    r'    const size_t off_Aq\s*=\s*0;\n'
    r'    const size_t off_x\s*=\s*8192;\n'
    r'    const size_t off_h\s*=\s*12288;\n'
    r'    const size_t off_residual\s*=\s*16384;\n'
    r'    const size_t off_biased\s*=\s*20480;\n'
    r'    const size_t off_prob\s*=\s*24576;\n'
    r'    const size_t off_y\s*=\s*28672;\n'
    r'    const size_t total_size\s*=\s*28928;\n',
    '    const size_t off_Aq       = 0;\n'
    '    const size_t off_x        = 8192;\n'
    '    const size_t off_h        = 12288;\n'
    '    const size_t off_scale    = 16384;\n'
    '    const size_t off_scaled   = 16640;\n'
    '    const size_t off_residual = 20480;\n'
    '    const size_t off_biased   = 24576;\n'
    '    const size_t off_prob     = 28672;\n'
    '    const size_t off_y        = 32768;\n'
    '    const size_t total_size = 33024;\n',
    "offset block"
)

s = s.replace(
    'std::printf("backend-owned q8_0 residual-add-softmax-sumrows pipeline buffer size: %zu\\n", buffer->size);',
    'std::printf("backend-owned q8_0 residual-scale-add-softmax-sumrows pipeline buffer size: %zu\\n", buffer->size);'
)

# Layouts.
s = sub_once(
    s,
    r'    force_2d_q8_0_data_layout\(Aq, cols, rows, base_u8 \+ off_Aq\);\n'
    r'    force_1d_f32_data_layout\(x,\s*cols, base_u8 \+ off_x\);\n'
    r'    force_1d_f32_data_layout\(h,\s*rows, base_u8 \+ off_h\);\n'
    r'    force_1d_f32_data_layout\(residual,\s*rows, base_u8 \+ off_residual\);\n'
    r'    force_1d_f32_data_layout\(biased,\s*rows, base_u8 \+ off_biased\);\n'
    r'    force_1d_f32_data_layout\(prob,\s*rows, base_u8 \+ off_prob\);\n'
    r'    force_1d_f32_data_layout\(y,\s*1,\s*base_u8 \+ off_y\);\n',
    '    force_2d_q8_0_data_layout(Aq, cols, rows, base_u8 + off_Aq);\n'
    '    force_1d_f32_data_layout(x,        cols, base_u8 + off_x);\n'
    '    force_1d_f32_data_layout(h,        rows, base_u8 + off_h);\n'
    '    force_1d_f32_data_layout(scale,    1,    base_u8 + off_scale);\n'
    '    force_1d_f32_data_layout(scaled,   rows, base_u8 + off_scaled);\n'
    '    force_1d_f32_data_layout(residual, rows, base_u8 + off_residual);\n'
    '    force_1d_f32_data_layout(biased,   rows, base_u8 + off_biased);\n'
    '    force_1d_f32_data_layout(prob,     rows, base_u8 + off_prob);\n'
    '    force_1d_f32_data_layout(y,        1,    base_u8 + off_y);\n',
    "force layouts"
)

# Init tensors.
s = sub_once(
    s,
    r'    if \(buffer->iface.init_tensor\(buffer, Aq\)\s*!= GGML_STATUS_SUCCESS \|\|\n'
    r'        buffer->iface.init_tensor\(buffer, x\)\s*!= GGML_STATUS_SUCCESS \|\|\n'
    r'        buffer->iface.init_tensor\(buffer, h\)\s*!= GGML_STATUS_SUCCESS \|\|\n'
    r'        buffer->iface.init_tensor\(buffer, residual\)\s*!= GGML_STATUS_SUCCESS \|\|\n'
    r'        buffer->iface.init_tensor\(buffer, biased\)\s*!= GGML_STATUS_SUCCESS \|\|\n'
    r'        buffer->iface.init_tensor\(buffer, prob\)\s*!= GGML_STATUS_SUCCESS \|\|\n'
    r'        buffer->iface.init_tensor\(buffer, y\)\s*!= GGML_STATUS_SUCCESS\) \{',
    '    if (buffer->iface.init_tensor(buffer, Aq)       != GGML_STATUS_SUCCESS ||\n'
    '        buffer->iface.init_tensor(buffer, x)        != GGML_STATUS_SUCCESS ||\n'
    '        buffer->iface.init_tensor(buffer, h)        != GGML_STATUS_SUCCESS ||\n'
    '        buffer->iface.init_tensor(buffer, scale)    != GGML_STATUS_SUCCESS ||\n'
    '        buffer->iface.init_tensor(buffer, scaled)   != GGML_STATUS_SUCCESS ||\n'
    '        buffer->iface.init_tensor(buffer, residual) != GGML_STATUS_SUCCESS ||\n'
    '        buffer->iface.init_tensor(buffer, biased)   != GGML_STATUS_SUCCESS ||\n'
    '        buffer->iface.init_tensor(buffer, prob)     != GGML_STATUS_SUCCESS ||\n'
    '        buffer->iface.init_tensor(buffer, y)        != GGML_STATUS_SUCCESS) {',
    "init tensors"
)

# Residency checks.
s = sub_once(
    s,
    r'    if \(!expect_resident\("Aq",\s*Aq,\s*bytes_Aq,\s*buffer, off_Aq\)\s*\|\|\n'
    r'        !expect_resident\("x",\s*x,\s*bytes_x,\s*buffer, off_x\)\s*\|\|\n'
    r'        !expect_resident\("h",\s*h,\s*bytes_y,\s*buffer, off_h\)\s*\|\|\n'
    r'        !expect_resident\("residual",\s*residual,\s*bytes_y,\s*buffer, off_residual\)\s*\|\|\n'
    r'        !expect_resident\("biased",\s*biased,\s*bytes_y,\s*buffer, off_biased\)\s*\|\|\n'
    r'        !expect_resident\("prob",\s*prob,\s*bytes_y,\s*buffer, off_prob\)\s*\|\|\n'
    r'        !expect_resident\("y",\s*y,\s*sizeof\(float\),\s*buffer, off_y\)\) \{',
    '    if (!expect_resident("Aq",       Aq,       bytes_Aq,       buffer, off_Aq)       ||\n'
    '        !expect_resident("x",        x,        bytes_x,        buffer, off_x)        ||\n'
    '        !expect_resident("h",        h,        bytes_y,        buffer, off_h)        ||\n'
    '        !expect_resident("scale",    scale,    sizeof(float),  buffer, off_scale)    ||\n'
    '        !expect_resident("scaled",   scaled,   bytes_y,        buffer, off_scaled)   ||\n'
    '        !expect_resident("residual", residual, bytes_y,        buffer, off_residual) ||\n'
    '        !expect_resident("biased",   biased,   bytes_y,        buffer, off_biased)   ||\n'
    '        !expect_resident("prob",     prob,     bytes_y,        buffer, off_prob)     ||\n'
    '        !expect_resident("y",        y,        sizeof(float),  buffer, off_y)) {',
    "residency checks"
)

# Host reference: add scaled_ref and scale_host.
s = sub_once(
    s,
    r'    std::vector<float> h_ref\(rows, 0\.0f\);\n'
    r'    std::vector<float> residual_host\(rows, 0\.0f\);\n'
    r'    std::vector<float> residual_after\(rows, 0\.0f\);\n'
    r'    std::vector<float> biased_ref\(rows, 0\.0f\);\n'
    r'    std::vector<float> prob_ref\(rows, 0\.0f\);\n'
    r'    std::vector<float> y_ref\(1, 0\.0f\);\n'
    r'    std::vector<float> y_out\(1, 0\.0f\);\n\n'
    r'    fill_f32_matrix\(A_f32, rows, cols\);\n'
    r'    pack_q8_0\(A_f32, Aq_host, rows, cols\);\n'
    r'    fill_x\(x_host\);\n'
    r'    fill_residual\(residual_host\);\n'
    r'    cpu_ref_q8_mmv\(Aq_host, x_host, h_ref, rows, cols\);\n'
    r'    add_ref\(h_ref, residual_host, biased_ref\);\n'
    r'    softmax_ref\(biased_ref, prob_ref\);\n'
    r'    sum_ref\(prob_ref, y_ref\);\n',
    '    std::vector<float> h_ref(rows, 0.0f);\n'
    '    std::vector<float> scaled_ref(rows, 0.0f);\n'
    '    std::vector<float> residual_host(rows, 0.0f);\n'
    '    std::vector<float> residual_after(rows, 0.0f);\n'
    '    std::vector<float> biased_ref(rows, 0.0f);\n'
    '    std::vector<float> prob_ref(rows, 0.0f);\n'
    '    std::vector<float> y_ref(1, 0.0f);\n'
    '    std::vector<float> y_out(1, 0.0f);\n'
    '    std::vector<float> scale_host(1);\n\n'
    '    scale_host[0] = 0.75f;\n\n'
    '    fill_f32_matrix(A_f32, rows, cols);\n'
    '    pack_q8_0(A_f32, Aq_host, rows, cols);\n'
    '    fill_x(x_host);\n'
    '    fill_residual(residual_host);\n'
    '    cpu_ref_q8_mmv(Aq_host, x_host, h_ref, rows, cols);\n'
    '    mul_scalar_ref(h_ref, scale_host[0], scaled_ref);\n'
    '    add_ref(scaled_ref, residual_host, biased_ref);\n'
    '    softmax_ref(biased_ref, prob_ref);\n'
    '    sum_ref(prob_ref, y_ref);\n',
    "host reference setup"
)

# Upload scale tensor too.
s = sub_once(
    s,
    r'    buffer->iface.set_tensor\(buffer, Aq,\s*&Aq_host\[0\],\s*0, bytes_Aq\);\n'
    r'    buffer->iface.set_tensor\(buffer, x,\s*&x_host\[0\],\s*0, bytes_x\);\n'
    r'    buffer->iface.set_tensor\(buffer, residual,\s*&residual_host\[0\],\s*0, bytes_y\);\n',
    '    buffer->iface.set_tensor(buffer, Aq,       &Aq_host[0],       0, bytes_Aq);\n'
    '    buffer->iface.set_tensor(buffer, x,        &x_host[0],        0, bytes_x);\n'
    '    buffer->iface.set_tensor(buffer, scale,    &scale_host[0],    0, sizeof(float));\n'
    '    buffer->iface.set_tensor(buffer, residual, &residual_host[0], 0, bytes_y);\n',
    "set tensors"
)

s = s.replace(
    "real graph-builder packed Q8_0xF32_VEC->RESIDUAL_ADD_F32->SOFTMAX_ROWS_F32->REDUCE_SUM_ROWS_F32",
    "real graph-builder packed Q8_0xF32_VEC->MUL_SCALAR_F32->RESIDUAL_ADD_F32->SOFTMAX_ROWS_F32->REDUCE_SUM_ROWS_F32"
)

backup(SRC_OUT, "g24a")
write_file(SRC_OUT, s)
print("wrote", SRC_OUT)

cm = backup(CMAKE, "g24a")
if TARGET not in cm:
    if not cm.endswith("\n"):
        cm += "\n"
    cm += "\n" + CMAKE_BLOCK.strip() + "\n"
    write_file(CMAKE, cm)
    print("patched", CMAKE)
else:
    print("CMake target already present")

print("G24A writer complete.")
