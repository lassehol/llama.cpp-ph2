#!/usr/bin/env python3
# G18A_writer.py
#
# Create G18A real graph-builder two-op quantized pipeline smoke:
#
#   h = mul_mat(Q8_0, x)
#   y = h + bias
#
# This reuses the validated G17C packed-Q8_0 smoke as a template, then writes
# a new source file and CMake target.
#
# Python 3.5-compatible: no f-strings.

import os
import time

ROOT = "/workspace/notebooks/llama.cpp-ph2"
BASE = os.path.join(ROOT, "ggml/src/ggml-cuda8")

SRC_IN = os.path.join(BASE, "ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke.cpp")
SRC_OUT = os.path.join(BASE, "ggml-cuda8-ggml-backend-graph-builder-q8_0-add-smoke.cpp")
CMAKE = os.path.join(BASE, "CMakeLists.txt")

TARGET = "ggml-cuda8-ggml-backend-graph-builder-q8_0-add-smoke"

OLD_NAME = "ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke"
NEW_NAME = "ggml-cuda8-ggml-backend-graph-builder-q8_0-add-smoke"

CMAKE_BLOCK = r'''
# ---------------------------------------------------------------------------
# G18A real GGML graph-builder Q8_0 MUL_MAT -> ADD pipeline smoke
# Uses standalone-GC GGML graph-builder subset.
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-backend-graph-builder-q8_0-add-smoke
    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    ggml-cuda8-ggml-backend-graph-builder-q8_0-add-smoke.cpp
)

target_include_directories(ggml-cuda8-ggml-backend-graph-builder-q8_0-add-smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_compile_definitions(ggml-cuda8-ggml-backend-graph-builder-q8_0-add-smoke PRIVATE
    GGML_VERSION="${GGML_VERSION}"
    GGML_COMMIT="${GGML_COMMIT}"
)

target_compile_options(ggml-cuda8-ggml-backend-graph-builder-q8_0-add-smoke PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:C>:-fdata-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-fdata-sections>
)

set_property(TARGET ggml-cuda8-ggml-backend-graph-builder-q8_0-add-smoke APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--gc-sections")

target_link_libraries(ggml-cuda8-ggml-backend-graph-builder-q8_0-add-smoke
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


def replace_once(s, old, new, label):
    if old not in s:
        raise RuntimeError("could not find block: " + label)
    return s.replace(old, new, 1)


s = read_file(SRC_IN)
s = s.replace(OLD_NAME, NEW_NAME)
s = s.replace(
    "G17A: real GGML graph-builder Q8_0 x F32 vector smoke through CUDA8 graph_compute.",
    "G18A: real GGML graph-builder Q8_0 MUL_MAT -> ADD pipeline through CUDA8 graph_compute."
)

# Add bias/reference helpers before the CPU Q8_0 MMV reference.
helper_insert = r'''
static void fill_bias(
    std::vector<float> & bias
) {
    for (size_t i = 0; i < bias.size(); ++i) {
        const int v = ((int) i * 11 + 3) % 23;
        bias[i] = ((float) v - 11.0f) * 0.0078125f;
    }
}

static void add_ref(
    const std::vector<float> & a,
    const std::vector<float> & b,
    std::vector<float> & y
) {
    for (size_t i = 0; i < y.size(); ++i) {
        y[i] = a[i] + b[i];
    }
}

'''

cpu_ref_anchor = "static void cpu_ref_q8_mmv("
if helper_insert.strip() not in s:
    idx = s.find(cpu_ref_anchor)
    if idx < 0:
        raise RuntimeError("could not find cpu_ref_q8_mmv anchor")
    s = s[:idx] + helper_insert + s[idx:]

old = '''    ggml_tensor * Aq = ggml_new_tensor_2d(gctx, GGML_TYPE_Q8_0, cols, rows);
    ggml_tensor * x  = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, cols);
    ggml_tensor * y  = ggml_mul_mat(gctx, Aq, x);

    if (Aq == NULL || x == NULL || y == NULL) {
'''
new = '''    ggml_tensor * Aq   = ggml_new_tensor_2d(gctx, GGML_TYPE_Q8_0, cols, rows);
    ggml_tensor * x    = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, cols);
    ggml_tensor * h    = ggml_mul_mat(gctx, Aq, x);
    ggml_tensor * bias = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, rows);
    ggml_tensor * y    = ggml_add(gctx, h, bias);

    if (Aq == NULL || x == NULL || h == NULL || bias == NULL || y == NULL) {
'''
s = replace_once(s, old, new, "tensor creation")

# Convert the G17C single-node graph assertion into a G18A two-node assertion.
# Use simple literal substitutions to avoid depending on exact C++ formatting.
if "graph->n_nodes != 1" not in s:
    raise RuntimeError("could not find graph->n_nodes != 1 in source template")
s = s.replace("graph->n_nodes != 1", "graph->n_nodes != 2", 1)
s = s.replace("expected 1 graph node", "expected 2 graph nodes", 1)


old = '''    const size_t off_Aq = 0;
    const size_t off_x  = 8192;
    const size_t off_y  = 12288;
    const size_t total_size = 16384;
'''
new = '''    const size_t off_Aq   = 0;
    const size_t off_x    = 8192;
    const size_t off_h    = 12288;
    const size_t off_bias = 16384;
    const size_t off_y    = 20480;
    const size_t total_size = 24576;
'''
s = replace_once(s, old, new, "offset block")

s = s.replace(
    'std::printf("backend-owned q8_0 mmv buffer size: %zu\\n", buffer->size);',
    'std::printf("backend-owned q8_0 add pipeline buffer size: %zu\\n", buffer->size);'
)

old = '''    force_2d_q8_0_data_layout(Aq, cols, rows, base_u8 + off_Aq);
    force_1d_f32_data_layout(x,  cols,       base_u8 + off_x);
    force_1d_f32_data_layout(y,  rows,       base_u8 + off_y);
'''
new = '''    force_2d_q8_0_data_layout(Aq, cols, rows, base_u8 + off_Aq);
    force_1d_f32_data_layout(x,    cols, base_u8 + off_x);
    force_1d_f32_data_layout(h,    rows, base_u8 + off_h);
    force_1d_f32_data_layout(bias, rows, base_u8 + off_bias);
    force_1d_f32_data_layout(y,    rows, base_u8 + off_y);
'''
s = replace_once(s, old, new, "force layouts")

old = '''    if (buffer->iface.init_tensor(buffer, Aq) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, x)  != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, y)  != GGML_STATUS_SUCCESS) {
'''
new = '''    if (buffer->iface.init_tensor(buffer, Aq)   != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, x)    != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, h)    != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, bias) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, y)    != GGML_STATUS_SUCCESS) {
'''
s = replace_once(s, old, new, "init tensors")

old = '''    if (!expect_resident("Aq", Aq, bytes_Aq, buffer, off_Aq) ||
        !expect_resident("x",  x,  bytes_x,  buffer, off_x)  ||
        !expect_resident("y",  y,  bytes_y,  buffer, off_y)) {
'''
new = '''    if (!expect_resident("Aq",   Aq,   bytes_Aq, buffer, off_Aq)   ||
        !expect_resident("x",    x,    bytes_x,  buffer, off_x)    ||
        !expect_resident("h",    h,    bytes_y,  buffer, off_h)    ||
        !expect_resident("bias", bias, bytes_y,  buffer, off_bias) ||
        !expect_resident("y",    y,    bytes_y,  buffer, off_y)) {
'''
s = replace_once(s, old, new, "residency checks")

old = '''    std::vector<float> A_f32;
    std::vector<ggml_cuda8_q8_0_block> Aq_host;
    std::vector<float> x_host(cols);
    std::vector<float> y_ref(rows, 0.0f);
    std::vector<float> y_out(rows, 0.0f);

    fill_f32_matrix(A_f32, rows, cols);
    pack_q8_0(A_f32, Aq_host, rows, cols);
    fill_x(x_host);
    cpu_ref_q8_mmv(Aq_host, x_host, y_ref, rows, cols);
'''
new = '''    std::vector<float> A_f32;
    std::vector<ggml_cuda8_q8_0_block> Aq_host;
    std::vector<float> x_host(cols);
    std::vector<float> h_ref(rows, 0.0f);
    std::vector<float> bias_host(rows, 0.0f);
    std::vector<float> y_ref(rows, 0.0f);
    std::vector<float> y_out(rows, 0.0f);

    fill_f32_matrix(A_f32, rows, cols);
    pack_q8_0(A_f32, Aq_host, rows, cols);
    fill_x(x_host);
    fill_bias(bias_host);
    cpu_ref_q8_mmv(Aq_host, x_host, h_ref, rows, cols);
    add_ref(h_ref, bias_host, y_ref);
'''
s = replace_once(s, old, new, "host reference setup")

old = '''    buffer->iface.set_tensor(buffer, Aq, &Aq_host[0], 0, bytes_Aq);
    buffer->iface.set_tensor(buffer, x,  &x_host[0],  0, bytes_x);
'''
new = '''    buffer->iface.set_tensor(buffer, Aq,   &Aq_host[0],   0, bytes_Aq);
    buffer->iface.set_tensor(buffer, x,    &x_host[0],    0, bytes_x);
    buffer->iface.set_tensor(buffer, bias, &bias_host[0], 0, bytes_y);
'''
s = replace_once(s, old, new, "set tensors")

s = s.replace(
    "real graph-builder packed Q8_0xF32_VEC",
    "real graph-builder packed Q8_0xF32_VEC->ADD_F32"
)

backup(SRC_OUT, "g18a")
write_file(SRC_OUT, s)
print("wrote", SRC_OUT)

cm = backup(CMAKE, "g18a")
if TARGET not in cm:
    if not cm.endswith("\n"):
        cm += "\n"
    cm += "\n" + CMAKE_BLOCK.strip() + "\n"
    write_file(CMAKE, cm)
    print("patched", CMAKE)
else:
    print("CMake target already present")

print("G18A writer complete.")
