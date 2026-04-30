#!/usr/bin/env python3
# G22A_writer.py
#
# Create G22A real graph-builder residual-softmax quantized pipeline smoke:
#
#   h      = mul_mat(Q8_0, x)
#   biased = h + residual
#   y      = softmax(biased)
#
# This derives from the validated G21A Q8_0 -> residual ADD smoke and preserves
# the residual input isolation check after graph_compute.
#
# Python 3.5-compatible: no f-strings.

import os
import re
import time

ROOT = "/workspace/notebooks/llama.cpp-ph2"
BASE = os.path.join(ROOT, "ggml/src/ggml-cuda8")

SRC_IN = os.path.join(BASE, "ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-smoke.cpp")
SRC_OUT = os.path.join(BASE, "ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-smoke.cpp")
CMAKE = os.path.join(BASE, "CMakeLists.txt")

TARGET = "ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-smoke"

OLD_NAME = "ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-smoke"
NEW_NAME = "ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-smoke"

CMAKE_BLOCK = r'''
# ---------------------------------------------------------------------------
# G22A real GGML graph-builder Q8_0 MUL_MAT -> residual ADD -> SOFTMAX smoke
# Uses standalone-GC GGML graph-builder subset.
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-smoke
    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-smoke.cpp
)

target_include_directories(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_compile_definitions(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-smoke PRIVATE
    GGML_VERSION="${GGML_VERSION}"
    GGML_COMMIT="${GGML_COMMIT}"
)

target_compile_options(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-smoke PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:C>:-fdata-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-fdata-sections>
)

set_property(TARGET ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-smoke APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--gc-sections")

target_link_libraries(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-softmax-smoke
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
    "G21A: real GGML graph-builder Q8_0 MUL_MAT -> residual ADD branch through CUDA8 graph_compute.",
    "G22A: real GGML graph-builder Q8_0 MUL_MAT -> residual ADD -> SOFTMAX branch through CUDA8 graph_compute."
)

helper_insert = r'''
static void softmax_ref(
    const std::vector<float> & src,
    std::vector<float> & dst
) {
    float max_v = src[0];

    for (size_t i = 1; i < src.size(); ++i) {
        if (src[i] > max_v) {
            max_v = src[i];
        }
    }

    double sum = 0.0;

    for (size_t i = 0; i < src.size(); ++i) {
        const double e = std::exp((double) src[i] - (double) max_v);
        dst[i] = (float) e;
        sum += e;
    }

    for (size_t i = 0; i < dst.size(); ++i) {
        dst[i] = (float) ((double) dst[i] / sum);
    }
}

'''

if "static void softmax_ref(" not in s:
    anchor = "static void cpu_ref_q8_mmv("
    idx = s.find(anchor)
    if idx < 0:
        raise RuntimeError("could not find cpu_ref_q8_mmv anchor")
    s = s[:idx] + helper_insert + s[idx:]

# Convert:
#   y = ggml_add(h, residual)
# into:
#   biased = ggml_add(h, residual)
#   y      = ggml_soft_max(biased)
#
# Use regex because the generated G21A source may have different alignment.
tensor_pat = (
    r'(    ggml_tensor \* Aq\s*=\s*ggml_new_tensor_2d\(gctx, GGML_TYPE_Q8_0, cols, rows\);\n'
    r'    ggml_tensor \* x\s*=\s*ggml_new_tensor_1d\(gctx, GGML_TYPE_F32, cols\);\n'
    r'    ggml_tensor \* h\s*=\s*ggml_mul_mat\(gctx, Aq, x\);\n'
    r'    ggml_tensor \* residual\s*=\s*ggml_new_tensor_1d\(gctx, GGML_TYPE_F32, rows\);\n)'
    r'    ggml_tensor \* y\s*=\s*ggml_add\(gctx, h, residual\);\n\n'
    r'    if \(Aq == NULL \|\| x == NULL \|\| h == NULL \|\| residual == NULL \|\| y == NULL\) \{'
)

tensor_repl = (
    r'\1'
    '    ggml_tensor * biased   = ggml_add(gctx, h, residual);\n'
    '    ggml_tensor * y        = ggml_soft_max(gctx, biased);\n\n'
    '    if (Aq == NULL || x == NULL || h == NULL || residual == NULL || biased == NULL || y == NULL) {'
)

s_new, n = re.subn(tensor_pat, tensor_repl, s, count=1)
if n != 1:
    raise RuntimeError("could not patch tensor creation regex block")

s = s_new


s = s.replace("graph->n_nodes != 2", "graph->n_nodes != 3", 1)
s = s.replace("expected 2 graph nodes", "expected 3 graph nodes", 1)

# Add an intermediate biased tensor and move y after it.
offset_pat = (
    r'    const size_t off_Aq\s*=\s*0;\n'
    r'    const size_t off_x\s*=\s*8192;\n'
    r'    const size_t off_h\s*=\s*12288;\n'
    r'    const size_t off_residual\s*=\s*16384;\n'
    r'    const size_t off_y\s*=\s*20480;\n'
    r'    const size_t total_size\s*=\s*24576;\n'
)

offset_repl = (
    '    const size_t off_Aq       = 0;\n'
    '    const size_t off_x        = 8192;\n'
    '    const size_t off_h        = 12288;\n'
    '    const size_t off_residual = 16384;\n'
    '    const size_t off_biased   = 20480;\n'
    '    const size_t off_y        = 24576;\n'
    '    const size_t total_size = 28672;\n'
)

s_new, n = re.subn(offset_pat, offset_repl, s, count=1)
if n != 1:
    raise RuntimeError("could not patch offset block with regex")
s = s_new


s = s.replace(
    'std::printf("backend-owned q8_0 residual-add pipeline buffer size: %zu\\n", buffer->size);',
    'std::printf("backend-owned q8_0 residual-add-softmax pipeline buffer size: %zu\\n", buffer->size);'
)

# Add explicit layout for the biased intermediate tensor.
force_pat = (
    r'    force_2d_q8_0_data_layout\(Aq, cols, rows, base_u8 \+ off_Aq\);\n'
    r'    force_1d_f32_data_layout\(x,\s*cols, base_u8 \+ off_x\);\n'
    r'    force_1d_f32_data_layout\(h,\s*rows, base_u8 \+ off_h\);\n'
    r'    force_1d_f32_data_layout\(residual,\s*rows, base_u8 \+ off_residual\);\n'
    r'    force_1d_f32_data_layout\(y,\s*rows, base_u8 \+ off_y\);\n'
)

force_repl = (
    '    force_2d_q8_0_data_layout(Aq, cols, rows, base_u8 + off_Aq);\n'
    '    force_1d_f32_data_layout(x,        cols, base_u8 + off_x);\n'
    '    force_1d_f32_data_layout(h,        rows, base_u8 + off_h);\n'
    '    force_1d_f32_data_layout(residual, rows, base_u8 + off_residual);\n'
    '    force_1d_f32_data_layout(biased,   rows, base_u8 + off_biased);\n'
    '    force_1d_f32_data_layout(y,        rows, base_u8 + off_y);\n'
)

s_new, n = re.subn(force_pat, force_repl, s, count=1)
if n != 1:
    raise RuntimeError("could not patch force layouts with regex")
s = s_new


# Initialize the biased intermediate tensor in the CUDA8 buffer.
init_pat = (
    r'    if \(buffer->iface.init_tensor\(buffer, Aq\)\s*!= GGML_STATUS_SUCCESS \|\|\n'
    r'        buffer->iface.init_tensor\(buffer, x\)\s*!= GGML_STATUS_SUCCESS \|\|\n'
    r'        buffer->iface.init_tensor\(buffer, h\)\s*!= GGML_STATUS_SUCCESS \|\|\n'
    r'        buffer->iface.init_tensor\(buffer, residual\)\s*!= GGML_STATUS_SUCCESS \|\|\n'
    r'        buffer->iface.init_tensor\(buffer, y\)\s*!= GGML_STATUS_SUCCESS\) \{'
)

init_repl = (
    '    if (buffer->iface.init_tensor(buffer, Aq)       != GGML_STATUS_SUCCESS ||\n'
    '        buffer->iface.init_tensor(buffer, x)        != GGML_STATUS_SUCCESS ||\n'
    '        buffer->iface.init_tensor(buffer, h)        != GGML_STATUS_SUCCESS ||\n'
    '        buffer->iface.init_tensor(buffer, residual) != GGML_STATUS_SUCCESS ||\n'
    '        buffer->iface.init_tensor(buffer, biased)   != GGML_STATUS_SUCCESS ||\n'
    '        buffer->iface.init_tensor(buffer, y)        != GGML_STATUS_SUCCESS) {'
)

s_new, n = re.subn(init_pat, init_repl, s, count=1)
if n != 1:
    raise RuntimeError("could not patch init tensors with regex")
s = s_new


# Check residency of the biased intermediate tensor.
residency_pat = (
    r'    if \(!expect_resident\("Aq",\s*Aq,\s*bytes_Aq, buffer, off_Aq\)\s*\|\|\n'
    r'        !expect_resident\("x",\s*x,\s*bytes_x,\s*buffer, off_x\)\s*\|\|\n'
    r'        !expect_resident\("h",\s*h,\s*bytes_y,\s*buffer, off_h\)\s*\|\|\n'
    r'        !expect_resident\("residual",\s*residual,\s*bytes_y,\s*buffer, off_residual\)\s*\|\|\n'
    r'        !expect_resident\("y",\s*y,\s*bytes_y,\s*buffer, off_y\)\) \{'
)

residency_repl = (
    '    if (!expect_resident("Aq",       Aq,       bytes_Aq, buffer, off_Aq)       ||\n'
    '        !expect_resident("x",        x,        bytes_x,  buffer, off_x)        ||\n'
    '        !expect_resident("h",        h,        bytes_y,  buffer, off_h)        ||\n'
    '        !expect_resident("residual", residual, bytes_y,  buffer, off_residual) ||\n'
    '        !expect_resident("biased",   biased,   bytes_y,  buffer, off_biased)   ||\n'
    '        !expect_resident("y",        y,        bytes_y,  buffer, off_y)) {'
)

s_new, n = re.subn(residency_pat, residency_repl, s, count=1)
if n != 1:
    raise RuntimeError("could not patch residency checks with regex")
s = s_new


# Change CPU reference from add-only output to add followed by softmax.
host_ref_pat = (
    r'    std::vector<float> h_ref\(rows, 0\.0f\);\n'
    r'    std::vector<float> residual_host\(rows, 0\.0f\);\n'
    r'    std::vector<float> residual_after\(rows, 0\.0f\);\n'
    r'    std::vector<float> y_ref\(rows, 0\.0f\);\n'
    r'    std::vector<float> y_out\(rows, 0\.0f\);\n\n'
    r'    fill_f32_matrix\(A_f32, rows, cols\);\n'
    r'    pack_q8_0\(A_f32, Aq_host, rows, cols\);\n'
    r'    fill_x\(x_host\);\n'
    r'    fill_residual\(residual_host\);\n'
    r'    cpu_ref_q8_mmv\(Aq_host, x_host, h_ref, rows, cols\);\n'
    r'    add_ref\(h_ref, residual_host, y_ref\);\n'
)

host_ref_repl = (
    '    std::vector<float> h_ref(rows, 0.0f);\n'
    '    std::vector<float> residual_host(rows, 0.0f);\n'
    '    std::vector<float> residual_after(rows, 0.0f);\n'
    '    std::vector<float> biased_ref(rows, 0.0f);\n'
    '    std::vector<float> y_ref(rows, 0.0f);\n'
    '    std::vector<float> y_out(rows, 0.0f);\n\n'
    '    fill_f32_matrix(A_f32, rows, cols);\n'
    '    pack_q8_0(A_f32, Aq_host, rows, cols);\n'
    '    fill_x(x_host);\n'
    '    fill_residual(residual_host);\n'
    '    cpu_ref_q8_mmv(Aq_host, x_host, h_ref, rows, cols);\n'
    '    add_ref(h_ref, residual_host, biased_ref);\n'
    '    softmax_ref(biased_ref, y_ref);\n'
)

s_new, n = re.subn(host_ref_pat, host_ref_repl, s, count=1)
if n != 1:
    raise RuntimeError("could not patch host reference setup with regex")
s = s_new


s = s.replace(
    "real graph-builder packed Q8_0xF32_VEC->RESIDUAL_ADD_F32",
    "real graph-builder packed Q8_0xF32_VEC->RESIDUAL_ADD_F32->SOFTMAX_ROWS_F32"
)

backup(SRC_OUT, "g22a")
write_file(SRC_OUT, s)
print("wrote", SRC_OUT)

cm = backup(CMAKE, "g22a")
if TARGET not in cm:
    if not cm.endswith("\n"):
        cm += "\n"
    cm += "\n" + CMAKE_BLOCK.strip() + "\n"
    write_file(CMAKE, cm)
    print("patched", CMAKE)
else:
    print("CMake target already present")

print("G22A writer complete.")
