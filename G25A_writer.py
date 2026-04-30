#!/usr/bin/env python3
# G25A_writer.py
#
# Create G25A smoke:
#
#   Q8_0 MUL_MAT -> MUL_SCALAR -> residual ADD -> SOFTMAX -> SUM_ROWS
#
# Same validated graph as G24A, but additionally verifies that both:
#   - scale scalar input branch
#   - residual vector input branch
# remain unchanged after ggml_backend_i.graph_compute.
#
# Python 3.5-compatible.

import os
import re
import time

ROOT = "/workspace/notebooks/llama.cpp-ph2"
BASE = os.path.join(ROOT, "ggml/src/ggml-cuda8")

SRC_IN = os.path.join(BASE, "ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-smoke.cpp")
SRC_OUT = os.path.join(BASE, "ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-isolation-smoke.cpp")
CMAKE = os.path.join(BASE, "CMakeLists.txt")

TARGET = "ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-isolation-smoke"

OLD_NAME = "ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-smoke"
NEW_NAME = "ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-isolation-smoke"

CMAKE_BLOCK = r'''
# ---------------------------------------------------------------------------
# G25A real GGML graph-builder Q8_0 MUL_MAT -> MUL_SCALAR -> residual ADD -> SOFTMAX -> SUM_ROWS
# with explicit scale and residual input branch isolation checks.
# Uses standalone-GC GGML graph-builder subset.
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-isolation-smoke
    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-isolation-smoke.cpp
)

target_include_directories(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-isolation-smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_compile_definitions(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-isolation-smoke PRIVATE
    GGML_VERSION="${GGML_VERSION}"
    GGML_COMMIT="${GGML_COMMIT}"
)

target_compile_options(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-isolation-smoke PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:C>:-fdata-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-fdata-sections>
)

set_property(TARGET ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-isolation-smoke APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--gc-sections")

target_link_libraries(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-scale-add-softmax-sumrows-isolation-smoke
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


if not os.path.exists(SRC_IN):
    raise RuntimeError("missing G24A source: " + SRC_IN)

s = read_file(SRC_IN)
s = s.replace(OLD_NAME, NEW_NAME)

s = s.replace(
    "G24A: real GGML graph-builder Q8_0 MUL_MAT -> MUL_SCALAR -> residual ADD -> SOFTMAX -> SUM_ROWS branch through CUDA8 graph_compute.",
    "G25A: real GGML graph-builder Q8_0 MUL_MAT -> MUL_SCALAR -> residual ADD -> SOFTMAX -> SUM_ROWS with scale and residual input isolation through CUDA8 graph_compute."
)

s = s.replace(
    "backend-owned q8_0 residual-scale-add-softmax-sumrows pipeline buffer size",
    "backend-owned q8_0 residual-scale-add-softmax-sumrows-isolation pipeline buffer size"
)

s = s.replace(
    "real graph-builder packed Q8_0xF32_VEC->MUL_SCALAR_F32->RESIDUAL_ADD_F32->SOFTMAX_ROWS_F32->REDUCE_SUM_ROWS_F32",
    "real graph-builder packed Q8_0xF32_VEC->MUL_SCALAR_F32->RESIDUAL_ADD_F32->SOFTMAX_ROWS_F32->REDUCE_SUM_ROWS_F32+INPUT_ISOLATION"
)

# Add scale_after host storage next to scale_host.
s = sub_once(
    s,
    r'    std::vector<float> scale_host\(1\);\n\n'
    r'    scale_host\[0\] = 0\.75f;\n',
    '    std::vector<float> scale_host(1);\n'
    '    std::vector<float> scale_after(1, 0.0f);\n\n'
    '    scale_host[0] = 0.75f;\n',
    "scale_after vector"
)

# Read scale back after graph_compute, in addition to y and residual.
s = sub_once(
    s,
    r'    buffer->iface.get_tensor\(buffer, y, &y_out\[0\], 0, sizeof\(float\)\);\n'
    r'    buffer->iface.get_tensor\(buffer, residual, &residual_after\[0\], 0, bytes_y\);\n',
    '    buffer->iface.get_tensor(buffer, y, &y_out[0], 0, sizeof(float));\n'
    '    buffer->iface.get_tensor(buffer, scale, &scale_after[0], 0, sizeof(float));\n'
    '    buffer->iface.get_tensor(buffer, residual, &residual_after[0], 0, bytes_y);\n',
    "read scale_after"
)

# Insert explicit scale isolation verification before residual isolation block.
scale_check = r'''
    const double scale_abs_err = std::fabs((double) scale_after[0] - (double) scale_host[0]);

    std::printf("real graph-builder scale branch isolation max_abs_err=%.9g\n",
        scale_abs_err);

    if (scale_abs_err != 0.0) {
        std::fprintf(stderr, "real graph-builder scale branch isolation FAIL\n");
        buffer->iface.free_buffer(buffer);
        backend->iface.free(backend);
        ggml_free(gctx);
        return 1;
    }

    std::printf("real graph-builder scale branch isolation PASS\n");

'''

s = sub_once(
    s,
    r'    double residual_max_abs_err = 0\.0;\n',
    scale_check + '    double residual_max_abs_err = 0.0;\n',
    "insert scale isolation"
)

backup(SRC_OUT, "g25a")
write_file(SRC_OUT, s)
print("wrote", SRC_OUT)

cm = backup(CMAKE, "g25a")
if TARGET not in cm:
    if not cm.endswith("\n"):
        cm += "\n"
    cm += "\n" + CMAKE_BLOCK.strip() + "\n"
    write_file(CMAKE, cm)
    print("patched", CMAKE)
else:
    print("CMake target already present")

print("G25A writer complete.")
