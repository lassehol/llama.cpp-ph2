#!/usr/bin/env python3
# G21A_writer.py
#
# Create G21A real graph-builder residual-branch quantized pipeline smoke:
#
#   h = mul_mat(Q8_0, x)
#   y = h + residual
#
# This derives from the validated G18A Q8_0 -> ADD smoke, but renames the
# second F32 input as an explicit residual branch and adds an input-isolation
# check after graph_compute.
#
# Python 3.5-compatible: no f-strings.

import os
import time

ROOT = "/workspace/notebooks/llama.cpp-ph2"
BASE = os.path.join(ROOT, "ggml/src/ggml-cuda8")

SRC_IN = os.path.join(BASE, "ggml-cuda8-ggml-backend-graph-builder-q8_0-add-smoke.cpp")
SRC_OUT = os.path.join(BASE, "ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-smoke.cpp")
CMAKE = os.path.join(BASE, "CMakeLists.txt")

TARGET = "ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-smoke"

OLD_NAME = "ggml-cuda8-ggml-backend-graph-builder-q8_0-add-smoke"
NEW_NAME = "ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-smoke"

CMAKE_BLOCK = r'''
# ---------------------------------------------------------------------------
# G21A real GGML graph-builder Q8_0 MUL_MAT -> residual ADD smoke
# Uses standalone-GC GGML graph-builder subset.
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-smoke
    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-smoke.cpp
)

target_include_directories(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_compile_definitions(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-smoke PRIVATE
    GGML_VERSION="${GGML_VERSION}"
    GGML_COMMIT="${GGML_COMMIT}"
)

target_compile_options(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-smoke PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:C>:-fdata-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-fdata-sections>
)

set_property(TARGET ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-smoke APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--gc-sections")

target_link_libraries(ggml-cuda8-ggml-backend-graph-builder-q8_0-residual-add-smoke
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
    "G18A: real GGML graph-builder Q8_0 MUL_MAT -> ADD pipeline through CUDA8 graph_compute.",
    "G21A: real GGML graph-builder Q8_0 MUL_MAT -> residual ADD branch through CUDA8 graph_compute."
)

# Convert the second F32 input from a generic bias vector into an explicit
# residual branch. G18A has no "biased" variable, so this token-level rewrite is
# safe for the current source.
s = s.replace("fill_bias", "fill_residual")
s = s.replace("bias", "residual")
s = s.replace("Bias", "Residual")
s = s.replace("BIAS", "RESIDUAL")

s = s.replace(
    "backend-owned q8_0 add pipeline buffer size",
    "backend-owned q8_0 residual-add pipeline buffer size"
)

s = s.replace(
    "real graph-builder packed Q8_0xF32_VEC->ADD_F32",
    "real graph-builder packed Q8_0xF32_VEC->RESIDUAL_ADD_F32"
)

# Add a residual input isolation buffer in host-side verification state.
old = '''    std::vector<float> h_ref(rows, 0.0f);
    std::vector<float> residual_host(rows, 0.0f);
    std::vector<float> y_ref(rows, 0.0f);
    std::vector<float> y_out(rows, 0.0f);
'''
new = '''    std::vector<float> h_ref(rows, 0.0f);
    std::vector<float> residual_host(rows, 0.0f);
    std::vector<float> residual_after(rows, 0.0f);
    std::vector<float> y_ref(rows, 0.0f);
    std::vector<float> y_out(rows, 0.0f);
'''
s = replace_once(s, old, new, "host residual_after vector")

# Read the residual branch back after graph_compute. The residual input should
# remain unchanged by the ADD dispatch.
old = '''    buffer->iface.get_tensor(buffer, y, &y_out[0], 0, bytes_y);
'''
new = '''    buffer->iface.get_tensor(buffer, y, &y_out[0], 0, bytes_y);
    buffer->iface.get_tensor(buffer, residual, &residual_after[0], 0, bytes_y);
'''
s = replace_once(s, old, new, "read residual_after")

# Insert residual input isolation verification after the output verification.
old = '''    std::printf("real graph-builder packed Q8_0xF32_VEC->RESIDUAL_ADD_F32 verification PASS\\n");

    buffer->iface.free_buffer(buffer);
'''
new = '''    std::printf("real graph-builder packed Q8_0xF32_VEC->RESIDUAL_ADD_F32 verification PASS\\n");

    double residual_max_abs_err = 0.0;

    for (size_t i = 0; i < residual_host.size(); ++i) {
        const double abs_err = std::fabs((double) residual_after[i] - (double) residual_host[i]);
        if (abs_err > residual_max_abs_err) {
            residual_max_abs_err = abs_err;
        }
    }

    std::printf("real graph-builder residual branch isolation max_abs_err=%.9g\\n",
        residual_max_abs_err);

    if (residual_max_abs_err != 0.0) {
        std::fprintf(stderr, "real graph-builder residual branch isolation FAIL\\n");
        buffer->iface.free_buffer(buffer);
        backend->iface.free(backend);
        ggml_free(gctx);
        return 1;
    }

    std::printf("real graph-builder residual branch isolation PASS\\n");

    buffer->iface.free_buffer(buffer);
'''
s = replace_once(s, old, new, "residual isolation check")

backup(SRC_OUT, "g21a")
write_file(SRC_OUT, s)
print("wrote", SRC_OUT)

cm = backup(CMAKE, "g21a")
if TARGET not in cm:
    if not cm.endswith("\n"):
        cm += "\n"
    cm += "\n" + CMAKE_BLOCK.strip() + "\n"
    write_file(CMAKE, cm)
    print("patched", CMAKE)
else:
    print("CMake target already present")

print("G21A writer complete.")
