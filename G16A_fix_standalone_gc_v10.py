#!/usr/bin/env python3
# G16A_fix_standalone_gc_v10.py
#
# G16A v10 fix:
#   Linking the full ggml target drags in gguf.cpp, which needs newer C++ support
#   than this legacy CUDA8/Fermi build environment is effectively using.
#
#   The earlier standalone attempt got close but pulled ggml-backend.cpp, which
#   started another dependency chain.  For this smoke we only need:
#     - ggml_init / ggml_new_tensor_1d / ggml_add / ggml_new_graph / ggml_build_forward_expand
#     - ggml threading helpers used by ggml_init
#     - quant support referenced by ggml.c type trait tables
#
#   v10 builds a standalone G16A target from ggml.c + ggml-quants.c +
#   ggml-threading.cpp, and uses section GC so unused ggml.c routines that refer
#   to backend/allocator helpers are discarded instead of requiring the full ggml
#   backend source stack.

import os
import time

ROOT = "/workspace/notebooks/llama.cpp-ph2"
CMAKE = os.path.join(ROOT, "ggml/src/ggml-cuda8/CMakeLists.txt")
TARGET = "ggml-cuda8-ggml-backend-graph-builder-add-smoke"

G16A_CMAKE_BLOCK = """
# ---------------------------------------------------------------------------
# G16A real GGML graph-builder ADD graph -> backend->iface.graph_compute smoke
#
# Standalone graph-builder subset for the CUDA8/Fermi legacy build:
#   - avoids full ggml target so gguf.cpp / newer C++ sources are not built
#   - avoids ggml-backend.cpp dependency chain
#   - uses function/data sections + linker GC so unused ggml.c backend helpers
#     do not require unresolved backend/allocator symbols
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-backend-graph-builder-add-smoke
    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    ggml-cuda8-ggml-backend-graph-builder-add-smoke.cpp
)

target_include_directories(ggml-cuda8-ggml-backend-graph-builder-add-smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_compile_definitions(ggml-cuda8-ggml-backend-graph-builder-add-smoke PRIVATE
    GGML_VERSION=\"${GGML_VERSION}\"
    GGML_COMMIT=\"${GGML_COMMIT}\"
)

target_compile_options(ggml-cuda8-ggml-backend-graph-builder-add-smoke PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:C>:-fdata-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections>
    $<$<COMPILE_LANGUAGE:CXX>:-fdata-sections>
)

set_property(TARGET ggml-cuda8-ggml-backend-graph-builder-add-smoke APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--gc-sections")

target_link_libraries(ggml-cuda8-ggml-backend-graph-builder-add-smoke
    ggml-cuda8-kernels
    ${CUDA_LIBRARIES}
)
"""

def read_file(path):
    with open(path, "r") as f:
        return f.read()

def write_file(path, data):
    with open(path, "w") as f:
        f.write(data)

cm = read_file(CMAKE)
backup = CMAKE + ".g16a-standalone-gc-v10-backup-" + str(int(time.time()))
write_file(backup, cm)

needle = "cuda_add_executable(" + TARGET
start = cm.find(needle)
if start < 0:
    raise RuntimeError("could not find G16A cuda_add_executable target block")

comment_marker = "# G16A real GGML graph-builder ADD graph -> backend->iface.graph_compute smoke"
comment_pos = cm.rfind(comment_marker, 0, start)
if comment_pos >= 0:
    sep_pos = cm.rfind("# ---------------------------------------------------------------------------", 0, comment_pos)
    replace_start = sep_pos if sep_pos >= 0 else comment_pos
else:
    replace_start = start

next_section = cm.find("\n# ---------------------------------------------------------------------------", start + 1)
replace_end = len(cm) if next_section < 0 else next_section + 1

new_cm = cm[:replace_start].rstrip() + "\n\n" + G16A_CMAKE_BLOCK.strip() + "\n" + cm[replace_end:]
write_file(CMAKE, new_cm)

print("backup", backup)
print("patched", CMAKE)
print("G16A v10 fix complete: standalone ggml graph-builder subset + section GC.")
