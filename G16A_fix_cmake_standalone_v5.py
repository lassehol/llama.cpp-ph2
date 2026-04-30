#!/usr/bin/env python3
# G16A_fix_cmake_standalone_v5.py
#
# Fixes G16A standalone GGML graph-builder target after direct ../ggml.c linkage
# exposed missing transitive GGML symbols.
#
# v5 keeps avoiding the full ggml target because that target pulls
# ggml-backend-meta.cpp, which fails in this CUDA8/Fermi legacy C++ mode.
# Instead, v5 compiles only the minimal source set needed by ggml.c:
#   ../ggml.c
#   ../ggml-quants.c
#   ../ggml-threading.cpp
#   ../ggml-backend.cpp
#
# It also supplies GGML_VERSION/GGML_COMMIT compile definitions normally supplied
# by the upstream ggml target.

import os
import time

ROOT = "/workspace/notebooks/llama.cpp-ph2"
CMAKE = os.path.join(ROOT, "ggml/src/ggml-cuda8/CMakeLists.txt")
TARGET = "ggml-cuda8-ggml-backend-graph-builder-add-smoke"

NEW_BLOCK = """
# ---------------------------------------------------------------------------
# G16A real GGML graph-builder ADD graph -> backend->iface.graph_compute smoke
#
# This smoke intentionally avoids linking the full ggml target because that
# target pulls ggml-backend-meta.cpp, which currently requires a newer C++ mode
# than this CUDA8/Fermi build uses.  Instead, compile the minimal GGML source
# subset needed for the C graph-builder API.
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-backend-graph-builder-add-smoke
    ../ggml.c
    ../ggml-quants.c
    ../ggml-threading.cpp
    ../ggml-backend.cpp
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
backup = CMAKE + ".g16a-standalone-v5-backup-" + str(int(time.time()))
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

new_cm = cm[:replace_start].rstrip() + "\n\n" + NEW_BLOCK.strip() + "\n" + cm[replace_end:]
write_file(CMAKE, new_cm)

print("backup", backup)
print("patched", CMAKE)
print("G16A standalone CMake v5 fix complete: minimal GGML sources + version/commit definitions.")
