#!/usr/bin/env python3
# G16A_fix_cmake_standalone_v3.py
#
# Fix G16A target to avoid linking full ggml target, which pulls
# ggml-backend-meta.cpp and fails in this CUDA8/Fermi legacy C++ mode.
# G16A only needs ggml.c graph-builder C API, so compile ../ggml.c directly.

import os
import time

ROOT = "/workspace/notebooks/llama.cpp-ph2"
CMAKE = os.path.join(ROOT, "ggml/src/ggml-cuda8/CMakeLists.txt")
TARGET = "ggml-cuda8-ggml-backend-graph-builder-add-smoke"

NEW_BLOCK = """
# ---------------------------------------------------------------------------
# G16A real GGML graph-builder ADD graph -> backend->iface.graph_compute smoke
# Build this smoke against ggml.c directly to avoid pulling ggml-backend-meta.cpp
# into the CUDA8/Fermi legacy C++ mode.
# ---------------------------------------------------------------------------

cuda_add_executable(ggml-cuda8-ggml-backend-graph-builder-add-smoke
    ../ggml.c
    ggml-cuda8-ggml-backend-graph-builder-add-smoke.cpp
)

target_include_directories(ggml-cuda8-ggml-backend-graph-builder-add-smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
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
backup = CMAKE + ".g16a-standalone-v3-backup-" + str(int(time.time()))
write_file(backup, cm)

needle = "cuda_add_executable(" + TARGET
start = cm.find(needle)
if start < 0:
    raise RuntimeError("could not find G16A cuda_add_executable target block")

# Include the preceding G16A section comment if present.
comment_marker = "# G16A real GGML graph-builder ADD graph -> backend->iface.graph_compute smoke"
comment_pos = cm.rfind(comment_marker, 0, start)
if comment_pos >= 0:
    sep_pos = cm.rfind("# ---------------------------------------------------------------------------", 0, comment_pos)
    replace_start = sep_pos if sep_pos >= 0 else comment_pos
else:
    replace_start = start

# End at next section separator after target block, or EOF.
next_section = cm.find("\n# ---------------------------------------------------------------------------", start + 1)
replace_end = len(cm) if next_section < 0 else next_section + 1

new_cm = cm[:replace_start].rstrip() + "\n\n" + NEW_BLOCK.strip() + "\n" + cm[replace_end:]
write_file(CMAKE, new_cm)

print("backup", backup)
print("patched", CMAKE)
print("G16A standalone CMake fix complete.")
