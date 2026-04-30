#!/usr/bin/env python3
# G16A_fix_cmake_link_ggml_cxx17_v6.py
#
# G16A v6 fix:
#   The standalone-minimal-source approach avoids ggml-backend-meta.cpp, but ggml.c
#   has too many required transitive symbols.  The cleaner fix is to link the real
#   ggml target again, but force ggml-base C++ sources to build as C++17 so
#   ggml-backend-meta.cpp structured bindings compile.

import os
import time

ROOT = "/workspace/notebooks/llama.cpp-ph2"
CMAKE = os.path.join(ROOT, "ggml/src/ggml-cuda8/CMakeLists.txt")
TARGET = "ggml-cuda8-ggml-backend-graph-builder-add-smoke"

NEW_BLOCK = """
# ---------------------------------------------------------------------------
# G16A real GGML graph-builder ADD graph -> backend->iface.graph_compute smoke
#
# Link the real ggml target so the graph-builder API gets the correct full GGML
# transitive source set.  This checkout's ggml-backend-meta.cpp uses C++17 syntax
# such as structured bindings, so force ggml-base C++ compilation to C++17 for
# this CUDA8/Fermi build tree.
# ---------------------------------------------------------------------------

if (TARGET ggml-base)
    set_property(TARGET ggml-base PROPERTY CXX_STANDARD 17)
    set_property(TARGET ggml-base PROPERTY CXX_STANDARD_REQUIRED ON)
endif()

cuda_add_executable(ggml-cuda8-ggml-backend-graph-builder-add-smoke
    ggml-cuda8-ggml-backend-graph-builder-add-smoke.cpp
)

target_link_libraries(ggml-cuda8-ggml-backend-graph-builder-add-smoke
    ggml-cuda8-kernels
    ggml
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
backup = CMAKE + ".g16a-link-ggml-cxx17-v6-backup-" + str(int(time.time()))
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
print("G16A CMake v6 fix complete: link ggml target and force ggml-base CXX_STANDARD=17.")
