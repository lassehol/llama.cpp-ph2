#!/usr/bin/env python3
# G16A_fix_meta_cpp11_v7.py
#
# G16A v7 fix:
#   The real ggml target is the correct link dependency for real GGML graph-builder
#   API use.  The blocking issue is ggml-backend-meta.cpp using C++17-only syntax
#   while this CUDA8/Fermi build compiles ggml-base in an older C++ mode.
#
# This patch makes ggml-backend-meta.cpp compatible with the current legacy C++
# mode by replacing:
#   for (auto & [ctx, buf] : map)
# with:
#   for (auto & ctx_buf : map) { auto & ctx = ctx_buf.first; auto & buf = ctx_buf.second; ... }
# and replacing CTAD:
#   const std::pair key = ...
# with an explicit std::pair<const ggml_tensor *, bool>.
#
# It also ensures the G16A CMake target links the real ggml target instead of
# trying to compile a partial standalone subset of GGML sources.

import os
import time

ROOT = "/workspace/notebooks/llama.cpp-ph2"
META = os.path.join(ROOT, "ggml/src/ggml-backend-meta.cpp")
CMAKE = os.path.join(ROOT, "ggml/src/ggml-cuda8/CMakeLists.txt")
TARGET = "ggml-cuda8-ggml-backend-graph-builder-add-smoke"

G16A_CMAKE_BLOCK = """
# ---------------------------------------------------------------------------
# G16A real GGML graph-builder ADD graph -> backend->iface.graph_compute smoke
#
# Link the real ggml target so the graph-builder API gets the correct full GGML
# transitive source set.  ggml-backend-meta.cpp is patched to avoid C++17-only
# syntax in this CUDA8/Fermi legacy build mode.
# ---------------------------------------------------------------------------

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

def backup(path, tag):
    data = read_file(path)
    b = path + "." + tag + "-backup-" + str(int(time.time()))
    write_file(b, data)
    print("backup", b)
    return data

# 1) Patch ggml-backend-meta.cpp C++17 constructs to C++11-compatible code.
meta = backup(META, "g16a-meta-cpp11-v7")
orig_meta = meta

old_loop = "for (auto & [ctx, buf] : buf_ctx->buf_configs) {"
new_loop = "for (auto & ctx_buf : buf_ctx->buf_configs) {\n        auto & ctx = ctx_buf.first;\n        auto & buf = ctx_buf.second;"
if old_loop in meta:
    meta = meta.replace(old_loop, new_loop, 1)
    print("patched structured binding loop in ggml-backend-meta.cpp")
elif "for (auto & ctx_buf : buf_ctx->buf_configs)" in meta:
    print("structured binding loop already patched")
else:
    raise RuntimeError("could not find structured binding loop in ggml-backend-meta.cpp")

old_pair = "const std::pair key = std::make_pair(tensor, assume_sync);"
new_pair = "const std::pair<const ggml_tensor *, bool> key = std::make_pair(tensor, assume_sync);"
if old_pair in meta:
    meta = meta.replace(old_pair, new_pair, 1)
    print("patched std::pair CTAD in ggml-backend-meta.cpp")
elif new_pair in meta:
    print("std::pair CTAD already patched")
else:
    raise RuntimeError("could not find std::pair CTAD line in ggml-backend-meta.cpp")

if meta != orig_meta:
    write_file(META, meta)
    print("patched", META)
else:
    print("no ggml-backend-meta.cpp changes needed")

# 2) Ensure G16A target links real ggml target.
cm = backup(CMAKE, "g16a-link-ggml-v7")
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
print("patched", CMAKE)

print("G16A v7 fix complete: ggml-backend-meta.cpp C++11 compatibility + real ggml target link.")
