#!/usr/bin/env python3
# G16A_fix_meta_cpp11_v8.py
#
# v7 correctly identified that ggml-backend-meta.cpp must avoid C++17-only
# structured bindings / CTAD in this legacy CUDA8 build.  However, the first
# C++11 rewrite assumed buf_ctx->buf_configs stores std::pair-like objects.
# The build showed buf_ctx->buf_configs stores a custom aggregate
# buffer_config, so .first/.second are wrong.
#
# v8 parses struct buffer_config, extracts its first two field names, and rewrites
# the loop aliases to use those real field names. It also keeps the explicit
# std::pair<const ggml_tensor *, bool> CTAD fix and keeps G16A linked to ggml.

import os
import re
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

def get_buffer_config_fields(text):
    m = re.search(r"struct\s+buffer_config\s*\{(?P<body>.*?)\n\s*\};", text, re.S)
    if not m:
        raise RuntimeError("could not find struct buffer_config body")
    body = m.group("body")
    fields = []
    for raw in body.splitlines():
        line = raw.split("//", 1)[0].strip()
        if not line or line.startswith("static ") or line.startswith("using "):
            continue
        if not line.endswith(";"):
            continue
        # Drop initializer and trailing semicolon.
        line = line[:-1].split("=", 1)[0].strip()
        # Last identifier before optional array suffix is the field name.
        mfield = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?$", line)
        if mfield:
            fields.append(mfield.group(1))
    if len(fields) < 2:
        raise RuntimeError("could not infer first two fields of struct buffer_config; fields=%r" % (fields,))
    return fields[0], fields[1]

# 1) Patch ggml-backend-meta.cpp.
meta = backup(META, "g16a-meta-cpp11-v8")
orig_meta = meta
field0, field1 = get_buffer_config_fields(meta)
print("detected buffer_config fields:", field0, field1)

old_loop = "for (auto & [ctx, buf] : buf_ctx->buf_configs) {"
new_loop = "for (auto & ctx_buf : buf_ctx->buf_configs) {\n        auto & ctx = ctx_buf.%s;\n        auto & buf = ctx_buf.%s;" % (field0, field1)
if old_loop in meta:
    meta = meta.replace(old_loop, new_loop, 1)
    print("patched original structured binding loop")
else:
    # Repair v7's incorrect .first/.second rewrite or an already partially patched variant.
    meta2 = re.sub(
        r"for \(auto & ctx_buf : buf_ctx->buf_configs\) \{\n\s*auto & ctx = ctx_buf\.[A-Za-z_][A-Za-z0-9_]*;\n\s*auto & buf = ctx_buf\.[A-Za-z_][A-Za-z0-9_]*;",
        new_loop,
        meta,
        count=1,
    )
    if meta2 != meta:
        meta = meta2
        print("repaired existing ctx_buf field aliases")
    elif new_loop in meta:
        print("structured binding loop already patched correctly")
    else:
        raise RuntimeError("could not find structured binding loop or existing ctx_buf loop to repair")

old_pair = "const std::pair key = std::make_pair(tensor, assume_sync);"
new_pair = "const std::pair<const ggml_tensor *, bool> key = std::make_pair(tensor, assume_sync);"
if old_pair in meta:
    meta = meta.replace(old_pair, new_pair, 1)
    print("patched std::pair CTAD")
elif new_pair in meta:
    print("std::pair CTAD already patched")
else:
    raise RuntimeError("could not find std::pair CTAD line")

if meta != orig_meta:
    write_file(META, meta)
    print("patched", META)
else:
    print("no ggml-backend-meta.cpp changes needed")

# 2) Ensure G16A target links real ggml target.
cm = backup(CMAKE, "g16a-link-ggml-v8")
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

print("G16A v8 fix complete: buffer_config-aware C++11 rewrite + real ggml target link.")
