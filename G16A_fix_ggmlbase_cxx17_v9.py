#!/usr/bin/env python3
# G16A_fix_ggmlbase_cxx17_v9.py
#
# G16A v9 fix:
#   v8 fixed ggml-backend-meta.cpp's C++17 structured binding issue, then the
#   full ggml target progressed to gguf.cpp. gguf.cpp also requires C++17 because
#   it uses if constexpr and relies on C++17 std::string::data() returning char *.
#
# This patch keeps G16A linked to the real ggml target and appends an explicit
# -std=gnu++17 compile option to ggml-base C++ sources. This is stronger than
# CXX_STANDARD, which did not override the legacy build flags in this tree.

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
# transitive source set.  This checkout has C++17 syntax in ggml-base sources
# such as gguf.cpp and ggml-backend-meta.cpp, while the CUDA8/Fermi build mode
# otherwise defaults older.  Append -std=gnu++17 explicitly to ggml-base C++
# compilation; target CXX_STANDARD alone was not sufficient in this tree.
# ---------------------------------------------------------------------------

if (TARGET ggml-base)
    target_compile_options(ggml-base PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++17>)
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
        line = line[:-1].split("=", 1)[0].strip()
        mfield = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?$", line)
        if mfield:
            fields.append(mfield.group(1))
    if len(fields) < 2:
        raise RuntimeError("could not infer first two fields of struct buffer_config; fields=%r" % (fields,))
    return fields[0], fields[1]

# 1) Keep the ggml-backend-meta.cpp C++11-compatible patch from v8, idempotently.
meta = backup(META, "g16a-meta-cpp11-v9")
orig_meta = meta
field0, field1 = get_buffer_config_fields(meta)
print("detected buffer_config fields:", field0, field1)

old_loop = "for (auto & [ctx, buf] : buf_ctx->buf_configs) {"
new_loop = "for (auto & ctx_buf : buf_ctx->buf_configs) {\n        auto & ctx = ctx_buf.%s;\n        auto & buf = ctx_buf.%s;" % (field0, field1)
if old_loop in meta:
    meta = meta.replace(old_loop, new_loop, 1)
    print("patched original structured binding loop")
else:
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

# 2) Replace G16A CMake block with real ggml target + explicit C++17 compile option for ggml-base.
cm = backup(CMAKE, "g16a-ggmlbase-cxx17-v9")
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

print("G16A v9 fix complete: ggml-base explicit -std=gnu++17 + real ggml target link.")
