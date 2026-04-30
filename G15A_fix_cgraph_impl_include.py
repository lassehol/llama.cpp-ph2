#!/usr/bin/env python3
# G15A_fix_cgraph_impl_include.py
#
# Fix G15A compile failure:
#   invalid use of incomplete type 'struct ggml_cgraph'
#
# In this checkout, public ggml.h forward-declares struct ggml_cgraph.
# The full struct layout is needed by the G15A graph_compute walker and smoke,
# so include the internal ggml-impl.h beside ggml-backend-impl.h.

import os
import time

ROOT = "/workspace/notebooks/llama.cpp-ph2"
BASE = os.path.join(ROOT, "ggml/src/ggml-cuda8")
FILES = [
    os.path.join(BASE, "ggml-cuda8-ggml-backend.cpp"),
    os.path.join(BASE, "ggml-cuda8-ggml-backend-graph-compute-add-smoke.cpp"),
]


def read_file(path):
    with open(path, "r") as f:
        return f.read()


def write_file(path, data):
    with open(path, "w") as f:
        f.write(data)


def backup(path, data):
    b = path + ".g15a-cgraph-impl-backup-" + str(int(time.time()))
    write_file(b, data)
    print("backup", b)


def patch(path):
    old = read_file(path)
    new = old

    if '#include "ggml-impl.h"' not in new:
        if '#include "ggml.h"\n' in new:
            new = new.replace(
                '#include "ggml.h"\n',
                '#include "ggml.h"\n#include "ggml-impl.h"\n',
                1)
        else:
            raise SystemExit("could not find #include \"ggml.h\" in " + path)

    if new != old:
        backup(path, old)
        write_file(path, new)
        print("patched", path)
    else:
        print("already patched", path)


for path in FILES:
    if not os.path.exists(path):
        raise SystemExit("missing file: " + path)
    patch(path)

print("G15A cgraph impl include fix complete.")
