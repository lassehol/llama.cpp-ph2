#!/usr/bin/env python3
# G17A2_patch_mul_mat_graph_compute.py
#
# Patch ggml-cuda8-ggml-backend.cpp so real GGML graph-builder:
#
#   GGML_OP_MUL_MAT
#
# routes through the existing CUDA8 dispatcher op:
#
#   GGML_CUDA8_OP_MUL_MAT_Q8_0_F32_VEC
#
# Python 3.5-compatible: no f-strings.

import os
import time

ROOT = "/workspace/notebooks/llama.cpp-ph2"
F = os.path.join(ROOT, "ggml/src/ggml-cuda8/ggml-cuda8-ggml-backend.cpp")

with open(F, "r") as f:
    s = f.read()

backup = F + ".g17a2-mulmat-backup-" + str(int(time.time()))
with open(backup, "w") as f:
    f.write(s)

print("backup", backup)

if "GGML_CUDA8_OP_MUL_MAT_Q8_0_F32_VEC" in s and "MUL_MAT_Q8_0xF32_VEC" in s:
    print("MUL_MAT graph_compute mapping appears already present")
else:
    # Insert before the generic unsupported-node branch.
    needle = '            } else {\\n                std::fprintf(stderr, "ggml-cuda8/backend graph_compute: unsupported node %d op=%d\\\\n", i, (int) node->op);'
    insert = '''            } else if (node->op == GGML_OP_MUL_MAT) {
                cuda8_op = GGML_CUDA8_OP_MUL_MAT_Q8_0_F32_VEC;
                opname = "MUL_MAT_Q8_0xF32_VEC";
            } else {
                std::fprintf(stderr, "ggml-cuda8/backend graph_compute: unsupported node %d op=%d\\\\n", i, (int) node->op);'''

    if needle not in s:
        # Try a slightly less indentation-sensitive fallback.
        needle = '} else {\\n                std::fprintf(stderr, "ggml-cuda8/backend graph_compute: unsupported node %d op=%d\\\\n", i, (int) node->op);'
        insert = '''} else if (node->op == GGML_OP_MUL_MAT) {
                cuda8_op = GGML_CUDA8_OP_MUL_MAT_Q8_0_F32_VEC;
                opname = "MUL_MAT_Q8_0xF32_VEC";
            } else {
                std::fprintf(stderr, "ggml-cuda8/backend graph_compute: unsupported node %d op=%d\\\\n", i, (int) node->op);'''

    if needle not in s:
        print("Could not find expected unsupported-node branch.")
        print("Please run:")
        print("  nl -ba ggml/src/ggml-cuda8/ggml-cuda8-ggml-backend.cpp | sed -n '130,215p'")
        raise SystemExit(1)

    s = s.replace(needle, insert, 1)

    with open(F, "w") as f:
        f.write(s)

    print("patched", F)
    print("G17A2: added GGML_OP_MUL_MAT -> GGML_CUDA8_OP_MUL_MAT_Q8_0_F32_VEC mapping")
