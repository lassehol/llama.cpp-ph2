#!/usr/bin/env python3
# G17A2_patch_mul_mat_graph_compute_v2.py
#
# Add GGML_OP_MUL_MAT routing to ggml-cuda8 backend graph_compute:
#
#   GGML_OP_MUL_MAT
#     -> GGML_CUDA8_OP_MUL_MAT_Q8_0_F32_VEC
#
# This targets the current switch(node->op) layout in:
#   ggml/src/ggml-cuda8/ggml-cuda8-ggml-backend.cpp
#
# Python 3.5-compatible: no f-strings.

import os
import time

ROOT = "/workspace/notebooks/llama.cpp-ph2"
F = os.path.join(ROOT, "ggml/src/ggml-cuda8/ggml-cuda8-ggml-backend.cpp")

with open(F, "r") as f:
    s = f.read()

backup = F + ".g17a2-mulmat-v2-backup-" + str(int(time.time()))
with open(backup, "w") as f:
    f.write(s)

print("backup", backup)

if "case GGML_OP_MUL_MAT:" in s and "MUL_MAT_Q8_0xF32_VEC" in s:
    print("G17A2 MUL_MAT graph_compute mapping already present")
    raise SystemExit(0)

needle = '''            default:
                std::fprintf(stderr, "ggml-cuda8/backend graph_compute: unsupported node %d op=%d\\n", i, (int) node->op);
                ggml_cuda8_context_destroy(ctx);
                return (enum ggml_status) -1;
'''

insert = '''            case GGML_OP_MUL_MAT: {
                if (src0 == NULL || src1 == NULL ||
                    src0->type != GGML_TYPE_Q8_0 ||
                    src1->type != GGML_TYPE_F32 ||
                    node->type != GGML_TYPE_F32) {
                    std::fprintf(stderr, "ggml-cuda8/backend graph_compute: MUL_MAT node %d has unsupported types/sources\\n", i);
                    ggml_cuda8_context_destroy(ctx);
                    return (enum ggml_status) -1;
                }

                cuda8_op = GGML_CUDA8_OP_MUL_MAT_Q8_0_F32_VEC;
                opname = "MUL_MAT_Q8_0xF32_VEC";
            } break;

            default:
                std::fprintf(stderr, "ggml-cuda8/backend graph_compute: unsupported node %d op=%d\\n", i, (int) node->op);
                ggml_cuda8_context_destroy(ctx);
                return (enum ggml_status) -1;
'''

if needle not in s:
    print("Could not find exact default branch. Current relevant section:")
    print("  nl -ba ggml/src/ggml-cuda8/ggml-cuda8-ggml-backend.cpp | sed -n '185,210p'")
    raise SystemExit(1)

s = s.replace(needle, insert, 1)

with open(F, "w") as f:
    f.write(s)

print("patched", F)
print("G17A2 v2: added GGML_OP_MUL_MAT -> GGML_CUDA8_OP_MUL_MAT_Q8_0_F32_VEC mapping")
