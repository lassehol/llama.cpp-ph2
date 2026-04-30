#!/usr/bin/env python3
# G15F_cleanup_graph_compute.py
#
# Replaces the patcher-grown graph_compute block in ggml-cuda8-ggml-backend.cpp
# with a stable hand-maintained implementation.

import os
import time

ROOT = "/workspace/notebooks/llama.cpp-ph2"
BACKEND_CPP = os.path.join(ROOT, "ggml/src/ggml-cuda8/ggml-cuda8-ggml-backend.cpp")
CLEAN_BLOCK = 'static int64_t cuda8_graph_tensor_nelements_4d(const struct ggml_tensor * t) {\n    return t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3];\n}\n\nstatic bool cuda8_graph_is_f32(const struct ggml_tensor * t) {\n    return t != NULL && t->type == GGML_TYPE_F32;\n}\n\nstatic bool cuda8_graph_same_nelements(const struct ggml_tensor * a, const struct ggml_tensor * b) {\n    return a != NULL && b != NULL && cuda8_graph_tensor_nelements_4d(a) == cuda8_graph_tensor_nelements_4d(b);\n}\n\nstatic void cuda8_graph_make_flat_f32_alias(const struct ggml_tensor * src, struct ggml_tensor * dst) {\n    *dst = *src;\n    const int64_t n = cuda8_graph_tensor_nelements_4d(src);\n    dst->ne[0] = n;\n    dst->ne[1] = 1;\n    dst->ne[2] = 1;\n    dst->ne[3] = 1;\n    dst->nb[0] = sizeof(float);\n    dst->nb[1] = (size_t) n * sizeof(float);\n    dst->nb[2] = dst->nb[1];\n    dst->nb[3] = dst->nb[1];\n}\n\nstatic enum ggml_status cuda8_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {\n    if (!ggml_cuda8_ggml_backend_is_cuda8(backend)) {\n        std::fprintf(stderr, "ggml-cuda8/backend graph_compute: invalid backend\\n");\n        return (enum ggml_status) -1;\n    }\n\n    if (cgraph == NULL) {\n        std::printf("ggml-cuda8/backend: graph_compute NULL-graph stub PASS\\n");\n        return GGML_STATUS_SUCCESS;\n    }\n\n    ggml_cuda8_context * ctx = NULL;\n    if (ggml_cuda8_context_create(0, &ctx) != 0 || ctx == NULL) {\n        std::fprintf(stderr, "ggml-cuda8/backend graph_compute: failed to create CUDA8 context\\n");\n        return (enum ggml_status) -1;\n    }\n\n    std::printf("ggml-cuda8/backend: graph_compute starting n_nodes=%d\\n", cgraph->n_nodes);\n\n    for (int i = 0; i < cgraph->n_nodes; ++i) {\n        struct ggml_tensor * node = cgraph->nodes[i];\n        if (node == NULL || node->op == GGML_OP_NONE) {\n            continue;\n        }\n\n        const struct ggml_tensor * src0 = node->src[0];\n        const struct ggml_tensor * src1 = node->src[1];\n\n        struct ggml_tensor src0_flat;\n        struct ggml_tensor src1_flat;\n        struct ggml_tensor dst_flat;\n\n        const struct ggml_tensor * dispatch_src0 = src0;\n        const struct ggml_tensor * dispatch_src1 = src1;\n        struct ggml_tensor * dispatch_dst = node;\n\n        int cuda8_op = -1;\n        const char * opname = "UNKNOWN";\n\n        switch (node->op) {\n            case GGML_OP_ADD: {\n                if (!cuda8_graph_is_f32(node) ||\n                    !cuda8_graph_is_f32(src0) ||\n                    !cuda8_graph_is_f32(src1) ||\n                    !cuda8_graph_same_nelements(src0, node) ||\n                    !cuda8_graph_same_nelements(src1, node)) {\n                    std::fprintf(stderr, "ggml-cuda8/backend graph_compute: ADD node %d has unsupported types/sources\\n", i);\n                    ggml_cuda8_context_destroy(ctx);\n                    return (enum ggml_status) -1;\n                }\n\n                cuda8_graph_make_flat_f32_alias(src0, &src0_flat);\n                cuda8_graph_make_flat_f32_alias(src1, &src1_flat);\n                cuda8_graph_make_flat_f32_alias(node, &dst_flat);\n\n                dispatch_src0 = &src0_flat;\n                dispatch_src1 = &src1_flat;\n                dispatch_dst = &dst_flat;\n\n                cuda8_op = GGML_CUDA8_OP_ADD_F32;\n                opname = "ADD_F32";\n            } break;\n\n            case GGML_OP_MUL: {\n                if (!cuda8_graph_is_f32(node) ||\n                    !cuda8_graph_is_f32(src0) ||\n                    !cuda8_graph_is_f32(src1) ||\n                    cuda8_graph_tensor_nelements_4d(src1) != 1 ||\n                    !cuda8_graph_same_nelements(src0, node)) {\n                    std::fprintf(stderr, "ggml-cuda8/backend graph_compute: MUL node %d is not supported scalar F32 MUL\\n", i);\n                    ggml_cuda8_context_destroy(ctx);\n                    return (enum ggml_status) -1;\n                }\n\n                cuda8_graph_make_flat_f32_alias(src0, &src0_flat);\n                cuda8_graph_make_flat_f32_alias(node, &dst_flat);\n\n                dispatch_src0 = &src0_flat;\n                dispatch_src1 = src1;\n                dispatch_dst = &dst_flat;\n\n                cuda8_op = GGML_CUDA8_OP_MUL_SCALAR_F32;\n                opname = "MUL_SCALAR_F32";\n            } break;\n\n            case GGML_OP_SOFT_MAX: {\n                if (!cuda8_graph_is_f32(node) || !cuda8_graph_is_f32(src0)) {\n                    std::fprintf(stderr, "ggml-cuda8/backend graph_compute: SOFT_MAX node %d has unsupported types/sources\\n", i);\n                    ggml_cuda8_context_destroy(ctx);\n                    return (enum ggml_status) -1;\n                }\n\n                cuda8_op = GGML_CUDA8_OP_SOFTMAX_ROWS_F32;\n                opname = "SOFTMAX_ROWS_F32";\n            } break;\n\n            case GGML_OP_SUM_ROWS: {\n                if (!cuda8_graph_is_f32(node) || !cuda8_graph_is_f32(src0)) {\n                    std::fprintf(stderr, "ggml-cuda8/backend graph_compute: SUM_ROWS node %d has unsupported types/sources\\n", i);\n                    ggml_cuda8_context_destroy(ctx);\n                    return (enum ggml_status) -1;\n                }\n\n                cuda8_op = GGML_CUDA8_OP_REDUCE_SUM_ROWS_F32;\n                opname = "REDUCE_SUM_ROWS_F32";\n            } break;\n\n            default:\n                std::fprintf(stderr, "ggml-cuda8/backend graph_compute: unsupported node %d op=%d\\n", i, (int) node->op);\n                ggml_cuda8_context_destroy(ctx);\n                return (enum ggml_status) -1;\n        }\n\n        std::printf("ggml-cuda8/backend: graph_compute node %d %s\\n", i, opname);\n\n        if (ggml_cuda8_ggml_backend_dispatch_op(backend, ctx, cuda8_op, dispatch_src0, dispatch_src1, dispatch_dst) != 0) {\n            std::fprintf(stderr, "ggml-cuda8/backend graph_compute: dispatch failed at node %d op=%s\\n", i, opname);\n            ggml_cuda8_context_destroy(ctx);\n            return (enum ggml_status) -1;\n        }\n    }\n\n    ggml_cuda8_context_destroy(ctx);\n    std::printf("ggml-cuda8/backend: graph_compute SUCCESS\\n");\n    return GGML_STATUS_SUCCESS;\n}\n'

START_CANDIDATES = [
    "static int64_t cuda8_graph_tensor_nelements_4d",
    "static int64_t tensor_nelements_4d",
]
END_MARKER = "static void cuda8_backend_graph_optimize"

def read_file(path):
    with open(path, "r") as f:
        return f.read()

def write_file(path, data):
    with open(path, "w") as f:
        f.write(data)

def find_start(text):
    hits = []
    for marker in START_CANDIDATES:
        pos = text.find(marker)
        if pos >= 0:
            hits.append(pos)
    if not hits:
        raise RuntimeError("could not find graph_compute helper start marker")
    return min(hits)

text = read_file(BACKEND_CPP)
start = find_start(text)
end = text.find(END_MARKER, start)
if end < 0:
    raise RuntimeError("could not find graph_optimize end marker after graph_compute block")

backup = BACKEND_CPP + ".g15f-cleanup-backup-" + str(int(time.time()))
write_file(backup, text)

new_text = text[:start] + CLEAN_BLOCK.strip() + "\n\n" + text[end:]
write_file(BACKEND_CPP, new_text)

print("backup", backup)
print("cleaned", BACKEND_CPP)
print("G15F graph_compute cleanup complete.")
