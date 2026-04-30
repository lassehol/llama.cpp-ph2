#!/usr/bin/env python3
import os
import time

PATH = "/workspace/notebooks/llama.cpp-ph2/ggml/src/ggml-cuda8/ggml-cuda8-ggml-backend.cpp"

NEW_GRAPH_COMPUTE = r'''
static int64_t tensor_nelements_4d(const struct ggml_tensor * t) {
    return t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3];
}

static void make_flat_f32_alias(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    *dst = *src;
    const int64_t n = tensor_nelements_4d(src);
    dst->ne[0] = n;
    dst->ne[1] = 1;
    dst->ne[2] = 1;
    dst->ne[3] = 1;
    dst->nb[0] = sizeof(float);
    dst->nb[1] = (size_t) n * sizeof(float);
    dst->nb[2] = dst->nb[1];
    dst->nb[3] = dst->nb[1];
}

static enum ggml_status cuda8_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    if (!ggml_cuda8_ggml_backend_is_cuda8(backend)) {
        std::fprintf(stderr, "ggml-cuda8/backend graph_compute: invalid backend\n");
        return (enum ggml_status) -1;
    }

    if (cgraph == NULL) {
        std::printf("ggml-cuda8/backend: graph_compute NULL-graph stub PASS\n");
        return GGML_STATUS_SUCCESS;
    }

    ggml_cuda8_context * ctx = NULL;
    if (ggml_cuda8_context_create(0, &ctx) != 0 || ctx == NULL) {
        std::fprintf(stderr, "ggml-cuda8/backend graph_compute: failed to create CUDA8 context\n");
        return (enum ggml_status) -1;
    }

    std::printf("ggml-cuda8/backend: graph_compute starting n_nodes=%d\n", cgraph->n_nodes);

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (node == NULL || node->op == GGML_OP_NONE) {
            continue;
        }

        int cuda8_op = -1;
        const char * opname = "UNKNOWN";

        const struct ggml_tensor * src0 = node->src[0];
        const struct ggml_tensor * src1 = node->src[1];

        struct ggml_tensor src0_flat;
        struct ggml_tensor src1_flat;
        struct ggml_tensor dst_flat;

        const struct ggml_tensor * dispatch_src0 = src0;
        const struct ggml_tensor * dispatch_src1 = src1;
        struct ggml_tensor * dispatch_dst = node;

        if (node->op == GGML_OP_ADD) {
            if (node->type != GGML_TYPE_F32 ||
                src0 == NULL ||
                src1 == NULL ||
                src0->type != GGML_TYPE_F32 ||
                src1->type != GGML_TYPE_F32 ||
                tensor_nelements_4d(src0) != tensor_nelements_4d(node) ||
                tensor_nelements_4d(src1) != tensor_nelements_4d(node)) {
                std::fprintf(stderr, "ggml-cuda8/backend graph_compute: ADD node %d has unsupported types/sources\n", i);
                ggml_cuda8_context_destroy(ctx);
                return (enum ggml_status) -1;
            }

            make_flat_f32_alias(src0, &src0_flat);
            make_flat_f32_alias(src1, &src1_flat);
            make_flat_f32_alias(node, &dst_flat);

            dispatch_src0 = &src0_flat;
            dispatch_src1 = &src1_flat;
            dispatch_dst = &dst_flat;

            cuda8_op = GGML_CUDA8_OP_ADD_F32;
            opname = "ADD_F32";
        } else if (node->op == GGML_OP_MUL) {
            if (node->type != GGML_TYPE_F32 ||
                src0 == NULL ||
                src1 == NULL ||
                src0->type != GGML_TYPE_F32 ||
                src1->type != GGML_TYPE_F32 ||
                tensor_nelements_4d(src1) != 1 ||
                tensor_nelements_4d(src0) != tensor_nelements_4d(node)) {
                std::fprintf(stderr, "ggml-cuda8/backend graph_compute: MUL node %d is not supported scalar F32 MUL\n", i);
                ggml_cuda8_context_destroy(ctx);
                return (enum ggml_status) -1;
            }

            make_flat_f32_alias(src0, &src0_flat);
            make_flat_f32_alias(node, &dst_flat);

            dispatch_src0 = &src0_flat;
            dispatch_src1 = src1;
            dispatch_dst = &dst_flat;

            cuda8_op = GGML_CUDA8_OP_MUL_SCALAR_F32;
            opname = "MUL_SCALAR_F32";
        } else if (node->op == GGML_OP_SOFT_MAX) {
            if (node->type != GGML_TYPE_F32 ||
                src0 == NULL ||
                src0->type != GGML_TYPE_F32) {
                std::fprintf(stderr, "ggml-cuda8/backend graph_compute: SOFT_MAX node %d has unsupported types/sources\n", i);
                ggml_cuda8_context_destroy(ctx);
                return (enum ggml_status) -1;
            }

            cuda8_op = GGML_CUDA8_OP_SOFTMAX_ROWS_F32;
            opname = "SOFTMAX_ROWS_F32";
        } else {
            std::fprintf(stderr, "ggml-cuda8/backend graph_compute: unsupported node %d op=%d\n", i, (int) node->op);
            ggml_cuda8_context_destroy(ctx);
            return (enum ggml_status) -1;
        }

        std::printf("ggml-cuda8/backend: graph_compute node %d %s\n", i, opname);

        if (ggml_cuda8_ggml_backend_dispatch_op(
                backend,
                ctx,
                cuda8_op,
                dispatch_src0,
                dispatch_src1,
                dispatch_dst) != 0) {
            std::fprintf(stderr, "ggml-cuda8/backend graph_compute: dispatch failed at node %d op=%s\n", i, opname);
            ggml_cuda8_context_destroy(ctx);
            return (enum ggml_status) -1;
        }
    }

    ggml_cuda8_context_destroy(ctx);
    std::printf("ggml-cuda8/backend: graph_compute SUCCESS\n");
    return GGML_STATUS_SUCCESS;
}
'''

def replace_block(text, start_marker, end_marker, replacement):
    start = text.find(start_marker)
    if start < 0:
        raise RuntimeError("start marker not found: " + start_marker)

    end = text.find(end_marker, start)
    if end < 0:
        raise RuntimeError("end marker not found after start marker: " + end_marker)

    return text[:start] + replacement.strip() + "\n\n" + text[end:]

with open(PATH, "r") as f:
    old = f.read()

backup = PATH + ".g15c-flat-alias-backup-" + str(int(time.time()))
with open(backup, "w") as f:
    f.write(old)
print("backup", backup)

# Replace tensor_nelements_4d + graph_compute together.
# The block ends immediately before graph_optimize in our G15B/G15C backend file.
new = replace_block(
    old,
    "static int64_t tensor_nelements_4d",
    "static void cuda8_backend_graph_optimize",
    NEW_GRAPH_COMPUTE + "\n\n"
)

with open(PATH, "w") as f:
    f.write(new)

print("wrote", PATH)
print("G15C flat alias graph_compute fix complete.")
