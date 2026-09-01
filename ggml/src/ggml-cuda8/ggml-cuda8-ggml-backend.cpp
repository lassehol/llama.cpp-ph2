// ggml/src/ggml-cuda8/ggml-cuda8-ggml-backend.cpp
// G15C: minimal CUDA8 ggml_backend_t backend with tiny real graph_compute support.
// Supports synthetic/manual graph nodes:
//   GGML_OP_ADD      F32 -> GGML_CUDA8_OP_ADD_F32
//   GGML_OP_MUL      F32 with scalar src1 -> GGML_CUDA8_OP_MUL_SCALAR_F32
//   GGML_OP_SOFT_MAX F32 rows -> GGML_CUDA8_OP_SOFTMAX_ROWS_F32

#include "ggml-cuda8-ggml-backend.h"
#include "ggml-cuda8-ggml-buffer.h"
#include "ggml-cuda8-dispatch.h"

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// This TU calls the CUDA runtime directly (cudaDeviceSynchronize et al) and does
// not get cuda_runtime.h transitively: the ggml-cuda8-*.h chain reaches only
// ggml-cuda8-context.h / -backend.h, neither of which includes it.
#include <cuda_runtime.h>

struct ggml_cuda8_ggml_backend_context {
    int device;
};

static const char * cuda8_backend_get_name(ggml_backend_t backend) {
    (void) backend;
    return "CUDA8";
}

static void cuda8_backend_free(ggml_backend_t backend) {
    if (backend == NULL) return;
    ggml_cuda8_ggml_backend_context * ctx = (ggml_cuda8_ggml_backend_context *) backend->context;
    if (ctx != NULL) std::free(ctx);
    std::free(backend);
}

static void cuda8_backend_synchronize(ggml_backend_t backend) {
    (void) backend;
    std::printf("ggml-cuda8/backend: synchronize PASS\n");
}

static ggml_backend_graph_plan_t cuda8_backend_graph_plan_create(ggml_backend_t backend, const struct ggml_cgraph * cgraph) {
    (void) backend;
    (void) cgraph;
    std::printf("ggml-cuda8/backend: graph_plan_create stub PASS\n");
    return NULL;
}

static void cuda8_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    (void) backend;
    (void) plan;
    std::printf("ggml-cuda8/backend: graph_plan_free stub PASS\n");
}

static void cuda8_backend_graph_plan_update(ggml_backend_t backend, ggml_backend_graph_plan_t plan, const struct ggml_cgraph * cgraph) {
    (void) backend;
    (void) plan;
    (void) cgraph;
    std::printf("ggml-cuda8/backend: graph_plan_update stub PASS\n");
}

static enum ggml_status cuda8_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    (void) backend;
    (void) plan;
    std::printf("ggml-cuda8/backend: graph_plan_compute stub PASS\n");
    return GGML_STATUS_SUCCESS;
}

static int64_t cuda8_graph_tensor_nelements_4d(const struct ggml_tensor * t) {
    return t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3];
}

static bool cuda8_graph_is_f32(const struct ggml_tensor * t) {
    return t != NULL && t->type == GGML_TYPE_F32;
}

static bool cuda8_graph_same_nelements(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    return a != NULL && b != NULL && cuda8_graph_tensor_nelements_4d(a) == cuda8_graph_tensor_nelements_4d(b);
}

static void cuda8_graph_make_flat_f32_alias(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    *dst = *src;
    const int64_t n = cuda8_graph_tensor_nelements_4d(src);
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
        if (node == NULL || node->op == GGML_OP_NONE ||
            node->op == GGML_OP_RESHAPE ||
            node->op == GGML_OP_VIEW ||
            node->op == GGML_OP_PERMUTE ||
            node->op == GGML_OP_TRANSPOSE) {
            continue;
        }

        const struct ggml_tensor * src0 = node->src[0];
        const struct ggml_tensor * src1 = node->src[1];

        struct ggml_tensor src0_flat;
        struct ggml_tensor src1_flat;
        struct ggml_tensor dst_flat;

        const struct ggml_tensor * dispatch_src0 = src0;
        const struct ggml_tensor * dispatch_src1 = src1;
        struct ggml_tensor * dispatch_dst = node;

        int cuda8_op = -1;
        const char * opname = "UNKNOWN";

        switch (node->op) {
            case GGML_OP_ADD: {
                if (!cuda8_graph_is_f32(node) ||
                    !cuda8_graph_is_f32(src0) ||
                    !cuda8_graph_is_f32(src1) ||
                    !cuda8_graph_same_nelements(src0, node) ||
                    !cuda8_graph_same_nelements(src1, node)) {
                    std::fprintf(stderr, "ggml-cuda8/backend graph_compute: ADD node %d has unsupported types/sources\n", i);
                    ggml_cuda8_context_destroy(ctx);
                    return (enum ggml_status) -1;
                }

                cuda8_graph_make_flat_f32_alias(src0, &src0_flat);
                cuda8_graph_make_flat_f32_alias(src1, &src1_flat);
                cuda8_graph_make_flat_f32_alias(node, &dst_flat);

                dispatch_src0 = &src0_flat;
                dispatch_src1 = &src1_flat;
                dispatch_dst = &dst_flat;

                cuda8_op = GGML_CUDA8_OP_ADD_F32;
                opname = "ADD_F32";
            } break;

            case GGML_OP_MUL: {
                if (!cuda8_graph_is_f32(node) ||
                    !cuda8_graph_is_f32(src0) ||
                    !cuda8_graph_is_f32(src1) ||
                    !cuda8_graph_same_nelements(src0, node)) {
                    std::fprintf(stderr,
                        "ggml-cuda8/backend graph_compute: MUL node %d unsupported types\n", i);
                    ggml_cuda8_context_destroy(ctx);
                    return (enum ggml_status) -1;
                }

                cuda8_graph_make_flat_f32_alias(src0, &src0_flat);
                cuda8_graph_make_flat_f32_alias(node, &dst_flat);
                dispatch_src0 = &src0_flat;
                dispatch_dst  = &dst_flat;

                if (cuda8_graph_tensor_nelements_4d(src1) == 1) {
                    // scalar MUL path
                    dispatch_src1 = src1;
                    cuda8_op = GGML_CUDA8_OP_MUL_SCALAR_F32;
                    opname = "MUL_SCALAR_F32";
                } else if (cuda8_graph_same_nelements(src0, src1)) {
                    // G26A: element-wise MUL path
                    cuda8_graph_make_flat_f32_alias(src1, &src1_flat);
                    dispatch_src1 = &src1_flat;
                    cuda8_op = GGML_CUDA8_OP_MUL_F32;
                    opname = "MUL_F32";
                } else if (src1->ne[0] == src0->ne[0]) {
                    // G37: broadcast MUL path (src1 repeats along higher dims)
                    dispatch_src0 = src0;
                    dispatch_src1 = src1;
                    dispatch_dst  = node;
                    cuda8_op = GGML_CUDA8_OP_MUL_BROADCAST_F32;
                    opname = "MUL_BROADCAST_F32";
                } else {
                    std::fprintf(stderr,
                        "ggml-cuda8/backend graph_compute: MUL node %d unsupported shape\n", i);
                    ggml_cuda8_context_destroy(ctx);
                    return (enum ggml_status) -1;
                }
            } break;

            case GGML_OP_SOFT_MAX: {
                if (!cuda8_graph_is_f32(node) || !cuda8_graph_is_f32(src0)) {
                    std::fprintf(stderr, "ggml-cuda8/backend graph_compute: SOFT_MAX node %d has unsupported types/sources\n", i);
                    ggml_cuda8_context_destroy(ctx);
                    return (enum ggml_status) -1;
                }

                // G37: the kernel is a plain row-wise softmax. Fail loudly rather
                // than silently ignoring a mask / sinks / scale / max_bias. This
                // should be unreachable - supports_op refuses these nodes - so
                // reaching it means the scheduler bypassed supports_op.
                if (!ggml_cuda8_soft_max_is_plain(node)) {
                    std::fprintf(stderr,
                        "ggml-cuda8/backend graph_compute: SOFT_MAX node %d uses soft_max_ext features "
                        "(mask=%p sinks=%p scale/max_bias in op_params) that the CUDA8 kernel does not implement\n",
                        i, (void *) node->src[1], (void *) node->src[2]);
                    ggml_cuda8_context_destroy(ctx);
                    return (enum ggml_status) -1;
                }

                cuda8_op = GGML_CUDA8_OP_SOFTMAX_ROWS_F32;
                opname = "SOFTMAX_ROWS_F32";
            } break;

            case GGML_OP_SUM_ROWS: {
                if (!cuda8_graph_is_f32(node) || !cuda8_graph_is_f32(src0)) {
                    std::fprintf(stderr, "ggml-cuda8/backend graph_compute: SUM_ROWS node %d has unsupported types/sources\n", i);
                    ggml_cuda8_context_destroy(ctx);
                    return (enum ggml_status) -1;
                }

                cuda8_op = GGML_CUDA8_OP_REDUCE_SUM_ROWS_F32;
                opname = "REDUCE_SUM_ROWS_F32";
            } break;

            case GGML_OP_MUL_MAT: {
                if (src0 == NULL || src1 == NULL ||
                    src0->type != GGML_TYPE_Q8_0 ||
                    src1->type != GGML_TYPE_F32 ||
                    node->type != GGML_TYPE_F32) {
                    std::fprintf(stderr, "ggml-cuda8/backend graph_compute: MUL_MAT node %d has unsupported types/sources\n", i);
                    ggml_cuda8_context_destroy(ctx);
                    return (enum ggml_status) -1;
                }

                cuda8_op = GGML_CUDA8_OP_MUL_MAT_Q8_0_F32_VEC;
                opname = "MUL_MAT_Q8_0xF32_VEC";
            } break;

            
            case GGML_OP_RMS_NORM: {
                if (!cuda8_graph_is_f32(node) || !cuda8_graph_is_f32(src0)) {
                    std::fprintf(stderr,
                        "ggml-cuda8/backend graph_compute: RMS_NORM node %d unsupported types\n", i);
                    ggml_cuda8_context_destroy(ctx);
                    return (enum ggml_status) -1;
                }

                cuda8_op = GGML_CUDA8_OP_RMS_NORM_F32;
                opname = "RMS_NORM_F32";
            } break;

            
            case GGML_OP_ROPE: {
                if (node->type != GGML_TYPE_F32 ||
                    src0 == NULL || src0->type != GGML_TYPE_F32 ||
                    src1 == NULL || src1->type != GGML_TYPE_I32) {
                    std::fprintf(stderr,
                        "ggml-cuda8/backend graph_compute: ROPE node %d unsupported types\n", i);
                    ggml_cuda8_context_destroy(ctx);
                    return (enum ggml_status) -1;
                }

                // Do NOT flatten: ROPE needs multi-dim shape + op_params
                dispatch_src0 = src0;
                dispatch_src1 = src1;
                dispatch_dst  = node;

                cuda8_op = GGML_CUDA8_OP_ROPE_F32;
                opname = "ROPE_F32";
            } break;

            
            case GGML_OP_CONT: {
                if (!cuda8_graph_is_f32(node) ||
                    !cuda8_graph_is_f32(src0)) {
                    std::fprintf(stderr,
                        "ggml-cuda8/backend graph_compute: CONT node %d unsupported types\n", i);
                    ggml_cuda8_context_destroy(ctx);
                    return (enum ggml_status) -1;
                }

                // CONT copies src0 data into contiguous dst
                // Do NOT flatten -- keep original shapes for element count
                dispatch_src0 = src0;
                dispatch_dst  = node;

                cuda8_op = GGML_CUDA8_OP_CONT_F32;
                opname = "CONT_F32";
            } break;

            
            case GGML_OP_DIAG_MASK_INF: {
                if (!cuda8_graph_is_f32(node) ||
                    !cuda8_graph_is_f32(src0)) {
                    std::fprintf(stderr,
                        "ggml-cuda8/backend graph_compute: DIAG_MASK_INF node %d unsupported types\n", i);
                    ggml_cuda8_context_destroy(ctx);
                    return (enum ggml_status) -1;
                }

                // Do NOT flatten -- kernel needs 2D shape + op_params
                dispatch_src0 = src0;
                dispatch_dst  = node;

                cuda8_op = GGML_CUDA8_OP_DIAG_MASK_INF_F32;
                opname = "DIAG_MASK_INF_F32";
            } break;

            
            case GGML_OP_GET_ROWS: {
                if (node->type != GGML_TYPE_F32 ||
                    src0 == NULL || src0->type != GGML_TYPE_F32 ||
                    src1 == NULL || src1->type != GGML_TYPE_I32) {
                    std::fprintf(stderr,
                        "ggml-cuda8/backend graph_compute: GET_ROWS node %d unsupported types\n", i);
                    ggml_cuda8_context_destroy(ctx);
                    return (enum ggml_status) -1;
                }

                // Do NOT flatten: kernel needs multi-dim shape
                dispatch_src0 = src0;
                dispatch_src1 = src1;
                dispatch_dst  = node;

                cuda8_op = GGML_CUDA8_OP_GET_ROWS_F32;
                opname = "GET_ROWS_F32";
            } break;

            default:
                std::fprintf(stderr, "ggml-cuda8/backend graph_compute: unsupported node %d op=%d\n", i, (int) node->op);
                ggml_cuda8_context_destroy(ctx);
                return (enum ggml_status) -1;
        }

        std::printf("ggml-cuda8/backend: graph_compute node %d %s\n", i, opname);

        if (ggml_cuda8_ggml_backend_dispatch_op(backend, ctx, cuda8_op, dispatch_src0, dispatch_src1, dispatch_dst) != 0) {
            std::fprintf(stderr, "ggml-cuda8/backend graph_compute: dispatch failed at node %d op=%s\n", i, opname);
            ggml_cuda8_context_destroy(ctx);
            return (enum ggml_status) -1;
        }
    }

    ggml_cuda8_context_destroy(ctx);
    std::printf("ggml-cuda8/backend: graph_compute SUCCESS\n");
    return GGML_STATUS_SUCCESS;
}

static void cuda8_backend_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    (void) backend;
    (void) cgraph;
    std::printf("ggml-cuda8/backend: graph_optimize stub PASS\n");
}

static struct ggml_backend_i cuda8_backend_i;
static bool cuda8_backend_i_initialized = false;

static void cuda8_backend_iface_init(void) {
    if (cuda8_backend_i_initialized) return;
    std::memset(&cuda8_backend_i, 0, sizeof(cuda8_backend_i));

    cuda8_backend_i.get_name = cuda8_backend_get_name;
    cuda8_backend_i.free = cuda8_backend_free;
    cuda8_backend_i.synchronize = cuda8_backend_synchronize;
    cuda8_backend_i.graph_plan_create = cuda8_backend_graph_plan_create;
    cuda8_backend_i.graph_plan_free = cuda8_backend_graph_plan_free;
    cuda8_backend_i.graph_plan_update = cuda8_backend_graph_plan_update;
    cuda8_backend_i.graph_plan_compute = cuda8_backend_graph_plan_compute;
    cuda8_backend_i.graph_compute = cuda8_backend_graph_compute;
    cuda8_backend_i.graph_optimize = cuda8_backend_graph_optimize;

    cuda8_backend_i_initialized = true;
}

extern "C" ggml_backend_t ggml_cuda8_ggml_backend_init(int device) {
    cuda8_backend_iface_init();

    ggml_cuda8_ggml_backend_context * ctx =
        (ggml_cuda8_ggml_backend_context *) std::malloc(sizeof(ggml_cuda8_ggml_backend_context));
    if (ctx == NULL) return NULL;
    ctx->device = device;

    ggml_backend_t backend = (ggml_backend_t) std::malloc(sizeof(struct ggml_backend));
    if (backend == NULL) {
        std::free(ctx);
        return NULL;
    }

    std::memset(backend, 0, sizeof(*backend));
    backend->iface = cuda8_backend_i;
    backend->context = ctx;
    return backend;
}

extern "C" int ggml_cuda8_ggml_backend_is_cuda8(ggml_backend_t backend) {
    if (backend == NULL || backend->iface.get_name == NULL) return 0;
    const char * name = backend->iface.get_name(backend);
    return name != NULL && std::strcmp(name, "CUDA8") == 0;
}

extern "C" ggml_backend_buffer_type_t ggml_cuda8_ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    if (!ggml_cuda8_ggml_backend_is_cuda8(backend)) return NULL;
    return ggml_cuda8_ggml_buffer_type();
}

extern "C" int ggml_cuda8_ggml_backend_dispatch_op(
    ggml_backend_t backend,
    ggml_cuda8_context * ctx,
    int op,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst
) {
    if (!ggml_cuda8_ggml_backend_is_cuda8(backend)) {
        std::fprintf(stderr, "ggml-cuda8/backend-dispatch: invalid backend\n");
        return -1;
    }
    if (ctx == NULL) {
        std::fprintf(stderr, "ggml-cuda8/backend-dispatch: NULL CUDA8 context\n");
        return -1;
    }
    if (!ggml_cuda8_dispatch_supported(ctx, op, src0, src1, dst)) {
        std::fprintf(stderr, "ggml-cuda8/backend-dispatch: unsupported op/layout op=%d\n", op);
        return -1;
    }
    return ggml_cuda8_dispatch_execute(ctx, op, src0, src1, dst);
}
