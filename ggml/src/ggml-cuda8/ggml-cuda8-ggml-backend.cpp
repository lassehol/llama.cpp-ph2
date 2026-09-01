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
#include <atomic>
#include <time.h>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
// This TU calls the CUDA runtime directly (cudaDeviceSynchronize et al) and does
// not get cuda_runtime.h transitively: the ggml-cuda8-*.h chain reaches only
// ggml-cuda8-context.h / -backend.h, neither of which includes it.
#include <cuda_runtime.h>
struct ggml_cuda8_ggml_backend_context {
    int device;
};
// -----------------------------------------------------------------------
// G-fix: sticky poisoned-device flag.
//
// A fatal CUDA error surfaced by cudaDeviceSynchronize() (illegal memory
// access, launch failure, uncorrectable ECC error, device-side assert)
// typically invalidates the CUDA context for that device at the driver
// level, not just the ggml_cuda8_context that happened to be alive when
// the fault was detected. On most driver versions every subsequent CUDA
// call in this process -- including a brand new
// ggml_cuda8_context_create() in the *next* graph_compute() call --
// keeps failing until the process exits.
//
// Previously cuda8_backend_synchronize() only logged this and returned
// (ggml_backend_i::synchronize is void, so there is no return channel to
// propagate it through anyway). That meant the next graph_compute() call
// failed with an unrelated-looking "failed to create CUDA8 context" or a
// dispatch failure buried inside some other op, instead of a message
// that points back at the real cause.
//
// This flag makes that failure loud and immediate at the next dispatch
// attempt. It is intentionally process-wide (not per ggml_cuda8_context
// or per ggml_backend_t) because the fault it flags is a process-wide
// CUDA/driver condition for the device, not a property of any one
// context or backend handle -- and because a fresh ggml_cuda8_context
// created after the fault would not itself carry any memory of it.
static std::atomic<bool> g_cuda8_device_poisoned{false};
// Fatal error classes: these correspond to the CUDA context (and, on
// most driver versions, the whole per-process CUDA state for that
// device) becoming unusable until process exit. This is deliberately a
// narrow allow-list rather than "anything != cudaSuccess", because most
// cudaErrorInvalidConfiguration / cudaErrorInvalidValue-class failures
// (e.g. an over-large grid without a stride loop, the class of bug G38
// fixed) are recoverable launch-time mistakes, not device-level
// corruption, and should not permanently disable the backend.
//
// Restricted to error codes confirmed present in CUDA 8.0.61 (the
// container toolchain this translation unit is actually built with, per
// the ggml-cuda8-kernels stage) rather than newer codes such as
// cudaErrorHardwareStackError / cudaErrorIllegalInstruction /
// cudaErrorMisalignedAddress / cudaErrorInvalidAddressSpace /
// cudaErrorInvalidPc that were considered but not included here pending
// a version check against that toolkit's cuda_runtime_api.h.
static bool cuda8_error_is_fatal(cudaError_t err) {
    switch (err) {
        case cudaErrorIllegalAddress:
        case cudaErrorLaunchFailure:
        case cudaErrorECCUncorrectable:
        case cudaErrorAssert:
            return true;
        default:
            return false;
    }
}
// Query hook for callers/tests that want to check device health without
// attempting (and failing) a dispatch first. Not yet declared in
// ggml-cuda8-ggml-backend.h -- add a prototype there if this needs to be
// called from outside this translation unit.
extern "C" int ggml_cuda8_ggml_backend_device_is_poisoned(void) {
    return g_cuda8_device_poisoned.load(std::memory_order_acquire) ? 1 : 0;
}
// ==== G57 timing probe — measurement only, gated by GGML_CUDA8_TIMING=1 ====
static bool cuda8_timing_enabled() {
    static int e = -1;
    if (e < 0) {
        const char * s = std::getenv("GGML_CUDA8_TIMING");
        e = (s != NULL && s[0] != '\0' && s[0] != '0') ? 1 : 0;
    }
    return e == 1;
}
static double cuda8_now_ms() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double) t.tv_sec * 1e3 + (double) t.tv_nsec / 1e6;
}
struct cuda8_timing_state {
    std::map<std::string, double> op_ms;
    std::map<std::string, long>   op_count;
    double total_graph_ms;
    long   graph_calls;
};
static cuda8_timing_state & cuda8_timing() {
    static cuda8_timing_state s;
    return s;
}
static void cuda8_timing_report() {
    cuda8_timing_state & s = cuda8_timing();
    if (s.graph_calls == 0) return;
    double sum_ops = 0.0;
    std::vector<std::pair<double, std::string> > v;
    for (std::map<std::string, double>::iterator it = s.op_ms.begin(); it != s.op_ms.end(); ++it) {
        v.push_back(std::make_pair(it->second, it->first));
        sum_ops += it->second;
    }
    std::sort(v.begin(), v.end());
    std::reverse(v.begin(), v.end());
    std::fprintf(stderr,
        "\nggml-cuda8 TIMING: %ld graph_compute calls\n"
        "  sum of per-op GPU compute (synced): %.1f ms  (= %.1f ms/graph)\n"
        "  --> compare against baseline ms/token = 1000 / gen_t_per_s\n"
        "  per-op-type GPU time, by total ms:\n",
        s.graph_calls, sum_ops, sum_ops / (double) s.graph_calls);
    for (size_t i = 0; i < v.size(); ++i) {
        const std::string & name = v[i].second;
        std::fprintf(stderr, "  %10.1f ms  %9ld calls  %8.4f ms/call  %s\n",
            v[i].first, s.op_count[name],
            v[i].first / (double) s.op_count[name], name.c_str());
    }
}
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
    // G-fix: sync the device this backend was actually initialized for,
    // rather than whatever device happens to be "current" on the calling
    // thread. Dead code today (single GTX 560 = device 0 in every
    // deployment so far), but cuda8_backend_synchronize() previously
    // ignored ctx->device entirely -- this only matters once a second
    // CUDA8 device, or another CUDA context that changes the current
    // device, ever coexists in the same process.
    ggml_cuda8_ggml_backend_context * ctx =
        (ggml_cuda8_ggml_backend_context *) (backend != NULL ? backend->context : NULL);
    if (ctx != NULL) {
        cudaSetDevice(ctx->device);
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "ggml-cuda8/backend: synchronize FAILED: %s\n", cudaGetErrorString(err));
        if (cuda8_error_is_fatal(err)) {
            // G-fix: latch the poisoned flag. See the comment above
            // g_cuda8_device_poisoned for why this is process-wide rather
            // than scoped to `ctx` or `backend`.
            const bool was_already_poisoned =
                g_cuda8_device_poisoned.exchange(true, std::memory_order_acq_rel);
            if (!was_already_poisoned) {
                std::fprintf(stderr,
                    "ggml-cuda8/backend: device marked POISONED after fatal error %s -- "
                    "this error class typically invalidates the CUDA context for the "
                    "remainder of the process. Further CUDA8 dispatch calls will be "
                    "refused immediately instead of failing with an unrelated-looking "
                    "error further down the pipeline. Restart the process to recover "
                    "the device.\n",
                    cudaGetErrorString(err));
            }
        }
    } else {
        std::printf("ggml-cuda8/backend: synchronize PASS\n");
    }
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
    // G-fix: fail fast and clearly if a previous fatal CUDA error already
    // poisoned the device, rather than proceeding into
    // ggml_cuda8_context_create() (which will likely also fail, but with
    // a generic "cudaSetDevice failed" / "failed to create CUDA8 context"
    // message that does not point back at the original fault).
    if (g_cuda8_device_poisoned.load(std::memory_order_acquire)) {
        std::fprintf(stderr,
            "ggml-cuda8/backend graph_compute: refusing to run -- device was "
            "marked poisoned by a previous fatal CUDA error (see the earlier "
            "'synchronize FAILED' / 'marked POISONED' log). Restart the "
            "process to recover the CUDA8 device.\n");
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
				if (ggml_cuda8_soft_max_is_plain(node)) {
					// Fast path, unchanged since before G41 - every existing
					// plain-softmax pipeline (G16C, G19A, G32A, ...) still takes this.
					cuda8_op = GGML_CUDA8_OP_SOFTMAX_ROWS_F32;
					opname = "SOFTMAX_ROWS_F32";
				} else if (ggml_cuda8_soft_max_is_supported_ext(node)) {
					// G41: mask/scale/ALiBi. dispatch_src0/src1/dst are already src0/
					// src1(mask)/node by default - not flattened, same as ROPE/
					// DIAG_MASK_INF/SWIGLU, since the kernel needs the real shape.
					cuda8_op = GGML_CUDA8_OP_SOFTMAX_EXT_F32;
					opname = "SOFTMAX_EXT_F32";
				} else {
					std::fprintf(stderr,
						"ggml-cuda8/backend graph_compute: SOFT_MAX node %d uses features "
						"this backend does not implement (attention sinks and/or non-F32 mask)\n", i);
					ggml_cuda8_context_destroy(ctx);
					return (enum ggml_status) -1;
				}
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
                    src1->type != GGML_TYPE_F32 ||
                    node->type != GGML_TYPE_F32) {
                    std::fprintf(stderr, "ggml-cuda8/backend graph_compute: MUL_MAT node %d unsupported src1/dst types\n", i);
                    ggml_cuda8_context_destroy(ctx);
                    return (enum ggml_status) -1;
                }
                if (src0->type == GGML_TYPE_Q8_0) {
                    cuda8_op = GGML_CUDA8_OP_MUL_MAT_Q8_0_F32_VEC;
                    opname = "MUL_MAT_Q8_0xF32";
                } else if (src0->type == GGML_TYPE_Q4_K) {
                    cuda8_op = GGML_CUDA8_OP_MUL_MAT_Q4_K_F32;
                    opname = "MUL_MAT_Q4_KxF32";
                } else if (src0->type == GGML_TYPE_Q6_K) {
                    cuda8_op = GGML_CUDA8_OP_MUL_MAT_Q6_K_F32;
                    opname = "MUL_MAT_Q6_KxF32";
                } else if (src0->type == GGML_TYPE_F32) {
                    // G42: batched attention matmul. Not flattened - kernel
                    // needs real ne[]/nb[] for broadcast + permuted views.
                    dispatch_src0 = src0;
                    dispatch_src1 = src1;
                    dispatch_dst  = node;
                    cuda8_op = GGML_CUDA8_OP_MUL_MAT_F32_F32;
                    opname = "MUL_MAT_F32xF32";
                } else {
                    std::fprintf(stderr, "ggml-cuda8/backend graph_compute: MUL_MAT node %d unsupported src0 type %d\n", i, (int)src0->type);
                    ggml_cuda8_context_destroy(ctx);
                    return (enum ggml_status) -1;
                }
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
                    src0 == NULL ||
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
                if (src0->type == GGML_TYPE_F32) {
                    cuda8_op = GGML_CUDA8_OP_GET_ROWS_F32;
                    opname = "GET_ROWS_F32";
                } else if (src0->type == GGML_TYPE_Q4_K) {
                    cuda8_op = GGML_CUDA8_OP_GET_ROWS_Q4_K;
                    opname = "GET_ROWS_Q4_K";
                } else if (src0->type == GGML_TYPE_Q6_K) {
                    cuda8_op = GGML_CUDA8_OP_GET_ROWS_Q6_K;
                    opname = "GET_ROWS_Q6_K";
                } else {
                    std::fprintf(stderr,
                        "ggml-cuda8/backend graph_compute: GET_ROWS node %d unsupported src0 type %d\n", i, (int)src0->type);
                    ggml_cuda8_context_destroy(ctx);
                    return (enum ggml_status) -1;
                }
            } break;
            // G40: SwiGLU. src1 may legitimately be NULL - that is the
            // halves form, where gate and up are the two halves of each
            // src0 row rather than separate tensors.
            case GGML_OP_GLU: {
                if (node->type != GGML_TYPE_F32 ||
                    src0 == NULL || src0->type != GGML_TYPE_F32 ||
                    (src1 != NULL && src1->type != GGML_TYPE_F32)) {
                    std::fprintf(stderr,
                        "ggml-cuda8/backend graph_compute: GLU node %d unsupported types\n", i);
                    ggml_cuda8_context_destroy(ctx);
                    return (enum ggml_status) -1;
                }
                const int32_t glu_op = ggml_get_op_params_i32(node, 0);
                if (glu_op != GGML_GLU_OP_SWIGLU) {
                    std::fprintf(stderr,
                        "ggml-cuda8/backend graph_compute: GLU node %d op %d is not SWIGLU\n",
                        i, (int) glu_op);
                    ggml_cuda8_context_destroy(ctx);
                    return (enum ggml_status) -1;
                }
                // Keep the multi-dim shape: the kernel works row-wise.
                dispatch_src0 = src0;
                dispatch_src1 = src1;
                dispatch_dst  = node;
                cuda8_op = GGML_CUDA8_OP_SWIGLU_F32;
                opname = "SWIGLU_F32";
            } break;
            // G43: SET_ROWS - the KV cache write.
            case GGML_OP_SET_ROWS: {
                if (node->type != GGML_TYPE_F32 ||
                    src0 == NULL || src0->type != GGML_TYPE_F32 ||
                    src1 == NULL || src1->type != GGML_TYPE_I64) {
                    std::fprintf(stderr,
                        "ggml-cuda8/backend graph_compute: SET_ROWS node %d unsupported types "
                        "(dst=%d src0=%d src1=%d; F16 cache needs G49)\n",
                        i, (int) node->type,
                        src0 ? (int) src0->type : -1,
                        src1 ? (int) src1->type : -1);
                    ggml_cuda8_context_destroy(ctx);
                    return (enum ggml_status) -1;
                }
                dispatch_src0 = src0;
                dispatch_src1 = src1;
                dispatch_dst  = node;
                cuda8_op = GGML_CUDA8_OP_SET_ROWS_F32;
                opname = "SET_ROWS_F32";
            } break;
            default:
                std::fprintf(stderr, "ggml-cuda8/backend graph_compute: unsupported node %d op=%d\n", i, (int) node->op);
                ggml_cuda8_context_destroy(ctx);
                return (enum ggml_status) -1;
        }
        std::printf("ggml-cuda8/backend: graph_compute node %d %s\n", i, opname);
        const bool timing = cuda8_timing_enabled();
        double op_t0 = 0.0;
        if (timing) { cudaDeviceSynchronize(); op_t0 = cuda8_now_ms(); }
        if (ggml_cuda8_ggml_backend_dispatch_op(backend, ctx, cuda8_op, dispatch_src0, dispatch_src1, dispatch_dst) != 0) {
            std::fprintf(stderr, "ggml-cuda8/backend graph_compute: dispatch failed at node %d op=%s\n", i, opname);
            ggml_cuda8_context_destroy(ctx);
            return (enum ggml_status) -1;
        }
        if (timing) {
            cudaDeviceSynchronize();
            cuda8_timing_state & ts = cuda8_timing();
            ts.op_ms[opname]    += cuda8_now_ms() - op_t0;
            ts.op_count[opname] += 1;
        }
    }
    ggml_cuda8_context_destroy(ctx);
    if (cuda8_timing_enabled()) {
        cuda8_timing_state & ts = cuda8_timing();
        ts.graph_calls += 1;
        static bool reg = false;
        if (!reg) { std::atexit(cuda8_timing_report); reg = true; }
    }
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
    // G-fix: refuse immediately if a previous fatal CUDA error already
    // poisoned the device. Without this check, a dispatch issued after
    // such a fault would either silently no-op, return a confusing
    // unrelated CUDA error from whatever call happens to touch the
    // driver first (cudaSetDevice inside the buffer layer, an upload,
    // etc.), or in the worst case appear to "succeed" while operating on
    // an already-corrupted context. Checked here rather than only in
    // cuda8_backend_graph_compute() so that direct callers of this
    // function (outside the graph_compute path) get the same protection.
    if (g_cuda8_device_poisoned.load(std::memory_order_acquire)) {
        std::fprintf(stderr,
            "ggml-cuda8/backend-dispatch: refusing op=%d -- device was marked "
            "poisoned by a previous fatal CUDA error (see the earlier "
            "'synchronize FAILED' / 'marked POISONED' log). Restart the "
            "process to recover the CUDA8 device.\n", op);
        return -1;
    }
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
