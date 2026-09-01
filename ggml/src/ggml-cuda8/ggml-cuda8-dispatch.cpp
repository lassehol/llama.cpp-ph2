
// ggml/src/ggml-cuda8/ggml-cuda8-dispatch.cpp
//
// Dispatcher:
//   MUL_MAT Q8_0xF32 vec
//   CPY_F32
//   ADD_F32
//   scalar ADD/MUL
//   REDUCE_SUM_ROWS_F32
//   REDUCE_MAX_ROWS_F32
#include "ggml-cuda8-dispatch.h"
#include <cstring>
#include "ggml-cuda8-cpy.h"
#include "ggml-cuda8-add.h"
#include "ggml-cuda8-scalar.h"
#include "ggml-cuda8-reduce.h"
#include "ggml-cuda8-softmax.h"
#include "ggml-cuda8-backend-buffer.h"
#include "ggml-cuda8-ggml-buffer.h"
#include "q8_0-mmv.cuh"
#include <cstdio>
#include "ggml-cuda8-mulmat-f32.h"
#include "ggml-cuda8-softmax-ext.h"
// Forward declarations for K-quant dispatch functions
static int supported_mul_mat_q4k_f32(const struct ggml_cuda8_context*, const struct ggml_tensor*, const struct ggml_tensor*, const struct ggml_tensor*);
static int supported_mul_mat_q6k_f32(const struct ggml_cuda8_context*, const struct ggml_tensor*, const struct ggml_tensor*, const struct ggml_tensor*);
static int supported_get_rows_q4k(const struct ggml_cuda8_context*, const struct ggml_tensor*, const struct ggml_tensor*, const struct ggml_tensor*);
static int supported_get_rows_q6k(const struct ggml_cuda8_context*, const struct ggml_tensor*, const struct ggml_tensor*, const struct ggml_tensor*);
static int exec_mul_mat_q4k_f32(struct ggml_cuda8_context*, const struct ggml_tensor*, const struct ggml_tensor*, struct ggml_tensor*);
static int exec_mul_mat_q6k_f32(struct ggml_cuda8_context*, const struct ggml_tensor*, const struct ggml_tensor*, struct ggml_tensor*);
static int exec_get_rows_q4k(struct ggml_cuda8_context*, const struct ggml_tensor*, const struct ggml_tensor*, struct ggml_tensor*);
static int exec_get_rows_q6k(struct ggml_cuda8_context*, const struct ggml_tensor*, const struct ggml_tensor*, struct ggml_tensor*);
static int check_tensor_ptrs(
    const struct ggml_tensor * a,
    const struct ggml_tensor * b,
    const struct ggml_tensor * c
) {
    if (a == NULL || b == NULL || c == NULL) {
        std::fprintf(stderr, "ggml-cuda8/dispatch: null tensor pointer\n");
        return -1;
    }
    if (a->data == NULL || b->data == NULL || c->data == NULL) {
        std::fprintf(stderr, "ggml-cuda8/dispatch: null tensor data\n");
        return -1;
    }
    return 0;
}
// G-fix: debug-only residency diagnostic for the K-quant MUL_MAT/GET_ROWS
// paths. Unlike Q8_0/ADD/scalar, exec_mul_mat_q4k_f32/exec_get_rows_q4k
// (and their Q6_K equivalents) pass src0/src1/dst pointers straight into
// the kernel launcher with no host<->device staging. That is the correct
// design for weight/activation tensors under normal graph_compute -- the
// ggml scheduler guarantees CUDA8-buffer residency for any tensor it
// assigns to this backend -- but it leaves no safety net if that
// invariant is ever violated (e.g. a direct dispatch call in a test that
// bypasses graph_compute, or a scheduler bug that assigns a host-backed
// tensor here). A host pointer reaching the kernel launch faults
// asynchronously on the device and typically only surfaces at the next
// unrelated cudaDeviceSynchronize()/cudaMemcpy(), which makes the real
// cause hard to trace back.
//
// This is intentionally a loud diagnostic, not a functional gate: it
// does not change supported()'s return value, so it will not reject
// tensors that are legitimately exercised outside graph_compute (e.g.
// smoke tests that allocate CUDA8 buffers via the lower-level
// ggml_cuda8_buffer_malloc() API directly rather than through the GGML
// buffer wrapper's residency registry). Only compiled into debug builds.
static void cuda8_kquant_debug_check_resident(const char * label, const struct ggml_tensor * t) {
#ifndef NDEBUG
    if (t == NULL || t->data == NULL) return;
    const size_t nbytes = ggml_nbytes(t);
    ggml_backend_buffer_t owner = NULL;
    size_t offset = 0;
    if (!ggml_cuda8_ggml_tensor_is_device_resident(t, nbytes, &owner, &offset)) {
        std::fprintf(stderr,
            "ggml-cuda8/dispatch: WARNING: %s (data=%p, %zu bytes) is not "
            "registered as CUDA8-device-resident -- K-quant kernels assume "
            "device residency and do not stage through host<->device "
            "copies; if this tensor is actually host memory, the kernel "
            "launch will fault\n",
            label, (const void *) t->data, nbytes);
    }
#else
    (void) label;
    (void) t;
#endif
}
const char * ggml_cuda8_op_name(int op_id) {
    switch (op_id) {
        case GGML_CUDA8_OP_MUL_MAT_Q8_0_F32_VEC: return "MUL_MAT_Q8_0xF32_VEC";
        case GGML_CUDA8_OP_CPY_F32:              return "CPY_F32";
        case GGML_CUDA8_OP_ADD_F32:              return "ADD_F32";
        case GGML_CUDA8_OP_ADD_SCALAR_F32:       return "ADD_SCALAR_F32";
        case GGML_CUDA8_OP_MUL_SCALAR_F32:       return "MUL_SCALAR_F32";
        case GGML_CUDA8_OP_REDUCE_SUM_ROWS_F32:  return "REDUCE_SUM_ROWS_F32";
        case GGML_CUDA8_OP_REDUCE_MAX_ROWS_F32:  return "REDUCE_MAX_ROWS_F32";
        case GGML_CUDA8_OP_SOFTMAX_ROWS_F32:     return "SOFTMAX_ROWS_F32";
        case GGML_CUDA8_OP_RMS_NORM_F32:        return "RMS_NORM_F32";
        case GGML_CUDA8_OP_MUL_F32:              return "MUL_F32";
        case GGML_CUDA8_OP_MUL_BROADCAST_F32: return "MUL_BROADCAST_F32";
        case GGML_CUDA8_OP_ROPE_F32:             return "ROPE_F32";
        case GGML_CUDA8_OP_CONT_F32:             return "CONT_F32";
        case GGML_CUDA8_OP_DIAG_MASK_INF_F32:  return "DIAG_MASK_INF_F32";
        case GGML_CUDA8_OP_GET_ROWS_F32:       return "GET_ROWS_F32";
        case GGML_CUDA8_OP_MUL_MAT_Q4_K_F32: return "MUL_MAT_Q4_KxF32";
        case GGML_CUDA8_OP_MUL_MAT_Q6_K_F32: return "MUL_MAT_Q6_KxF32";
        case GGML_CUDA8_OP_GET_ROWS_Q4_K:    return "GET_ROWS_Q4_K";
        case GGML_CUDA8_OP_GET_ROWS_Q6_K:    return "GET_ROWS_Q6_K";
        case GGML_CUDA8_OP_SWIGLU_F32:       return "SWIGLU_F32";
		case GGML_CUDA8_OP_MUL_MAT_F32_F32: return "MUL_MAT_F32xF32";
        case GGML_CUDA8_OP_SET_ROWS_F32:     return "SET_ROWS_F32";
		case GGML_CUDA8_OP_SOFTMAX_EXT_F32: return "SOFTMAX_EXT_F32";
        default:                                  return "UNKNOWN";
    }
}
static int supported_mul_mat_q8_0_f32_vec(
    const struct ggml_cuda8_context * ctx,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst
) {
    (void) ctx;
    if (check_tensor_ptrs(src0, src1, dst) != 0) return 0;
    if (src0->type != GGML_TYPE_Q8_0) return 0;
    if (src1->type != GGML_TYPE_F32)  return 0;
    if (dst->type  != GGML_TYPE_F32)  return 0;
    const int64_t cols = src0->ne[0];
    const int64_t rows = src0->ne[1];
    if (cols <= 0 || rows <= 0) return 0;
    if ((cols % GGML_CUDA8_QK8_0) != 0) return 0;
    // src1 must have matching cols; allow multi-token (ne[1] >= 1)
    if (src1->ne[0] != cols) return 0;
    // dst rows must match weight matrix rows
    if (dst->ne[0] != rows) return 0;
    // Basic stride sanity
    if (src1->nb[0] != sizeof(float)) return 0;
    if (dst->nb[0]  != sizeof(float)) return 0;
    return 1;
}
static int exec_mul_mat_q8_0_f32_vec(
    struct ggml_cuda8_context * ctx,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct ggml_tensor * dst
) {
    const int rows = (int) src0->ne[1];   // output features
    const int cols = (int) src0->ne[0];   // input features
    const int n_tokens = (int) src1->ne[1]; // batch size (1 for single token)
    const int blocks_per_row =
        (cols + GGML_CUDA8_QK8_0 - 1) / GGML_CUDA8_QK8_0;
    const size_t bytes_src0 =
        (size_t) rows * (size_t) blocks_per_row * sizeof(ggml_cuda8_q8_0_block);
    const size_t bytes_vec =
        (size_t) cols * sizeof(float);
    const size_t bytes_out =
        (size_t) rows * sizeof(float);
    // Weight matrix: upload once (or use device-resident pointer directly)
    ggml_cuda8_backend_buffer * b0 = NULL;
    ggml_cuda8_backend_buffer * b1 = NULL;
    ggml_cuda8_backend_buffer * bd = NULL;
    int rc = 0;
    if (ggml_cuda8_context_alloc_buffer(ctx, bytes_src0, &b0) != 0) return -1;
    if (ggml_cuda8_context_alloc_buffer(ctx, bytes_vec, &b1) != 0) {
        ggml_cuda8_backend_buffer_free(b0);
        return -1;
    }
    if (ggml_cuda8_context_alloc_buffer(ctx, bytes_out, &bd) != 0) {
        ggml_cuda8_backend_buffer_free(b0);
        ggml_cuda8_backend_buffer_free(b1);
        return -1;
    }
    // Upload weight matrix once
    if (ggml_cuda8_backend_buffer_upload(b0, 0, src0->data, bytes_src0) != 0) {
        ggml_cuda8_backend_buffer_free(b0);
        ggml_cuda8_backend_buffer_free(b1);
        ggml_cuda8_backend_buffer_free(bd);
        return -1;
    }
    // Process each token vector
    for (int t = 0; t < n_tokens && rc == 0; ++t) {
        const char * src1_ptr = (const char *) src1->data + (size_t) t * bytes_vec;
        char * dst_ptr = (char *) dst->data + (size_t) t * bytes_out;
        if (ggml_cuda8_backend_buffer_upload(b1, 0, src1_ptr, bytes_vec) != 0) { rc = -1; break; }
        if (ggml_cuda8_backend_buffer_clear(bd, 0) != 0) { rc = -1; break; }
        rc = ggml_cuda8_context_mul_mat_q8_0_f32(ctx, b0, b1, bd, rows, cols);
        if (rc != 0) break;
        if (ggml_cuda8_backend_buffer_download(bd, 0, dst_ptr, bytes_out) != 0) { rc = -1; break; }
    }
    ggml_cuda8_backend_buffer_free(b0);
    ggml_cuda8_backend_buffer_free(b1);
    ggml_cuda8_backend_buffer_free(bd);
    return rc;
}
// -- G24A: RMS_NORM_F32 helpers -----------------------------------------------
extern "C" int ggml_cuda8_op_rms_norm_f32(
        const float * x, float * y,
        int nrows, int ncols, float eps);
static int ggml_cuda8_supported_rms_norm_f32(
        const struct ggml_cuda8_context * ctx,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * dst) {
    (void) ctx;
    if (src0 == NULL || dst == NULL) return 0;
    if (src0->type != GGML_TYPE_F32) return 0;
    if (dst->type  != GGML_TYPE_F32) return 0;
    return 1;
}
static int ggml_cuda8_exec_rms_norm_f32(
        struct ggml_cuda8_context * ctx,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    (void) ctx;
    float eps;
    std::memcpy(&eps, dst->op_params, sizeof(float));
    const int ncols = (int) src0->ne[0];
    const int nrows = (int)(src0->ne[1] * src0->ne[2] * src0->ne[3]);
    return ggml_cuda8_op_rms_norm_f32(
        (const float *) src0->data,
        (float *)       dst->data,
        nrows, ncols, eps);
}
// -- G26A: MUL_F32 element-wise helpers ---------------------------------------
extern "C" int ggml_cuda8_mul_f32_launch(
        const float * a, const float * b, float * c, int n);
extern "C" int ggml_cuda8_mul_broadcast_f32_launch(
        const float * a, const float * b, float * c, int n_total, int n_repeat);
static int ggml_cuda8_supported_mul_f32(
        const struct ggml_cuda8_context * ctx,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        const struct ggml_tensor * dst) {
    (void) ctx;
    if (src0 == NULL || src1 == NULL || dst == NULL) return 0;
    if (src0->type != GGML_TYPE_F32) return 0;
    if (src1->type != GGML_TYPE_F32) return 0;
    if (dst->type  != GGML_TYPE_F32) return 0;
    return 1;
}
static int ggml_cuda8_exec_mul_f32(
        struct ggml_cuda8_context * ctx,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    (void) ctx;
    const int n = (int)(src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
    return ggml_cuda8_mul_f32_launch(
        (const float *) src0->data,
        (const float *) src1->data,
        (float *)       dst->data,
        n);
}
// -- G28A: ROPE_F32 helpers ---------------------------------------------------
// -- G43: SET_ROWS -----------------------------------------------------------
extern "C" int ggml_cuda8_op_set_rows_f32(
        const void * src0, const void * idx, void * dst,
        int nc, int nr, int ne02, int ne03,
        int ne11, int ne12, int ne1,
        size_t nb01, size_t nb02, size_t nb03,
        size_t nb10, size_t nb11, size_t nb12,
        size_t nb1,  size_t nb2,  size_t nb3);
static int ggml_cuda8_supported_set_rows_f32(
        const struct ggml_cuda8_context * ctx,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        const struct ggml_tensor * dst) {
    (void) ctx;
    if (src0 == NULL || src1 == NULL || dst == NULL) return 0;
    // F16 destinations need the G49 store path; until then the KV cache must
    // be F32, i.e. --cache-type-k/v f32.
    if (src0->type != GGML_TYPE_F32) return 0;
    if (dst->type  != GGML_TYPE_F32) return 0;
    // llama-kv-cache.cpp creates the index tensor as I64.
    if (src1->type != GGML_TYPE_I64) return 0;
    if (dst->ne[0] != src0->ne[0]) return 0;
    if (dst->ne[2] != src0->ne[2]) return 0;
    if (dst->ne[3] != src0->ne[3]) return 0;
    // Index broadcast requirements from the CPU reference.
    if (src1->ne[1] == 0 || src1->ne[2] == 0)      return 0;
    if (src0->ne[2] % src1->ne[1] != 0)            return 0;
    if (src0->ne[3] % src1->ne[2] != 0)            return 0;
    if (src1->ne[0] < src0->ne[1])                 return 0;
    return 1;
}
static int ggml_cuda8_exec_set_rows_f32(
        struct ggml_cuda8_context * ctx,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    (void) ctx;
    return ggml_cuda8_op_set_rows_f32(
        src0->data, src1->data, dst->data,
        (int) src0->ne[0],   // nc
        (int) src0->ne[1],   // nr
        (int) src0->ne[2],
        (int) src0->ne[3],
        (int) src1->ne[1],
        (int) src1->ne[2],
        (int) dst->ne[1],
        src0->nb[1], src0->nb[2], src0->nb[3],
        src1->nb[0], src1->nb[1], src1->nb[2],
        dst->nb[1],  dst->nb[2],  dst->nb[3]);
}
// -- G40: SWIGLU -------------------------------------------------------------
extern "C" int ggml_cuda8_op_swiglu_f32(
        const float * src0, const float * src1, float * dst,
        int nc, int nrows,
        int src0_stride, int src1_stride, int dst_stride,
        int swapped);
// Row stride in floats. ggml guarantees ggml_is_contiguous_1 here (rows
// contiguous, row spacing arbitrary), so nb[1] is the stride that matters.
static int cuda8_row_stride_f32(const struct ggml_tensor * t) {
    return (int) (t->nb[1] / sizeof(float));
}
static int ggml_cuda8_supported_swiglu_f32(
        const struct ggml_cuda8_context * ctx,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        const struct ggml_tensor * dst) {
    (void) ctx;
    if (src0 == NULL || dst == NULL) return 0;
    if (src0->type != GGML_TYPE_F32) return 0;
    if (dst->type  != GGML_TYPE_F32) return 0;
    if (!ggml_is_contiguous_1(src0)) return 0;
    if (!ggml_is_contiguous_1(dst))  return 0;
    if (src1 != NULL) {
        if (src1->type != GGML_TYPE_F32) return 0;
        if (!ggml_is_contiguous_1(src1)) return 0;
        if (!ggml_are_same_shape(src0, src1)) return 0;
    }
    // Output width: full row for the split form, half for the halves form.
    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    if (nc <= 0) return 0;
    if (dst->ne[0] != nc) return 0;
    if (ggml_nrows(dst) != ggml_nrows(src0)) return 0;
    if (!src1 && (src0->ne[0] % 2) != 0) return 0;
    // G-fix: op_params validation moved here from exec_swiglu_f32 so that
    // ggml_cuda8_dispatch_supported() (and therefore any scheduler
    // supports_op hook built on top of it) accurately reflects what this
    // backend can execute. Previously only tensor shape/type were checked
    // here, while the GLU-variant check lived exclusively in the exec
    // path -- a non-SWIGLU GLU op (GEGLU, SWIGLU_OAI, etc.) would pass
    // "supported" and only fail once dispatch actually started, instead of
    // letting the scheduler fall back to another backend.
    // op_params[0] = ggml_glu_op. Read directly rather than via
    // ggml_get_op_params_i32, which lives in ggml-impl.h and is not
    // included here.
    int32_t glu_op;
    std::memcpy(&glu_op, dst->op_params, sizeof(glu_op));
    // Only plain SWIGLU is implemented. SWIGLU_OAI carries alpha/limit in
    // op_params[2..3] which this kernel does not apply; the other GLU ops
    // use different activations entirely.
    if (glu_op != GGML_GLU_OP_SWIGLU) return 0;
    return 1;
}
static int ggml_cuda8_exec_swiglu_f32(
        struct ggml_cuda8_context * ctx,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    (void) ctx;
    // op_params[0] = ggml_glu_op, op_params[1] = swapped (ggml_glu_impl).
    // Read directly rather than via ggml_get_op_params_i32, which lives in
    // ggml-impl.h and is not included here.
    //
    // G-fix: the glu_op check itself now lives in
    // ggml_cuda8_supported_swiglu_f32() and is the authoritative gate.
    // dispatch_execute() always calls dispatch_supported() first, so
    // reaching this point with glu_op != GGML_GLU_OP_SWIGLU would mean the
    // caller bypassed the supported() gate -- kept as a defensive check
    // rather than removed outright.
    int32_t glu_params[2];
    std::memcpy(glu_params, dst->op_params, sizeof(glu_params));
    const int32_t glu_op  = glu_params[0];
    const int32_t swapped = glu_params[1];
    if (glu_op != GGML_GLU_OP_SWIGLU) {
        std::fprintf(stderr, "ggml-cuda8/swiglu: unsupported glu op %d (unreachable: supported() should have refused this)\n", (int) glu_op);
        return -1;
    }
    const int nc    = (int) (src1 ? src0->ne[0] : src0->ne[0] / 2);
    const int nrows = (int) ggml_nrows(src0);
    return ggml_cuda8_op_swiglu_f32(
        (const float *) src0->data,
        src1 ? (const float *) src1->data : NULL,
        (float *) dst->data,
        nc, nrows,
        cuda8_row_stride_f32(src0),
        src1 ? cuda8_row_stride_f32(src1) : 0,
        cuda8_row_stride_f32(dst),
        (int) swapped);
}
// G45: gained an int mode parameter (0 = NORMAL, 2 = NEOX).
extern "C" int ggml_cuda8_op_rope_f32(
        const float * x, float * dst, const int * pos,
        int ne0, int ne1, int ne2, int ne3,
        int n_dims, int mode, float freq_base, float freq_scale);
static int ggml_cuda8_supported_rope_f32(
        const struct ggml_cuda8_context * ctx,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        const struct ggml_tensor * dst) {
    (void) ctx;
    if (src0 == NULL || src1 == NULL || dst == NULL) return 0;
    if (src0->type != GGML_TYPE_F32) return 0;
    if (src1->type != GGML_TYPE_I32) return 0;
    if (dst->type  != GGML_TYPE_F32) return 0;
    // G-fix: op_params validation moved here from exec_rope_f32 so that
    // ggml_cuda8_dispatch_supported() accurately reflects what this
    // backend can execute. Previously mode/ext_factor/attn_factor/
    // freq_factors were only checked once exec had already started,
    // which meant a scheduler relying on supports_op (via this function)
    // could still assign a YaRN-scaled or otherwise unsupported ROPE node
    // to this backend and only discover the problem when the whole
    // graph_compute() call aborted.
    int32_t op_params[15];
    std::memcpy(op_params, dst->op_params, sizeof(op_params));
    const int mode = op_params[2];
    float ext_factor, attn_factor;
    std::memcpy(&ext_factor,  &op_params[7], sizeof(float));
    std::memcpy(&attn_factor, &op_params[8], sizeof(float));
    // G45: NORMAL and NEOX are implemented; everything else must be refused.
    if (mode != 0 && mode != 2) return 0;
    // YaRN scaling is not applied by the kernel.
    if (ext_factor != 0.0f) return 0;
    // A non-default attn_factor is not applied by the kernel.
    if (attn_factor != 1.0f) return 0;
    // freq_factors (src[2] on the graph node) are not applied by the kernel.
    if (dst->src[2] != NULL) return 0;
    return 1;
}
static int ggml_cuda8_exec_rope_f32(
        struct ggml_cuda8_context * ctx,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    (void) ctx;
    // Extract op_params from dst (the graph node)
    int32_t op_params[15];
    std::memcpy(op_params, dst->op_params, sizeof(op_params));
    int n_dims = op_params[1];
    int mode   = op_params[2];
    float freq_base, freq_scale;
    std::memcpy(&freq_base,   &op_params[5], sizeof(float));
    std::memcpy(&freq_scale,  &op_params[6], sizeof(float));
    // G-fix: mode/ext_factor/attn_factor/freq_factors are now validated in
    // ggml_cuda8_supported_rope_f32(), which dispatch_execute() always
    // calls before reaching here. Kept as a defensive re-check rather than
    // removed outright, since this mirrors the "defence in depth against
    // the G37 class of bug" reasoning used elsewhere in this file:
    // parameters the kernel does not read must never be silently dropped,
    // even if the scheduler bypasses supports_op.
    if (mode != 0 && mode != 2) {
        std::fprintf(stderr, "ggml-cuda8/rope: unsupported mode=%d (unreachable: supported() should have refused this)\n", mode);
        return -1;
    }
    float ext_factor, attn_factor;
    std::memcpy(&ext_factor,  &op_params[7], sizeof(float));
    std::memcpy(&attn_factor, &op_params[8], sizeof(float));
    if (ext_factor != 0.0f) {
        std::fprintf(stderr, "ggml-cuda8/rope: YaRN unsupported (ext_factor=%.3f) (unreachable: supported() should have refused this)\n",
                     (double) ext_factor);
        return -1;
    }
    if (attn_factor != 1.0f) {
        std::fprintf(stderr, "ggml-cuda8/rope: attn_factor=%.3f not applied by the kernel (unreachable: supported() should have refused this)\n",
                     (double) attn_factor);
        return -1;
    }
    if (dst->src[2] != NULL) {
        std::fprintf(stderr, "ggml-cuda8/rope: freq_factors (src[2]) not applied by the kernel (unreachable: supported() should have refused this)\n");
        return -1;
    }
    int ne0 = (int) src0->ne[0];
    int ne1 = (int) src0->ne[1];
    int ne2 = (int) src0->ne[2];
    int ne3 = (int) src0->ne[3];
    return ggml_cuda8_op_rope_f32(
        (const float *) src0->data,
        (float *)       dst->data,
        (const int *)   src1->data,
        ne0, ne1, ne2, ne3,
        n_dims, mode, freq_base, freq_scale);
}
// -- G30A/G55: CONT_F32 (strided gather) --------------------------------------
// G55: CONT exists specifically to make a NON-contiguous tensor contiguous
// (post-permute, in attention). The old implementation flat-copied src0->data
// as one contiguous byte run, ignoring src0->nb[], which silently scrambled
// every permuted input. This version gathers through src0's real strides into
// a packed dst.
extern "C" int ggml_cuda8_cont_f32_launch(
        const void * src0, float * dst,
        int ne0, int ne1, int ne2, int ne3,
        size_t nb0, size_t nb1, size_t nb2, size_t nb3);

static int ggml_cuda8_supported_cont_f32(
        const struct ggml_cuda8_context * ctx,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * dst) {
    (void) ctx;
    if (src0 == NULL || dst == NULL) return 0;
    if (src0->type != GGML_TYPE_F32) return 0;
    if (dst->type  != GGML_TYPE_F32) return 0;
    // dst must be contiguous (ggml guarantees this for CONT output); the
    // strided-gather kernel writes it packed.
    if (dst->nb[0] != sizeof(float)) return 0;
    return 1;
}

static int ggml_cuda8_exec_cont_f32(
        struct ggml_cuda8_context * ctx,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    (void) ctx;
    return ggml_cuda8_cont_f32_launch(
        src0->data, (float *) dst->data,
        (int) src0->ne[0], (int) src0->ne[1], (int) src0->ne[2], (int) src0->ne[3],
        src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
}
// -- G31A: DIAG_MASK_INF_F32 helpers -----------------------------------------
extern "C" int ggml_cuda8_op_diag_mask_inf_f32(
        const float * x, float * dst,
        int ncols, int nrows, int rows_per_channel, int n_past);
static int ggml_cuda8_supported_diag_mask_inf_f32(
        const struct ggml_cuda8_context * ctx,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * dst) {
    (void) ctx;
    if (src0 == NULL || dst == NULL) return 0;
    if (src0->type != GGML_TYPE_F32) return 0;
    if (dst->type  != GGML_TYPE_F32) return 0;
    return 1;
}
static int ggml_cuda8_exec_diag_mask_inf_f32(
        struct ggml_cuda8_context * ctx,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    (void) ctx;
    int32_t n_past;
    std::memcpy(&n_past, dst->op_params, sizeof(int32_t));
    const int ncols = (int) src0->ne[0];
    const int nrows = (int)(src0->ne[1] * src0->ne[2] * src0->ne[3]);
    const int rows_per_channel = (int) src0->ne[1];
    return ggml_cuda8_op_diag_mask_inf_f32(
        (const float *) src0->data,
        (float *)       dst->data,
        ncols, nrows, rows_per_channel, n_past);
}
// -- G33A: GET_ROWS_F32 helpers -----------------------------------------------
extern "C" int ggml_cuda8_op_get_rows_f32(
        const float * src0, const int * src1, float * dst,
        int ne00, int n_tokens);
static int ggml_cuda8_supported_get_rows_f32(
        const struct ggml_cuda8_context * ctx,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        const struct ggml_tensor * dst) {
    (void) ctx;
    if (src0 == NULL || src1 == NULL || dst == NULL) return 0;
    if (src0->type != GGML_TYPE_F32) return 0;
    if (src1->type != GGML_TYPE_I32) return 0;
    if (dst->type  != GGML_TYPE_F32) return 0;
    return 1;
}
static int ggml_cuda8_exec_get_rows_f32(
        struct ggml_cuda8_context * ctx,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    (void) ctx;
    const int ne00 = (int) src0->ne[0];
    const int n_tokens = (int) src1->ne[0];
    return ggml_cuda8_op_get_rows_f32(
        (const float *) src0->data,
        (const int *)   src1->data,
        (float *)       dst->data,
        ne00, n_tokens);
}
int ggml_cuda8_dispatch_supported(
    const struct ggml_cuda8_context * ctx,
    int op_id,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst
) {
    if (ctx == NULL) return 0;
    switch (op_id) {
        case GGML_CUDA8_OP_CPY_F32:
            return ggml_cuda8_supported_cpy_f32(ctx, src0, dst);
        case GGML_CUDA8_OP_ADD_F32:
            return ggml_cuda8_supported_add_f32(ctx, src0, src1, dst);
        case GGML_CUDA8_OP_ADD_SCALAR_F32:
            return ggml_cuda8_supported_scalar_f32(ctx, src0, src1, dst);
        case GGML_CUDA8_OP_MUL_SCALAR_F32:
            return ggml_cuda8_supported_scalar_f32(ctx, src0, src1, dst);
        case GGML_CUDA8_OP_REDUCE_SUM_ROWS_F32:
            return ggml_cuda8_supported_reduce_sum_rows_f32(ctx, src0, dst);
        case GGML_CUDA8_OP_REDUCE_MAX_ROWS_F32:
            return ggml_cuda8_supported_reduce_max_rows_f32(ctx, src0, dst);
        case GGML_CUDA8_OP_SOFTMAX_ROWS_F32:
            return ggml_cuda8_supported_softmax_rows_f32(ctx, src0, dst);
        case GGML_CUDA8_OP_MUL_MAT_Q8_0_F32_VEC:
            return supported_mul_mat_q8_0_f32_vec(ctx, src0, src1, dst);
        case GGML_CUDA8_OP_RMS_NORM_F32:
            return ggml_cuda8_supported_rms_norm_f32(ctx, src0, dst);
        case GGML_CUDA8_OP_MUL_F32:
        case GGML_CUDA8_OP_MUL_BROADCAST_F32:
            return ggml_cuda8_supported_mul_f32(ctx, src0, src1, dst);
        case GGML_CUDA8_OP_ROPE_F32:
            return ggml_cuda8_supported_rope_f32(ctx, src0, src1, dst);
        case GGML_CUDA8_OP_CONT_F32:
            return ggml_cuda8_supported_cont_f32(ctx, src0, dst);
        case GGML_CUDA8_OP_DIAG_MASK_INF_F32:
            return ggml_cuda8_supported_diag_mask_inf_f32(ctx, src0, dst);
        case GGML_CUDA8_OP_GET_ROWS_F32:
            return ggml_cuda8_supported_get_rows_f32(ctx, src0, src1, dst);
        case GGML_CUDA8_OP_MUL_MAT_Q4_K_F32:
            return supported_mul_mat_q4k_f32(ctx, src0, src1, dst);
        case GGML_CUDA8_OP_MUL_MAT_Q6_K_F32:
            return supported_mul_mat_q6k_f32(ctx, src0, src1, dst);
        case GGML_CUDA8_OP_GET_ROWS_Q4_K:
            return supported_get_rows_q4k(ctx, src0, src1, dst);
        case GGML_CUDA8_OP_GET_ROWS_Q6_K:
            return supported_get_rows_q6k(ctx, src0, src1, dst);
        case GGML_CUDA8_OP_SWIGLU_F32:
            return ggml_cuda8_supported_swiglu_f32(ctx, src0, src1, dst);
        case GGML_CUDA8_OP_SET_ROWS_F32:
            return ggml_cuda8_supported_set_rows_f32(ctx, src0, src1, dst);
		case GGML_CUDA8_OP_MUL_MAT_F32_F32:
            return ggml_cuda8_supported_mul_mat_f32_f32(ctx, src0, src1, dst);
		case GGML_CUDA8_OP_SOFTMAX_EXT_F32:
			return ggml_cuda8_supported_softmax_ext_f32(ctx, src0, src1, dst);
        default:
            return 0;
    }
}
int ggml_cuda8_dispatch_execute(
    struct ggml_cuda8_context * ctx,
    int op_id,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct ggml_tensor * dst
) {
    if (ctx == NULL) {
        std::fprintf(stderr, "ggml-cuda8/dispatch: null ctx\n");
        return -1;
    }
    if (!ggml_cuda8_dispatch_supported(ctx, op_id, src0, src1, dst)) {
        std::fprintf(stderr,
            "ggml-cuda8/dispatch: unsupported op=%s\n",
            ggml_cuda8_op_name(op_id));
        return -1;
    }
    switch (op_id) {
        case GGML_CUDA8_OP_CPY_F32:
            return ggml_cuda8_exec_cpy_f32(ctx, src0, dst);
        case GGML_CUDA8_OP_ADD_F32:
            return ggml_cuda8_exec_add_f32(ctx, src0, src1, dst);
        case GGML_CUDA8_OP_ADD_SCALAR_F32:
            return ggml_cuda8_exec_add_scalar_f32(ctx, src0, src1, dst);
        case GGML_CUDA8_OP_MUL_SCALAR_F32:
            return ggml_cuda8_exec_mul_scalar_f32(ctx, src0, src1, dst);
        case GGML_CUDA8_OP_REDUCE_SUM_ROWS_F32:
            return ggml_cuda8_exec_reduce_sum_rows_f32(ctx, src0, dst);
        case GGML_CUDA8_OP_REDUCE_MAX_ROWS_F32:
            return ggml_cuda8_exec_reduce_max_rows_f32(ctx, src0, dst);
        case GGML_CUDA8_OP_SOFTMAX_ROWS_F32:
            return ggml_cuda8_exec_softmax_rows_f32(ctx, src0, dst);
        case GGML_CUDA8_OP_MUL_MAT_Q8_0_F32_VEC:
            return exec_mul_mat_q8_0_f32_vec(ctx, src0, src1, dst);
        case GGML_CUDA8_OP_RMS_NORM_F32:
            return ggml_cuda8_exec_rms_norm_f32(ctx, src0, dst);
        case GGML_CUDA8_OP_MUL_F32:
            return ggml_cuda8_exec_mul_f32(ctx, src0, src1, dst);
        case GGML_CUDA8_OP_MUL_BROADCAST_F32: {
            const int n_total = (int)(src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
            const int n_repeat = (int)(src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3]);
            return ggml_cuda8_mul_broadcast_f32_launch(
                (const float *) src0->data,
                (const float *) src1->data,
                (float *)       dst->data,
                n_total, n_repeat);
        }
        case GGML_CUDA8_OP_ROPE_F32:
            return ggml_cuda8_exec_rope_f32(ctx, src0, src1, dst);
        case GGML_CUDA8_OP_CONT_F32:
            return ggml_cuda8_exec_cont_f32(ctx, src0, dst);
        case GGML_CUDA8_OP_DIAG_MASK_INF_F32:
            return ggml_cuda8_exec_diag_mask_inf_f32(ctx, src0, dst);
        case GGML_CUDA8_OP_GET_ROWS_F32:
            return ggml_cuda8_exec_get_rows_f32(ctx, src0, src1, dst);
        case GGML_CUDA8_OP_MUL_MAT_Q4_K_F32:
            return exec_mul_mat_q4k_f32(ctx, src0, src1, dst);
        case GGML_CUDA8_OP_MUL_MAT_Q6_K_F32:
            return exec_mul_mat_q6k_f32(ctx, src0, src1, dst);
        case GGML_CUDA8_OP_GET_ROWS_Q4_K:
            return exec_get_rows_q4k(ctx, src0, src1, dst);
        case GGML_CUDA8_OP_GET_ROWS_Q6_K:
            return exec_get_rows_q6k(ctx, src0, src1, dst);
        case GGML_CUDA8_OP_SWIGLU_F32:
            return ggml_cuda8_exec_swiglu_f32(ctx, src0, src1, dst);
		case GGML_CUDA8_OP_MUL_MAT_F32_F32:
            return ggml_cuda8_exec_mul_mat_f32_f32(ctx, src0, src1, dst);
        case GGML_CUDA8_OP_SET_ROWS_F32:
            return ggml_cuda8_exec_set_rows_f32(ctx, src0, src1, dst);
		case GGML_CUDA8_OP_SOFTMAX_EXT_F32:
			return ggml_cuda8_exec_softmax_ext_f32(ctx, src0, src1, dst);
        default:
            return -1;
    }
}
// -- K-quant: Q4_K and Q6_K support -----------------------------------------
extern "C" int ggml_cuda8_op_get_rows_q4k(const void*, const int*, float*, int, int, int, int);
extern "C" int ggml_cuda8_op_mul_mat_q4k_f32(const void*, const float*, float*, int, int, int, int, int);
extern "C" int ggml_cuda8_op_get_rows_q6k(const void*, const int*, float*, int, int, int, int);
extern "C" int ggml_cuda8_op_mul_mat_q6k_f32(const void*, const float*, float*, int, int, int, int, int);
// Q4_K MUL_MAT supported check
static int supported_mul_mat_q4k_f32(
        const struct ggml_cuda8_context * ctx,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        const struct ggml_tensor * dst) {
    (void) ctx;
    // G-fix: check_tensor_ptrs() also rejects NULL ->data, closing the gap
    // where a valid-but-unallocated tensor struct previously passed this
    // gate (only "!src0 || !src1 || !dst" was checked) and would have been
    // handed straight to the kernel launcher in exec_mul_mat_q4k_f32().
    if (check_tensor_ptrs(src0, src1, dst) != 0) return 0;
    if (src0->type != GGML_TYPE_Q4_K) return 0;
    if (src1->type != GGML_TYPE_F32)  return 0;
    if (dst->type  != GGML_TYPE_F32)  return 0;
    if (src0->ne[0] % 256 != 0) return 0;  // QK_K alignment
    cuda8_kquant_debug_check_resident("Q4_K MUL_MAT src0", src0);
    cuda8_kquant_debug_check_resident("Q4_K MUL_MAT src1", src1);
    cuda8_kquant_debug_check_resident("Q4_K MUL_MAT dst",  dst);
    return 1;
}
// Q6_K MUL_MAT supported check
static int supported_mul_mat_q6k_f32(
        const struct ggml_cuda8_context * ctx,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        const struct ggml_tensor * dst) {
    (void) ctx;
    // G-fix: see supported_mul_mat_q4k_f32() above -- check_tensor_ptrs()
    // additionally rejects NULL ->data on all three tensors.
    if (check_tensor_ptrs(src0, src1, dst) != 0) return 0;
    if (src0->type != GGML_TYPE_Q6_K) return 0;
    if (src1->type != GGML_TYPE_F32)  return 0;
    if (dst->type  != GGML_TYPE_F32)  return 0;
    if (src0->ne[0] % 256 != 0) return 0;
    cuda8_kquant_debug_check_resident("Q6_K MUL_MAT src0", src0);
    cuda8_kquant_debug_check_resident("Q6_K MUL_MAT src1", src1);
    cuda8_kquant_debug_check_resident("Q6_K MUL_MAT dst",  dst);
    return 1;
}
// Q4_K GET_ROWS supported check
static int supported_get_rows_q4k(
        const struct ggml_cuda8_context * ctx,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        const struct ggml_tensor * dst) {
    (void) ctx;
    // G-fix: see supported_mul_mat_q4k_f32() above -- check_tensor_ptrs()
    // additionally rejects NULL ->data on all three tensors.
    if (check_tensor_ptrs(src0, src1, dst) != 0) return 0;
    if (src0->type != GGML_TYPE_Q4_K) return 0;
    if (src1->type != GGML_TYPE_I32)  return 0;
    if (dst->type  != GGML_TYPE_F32)  return 0;
    cuda8_kquant_debug_check_resident("Q4_K GET_ROWS src0", src0);
    cuda8_kquant_debug_check_resident("Q4_K GET_ROWS src1", src1);
    cuda8_kquant_debug_check_resident("Q4_K GET_ROWS dst",  dst);
    return 1;
}
// Q6_K GET_ROWS supported check
static int supported_get_rows_q6k(
        const struct ggml_cuda8_context * ctx,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        const struct ggml_tensor * dst) {
    (void) ctx;
    // G-fix: see supported_mul_mat_q4k_f32() above -- check_tensor_ptrs()
    // additionally rejects NULL ->data on all three tensors.
    if (check_tensor_ptrs(src0, src1, dst) != 0) return 0;
    if (src0->type != GGML_TYPE_Q6_K) return 0;
    if (src1->type != GGML_TYPE_I32)  return 0;
    if (dst->type  != GGML_TYPE_F32)  return 0;
    cuda8_kquant_debug_check_resident("Q6_K GET_ROWS src0", src0);
    cuda8_kquant_debug_check_resident("Q6_K GET_ROWS src1", src1);
    cuda8_kquant_debug_check_resident("Q6_K GET_ROWS dst",  dst);
    return 1;
}
// Q4_K MUL_MAT execute
static int exec_mul_mat_q4k_f32(
        struct ggml_cuda8_context * ctx,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    (void) ctx;
    return ggml_cuda8_op_mul_mat_q4k_f32(
        src0->data, (const float *)src1->data, (float *)dst->data,
        (int)src0->ne[0], (int)src0->ne[1], (int)src1->ne[1],
        (int)src0->nb[1], (int)src1->nb[1]);
}
// Q6_K MUL_MAT execute
static int exec_mul_mat_q6k_f32(
        struct ggml_cuda8_context * ctx,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    (void) ctx;
    return ggml_cuda8_op_mul_mat_q6k_f32(
        src0->data, (const float *)src1->data, (float *)dst->data,
        (int)src0->ne[0], (int)src0->ne[1], (int)src1->ne[1],
        (int)src0->nb[1], (int)src1->nb[1]);
}
// Q4_K GET_ROWS execute
static int exec_get_rows_q4k(
        struct ggml_cuda8_context * ctx,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    (void) ctx;
    int n_tokens = (int)src1->ne[0];
    int ne00 = (int)src0->ne[0];
    int nb01 = (int)src0->nb[1];
    int nb1  = (int)(ne00 * sizeof(float));
    return ggml_cuda8_op_get_rows_q4k(
        src0->data, (const int *)src1->data, (float *)dst->data,
        ne00, n_tokens, nb01, nb1);
}
// Q6_K GET_ROWS execute
static int exec_get_rows_q6k(
        struct ggml_cuda8_context * ctx,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    (void) ctx;
    int n_tokens = (int)src1->ne[0];
    int ne00 = (int)src0->ne[0];
    int nb01 = (int)src0->nb[1];
    int nb1  = (int)(ne00 * sizeof(float));
    return ggml_cuda8_op_get_rows_q6k(
        src0->data, (const int *)src1->data, (float *)dst->data,
        ne00, n_tokens, nb01, nb1);
}
