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
#include "q8_0-mmv.cuh"

#include <cstdio>

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
        case GGML_CUDA8_OP_ROPE_F32:             return "ROPE_F32";
        case GGML_CUDA8_OP_CONT_F32:             return "CONT_F32";
        case GGML_CUDA8_OP_DIAG_MASK_INF_F32:  return "DIAG_MASK_INF_F32";
        case GGML_CUDA8_OP_GET_ROWS_F32:       return "GET_ROWS_F32";
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

    if (src1->ne[0] != cols) return 0;
    if (src1->ne[1] != 1)    return 0;

    if (dst->ne[0] != rows)  return 0;
    if (dst->ne[1] != 1)     return 0;

    if (src1->nb[0] != sizeof(float)) return 0;
    if (dst->nb[0]  != sizeof(float)) return 0;

    const int blocks_per_row =
        (int) ((cols + GGML_CUDA8_QK8_0 - 1) / GGML_CUDA8_QK8_0);

    const size_t block_sz = sizeof(ggml_cuda8_q8_0_block);

    if (src0->nb[0] != block_sz) return 0;
    if (src0->nb[1] != (size_t) blocks_per_row * block_sz) return 0;

    return 1;
}

static int exec_mul_mat_q8_0_f32_vec(
    struct ggml_cuda8_context * ctx,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct ggml_tensor * dst
) {
    const int rows = (int) src0->ne[1];
    const int cols = (int) src0->ne[0];

    const int blocks_per_row =
        (cols + GGML_CUDA8_QK8_0 - 1) / GGML_CUDA8_QK8_0;

    const size_t bytes_src0 =
        (size_t) rows * (size_t) blocks_per_row * sizeof(ggml_cuda8_q8_0_block);

    const size_t bytes_src1 =
        (size_t) cols * sizeof(float);

    const size_t bytes_dst =
        (size_t) rows * sizeof(float);

    ggml_cuda8_backend_buffer * b0 = NULL;
    ggml_cuda8_backend_buffer * b1 = NULL;
    ggml_cuda8_backend_buffer * bd = NULL;

    int rc = 0;

    if (ggml_cuda8_context_alloc_buffer(ctx, bytes_src0, &b0) != 0) return -1;

    if (ggml_cuda8_context_alloc_buffer(ctx, bytes_src1, &b1) != 0) {
        ggml_cuda8_backend_buffer_free(b0);
        return -1;
    }

    if (ggml_cuda8_context_alloc_buffer(ctx, bytes_dst, &bd) != 0) {
        ggml_cuda8_backend_buffer_free(b0);
        ggml_cuda8_backend_buffer_free(b1);
        return -1;
    }

    if (ggml_cuda8_backend_buffer_upload(b0, 0, src0->data, bytes_src0) != 0) rc = -1;
    if (rc == 0 && ggml_cuda8_backend_buffer_upload(b1, 0, src1->data, bytes_src1) != 0) rc = -1;
    if (rc == 0 && ggml_cuda8_backend_buffer_clear(bd, 0) != 0) rc = -1;

    if (rc == 0) {
        rc = ggml_cuda8_context_mul_mat_q8_0_f32(ctx, b0, b1, bd, rows, cols);
    }

    if (rc == 0) {
        if (ggml_cuda8_backend_buffer_download(bd, 0, dst->data, bytes_dst) != 0) {
            rc = -1;
        }
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
extern "C" int ggml_cuda8_op_rope_f32(
        const float * x, float * dst, const int * pos,
        int ne0, int ne1, int ne2, int ne3,
        int n_dims, float freq_base, float freq_scale);

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

    float freq_base, freq_scale, ext_factor;
    std::memcpy(&freq_base,   &op_params[5], sizeof(float));
    std::memcpy(&freq_scale,  &op_params[6], sizeof(float));
    std::memcpy(&ext_factor,  &op_params[7], sizeof(float));

    if (mode != 0 || ext_factor != 0.0f) {
        std::fprintf(stderr, "ggml-cuda8/rope: unsupported mode=%d ext=%.1f\n",
                     mode, (double)ext_factor);
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
        n_dims, freq_base, freq_scale);
}


// -- G30A: CONT_F32 helpers ---------------------------------------------------
extern "C" int ggml_cuda8_cpy_f32_d2d(
        const float * src, float * dst, size_t n_bytes);

static int ggml_cuda8_supported_cont_f32(
        const struct ggml_cuda8_context * ctx,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * dst) {
    (void) ctx;
    if (src0 == NULL || dst == NULL) return 0;
    if (src0->type != GGML_TYPE_F32) return 0;
    if (dst->type  != GGML_TYPE_F32) return 0;
    return 1;
}

static int ggml_cuda8_exec_cont_f32(
        struct ggml_cuda8_context * ctx,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    (void) ctx;
    const int n = (int)(src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3]);
    const size_t bytes = (size_t)n * sizeof(float);
    return ggml_cuda8_cpy_f32_d2d(
        (const float *) src0->data,
        (float *)       dst->data,
        bytes);
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
            return ggml_cuda8_supported_mul_f32(ctx, src0, src1, dst);

        case GGML_CUDA8_OP_ROPE_F32:
            return ggml_cuda8_supported_rope_f32(ctx, src0, src1, dst);

        case GGML_CUDA8_OP_CONT_F32:
            return ggml_cuda8_supported_cont_f32(ctx, src0, dst);

        case GGML_CUDA8_OP_DIAG_MASK_INF_F32:
            return ggml_cuda8_supported_diag_mask_inf_f32(ctx, src0, dst);

        case GGML_CUDA8_OP_GET_ROWS_F32:
            return ggml_cuda8_supported_get_rows_f32(ctx, src0, src1, dst);


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

        case GGML_CUDA8_OP_ROPE_F32:
            return ggml_cuda8_exec_rope_f32(ctx, src0, src1, dst);

        case GGML_CUDA8_OP_CONT_F32:
            return ggml_cuda8_exec_cont_f32(ctx, src0, dst);

        case GGML_CUDA8_OP_DIAG_MASK_INF_F32:
            return ggml_cuda8_exec_diag_mask_inf_f32(ctx, src0, dst);

        case GGML_CUDA8_OP_GET_ROWS_F32:
            return ggml_cuda8_exec_get_rows_f32(ctx, src0, src1, dst);


        default:
            return -1;
    }
}
