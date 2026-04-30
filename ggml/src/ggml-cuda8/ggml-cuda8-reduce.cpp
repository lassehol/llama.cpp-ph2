// ggml/src/ggml-cuda8/ggml-cuda8-reduce.cpp
// G11D: row-wise reduce sum/max with host-backed and CUDA8 device-resident paths.

#include "ggml-cuda8-reduce.h"
#include "ggml-cuda8-backend-buffer.h"
#include "ggml-cuda8-ggml-buffer.h"

#include <cstdio>

extern "C" int ggml_cuda8_reduce_sum_rows_f32_launch(const float * src, float * dst, int rows, int cols);
extern "C" int ggml_cuda8_reduce_max_rows_f32_launch(const float * src, float * dst, int rows, int cols);

static int tensor_is_contiguous_f32_matrix_2d(const ggml_tensor * t) {
    if (t == NULL || t->type != GGML_TYPE_F32 || t->data == NULL) return 0;
    if (t->ne[0] <= 0 || t->ne[1] <= 0 || t->ne[2] != 1 || t->ne[3] != 1) return 0;
    if (t->nb[0] != sizeof(float)) return 0;
    if (t->nb[1] != (size_t) t->ne[0] * sizeof(float)) return 0;
    return 1;
}

static int tensor_is_contiguous_f32_vector(const ggml_tensor * t) {
    if (t == NULL || t->type != GGML_TYPE_F32 || t->data == NULL) return 0;
    if (t->ne[0] <= 0 || t->ne[1] != 1 || t->ne[2] != 1 || t->ne[3] != 1) return 0;
    if (t->nb[0] != sizeof(float)) return 0;
    return 1;
}

int ggml_cuda8_supported_reduce_rows_f32(const ggml_cuda8_context * ctx, const ggml_tensor * src0, const ggml_tensor * dst) {
    (void) ctx;
    if (!tensor_is_contiguous_f32_matrix_2d(src0)) return 0;
    if (!tensor_is_contiguous_f32_vector(dst)) return 0;
    if (dst->ne[0] != src0->ne[1]) return 0;
    return 1;
}

int ggml_cuda8_supported_reduce_sum_rows_f32(const ggml_cuda8_context * ctx, const ggml_tensor * src0, const ggml_tensor * dst) {
    return ggml_cuda8_supported_reduce_rows_f32(ctx, src0, dst);
}

int ggml_cuda8_supported_reduce_max_rows_f32(const ggml_cuda8_context * ctx, const ggml_tensor * src0, const ggml_tensor * dst) {
    return ggml_cuda8_supported_reduce_rows_f32(ctx, src0, dst);
}

static int tensor_is_cuda8_resident(const ggml_tensor * t, size_t bytes) {
    ggml_backend_buffer_t owner = NULL;
    size_t offset = 0;
    return ggml_cuda8_ggml_tensor_is_device_resident(t, bytes, &owner, &offset);
}

static int exec_device(const ggml_tensor * src0, ggml_tensor * dst, int is_max, int rows, int cols) {
    if (is_max) return ggml_cuda8_reduce_max_rows_f32_launch((const float *) src0->data, (float *) dst->data, rows, cols);
    return ggml_cuda8_reduce_sum_rows_f32_launch((const float *) src0->data, (float *) dst->data, rows, cols);
}

static int exec_host(ggml_cuda8_context * ctx, const ggml_tensor * src0, ggml_tensor * dst, int is_max, int rows, int cols) {
    const size_t bytes_src = (size_t) rows * (size_t) cols * sizeof(float);
    const size_t bytes_dst = (size_t) rows * sizeof(float);
    ggml_cuda8_backend_buffer * b_src = NULL;
    ggml_cuda8_backend_buffer * b_dst = NULL;
    if (ggml_cuda8_context_alloc_buffer(ctx, bytes_src, &b_src) != 0) return -1;
    if (ggml_cuda8_context_alloc_buffer(ctx, bytes_dst, &b_dst) != 0) { ggml_cuda8_backend_buffer_free(b_src); return -1; }
    int rc = 0;
    if (ggml_cuda8_backend_buffer_upload(b_src, 0, src0->data, bytes_src) != 0) rc = -1;
    if (rc == 0 && ggml_cuda8_backend_buffer_clear(b_dst, 0) != 0) rc = -1;
    if (rc == 0) {
        const float * d_src = (const float *) ggml_cuda8_backend_buffer_get_base_const(b_src);
        float * d_dst = (float *) ggml_cuda8_backend_buffer_get_base(b_dst);
        rc = is_max ? ggml_cuda8_reduce_max_rows_f32_launch(d_src, d_dst, rows, cols)
                    : ggml_cuda8_reduce_sum_rows_f32_launch(d_src, d_dst, rows, cols);
    }
    if (rc == 0 && ggml_cuda8_backend_buffer_download(b_dst, 0, dst->data, bytes_dst) != 0) rc = -1;
    ggml_cuda8_backend_buffer_free(b_src);
    ggml_cuda8_backend_buffer_free(b_dst);
    return rc;
}

static int exec_common(ggml_cuda8_context * ctx, const ggml_tensor * src0, ggml_tensor * dst, int is_max) {
    if (!ggml_cuda8_supported_reduce_rows_f32(ctx, src0, dst)) { std::fprintf(stderr, "ggml-cuda8/reduce: unsupported layout\n"); return -1; }
    const int cols = (int) src0->ne[0];
    const int rows = (int) src0->ne[1];
    const size_t bytes_src = (size_t) rows * (size_t) cols * sizeof(float);
    const size_t bytes_dst = (size_t) rows * sizeof(float);
    const int src_dev = tensor_is_cuda8_resident(src0, bytes_src);
    const int dst_dev = tensor_is_cuda8_resident(dst, bytes_dst);
    if (src_dev && dst_dev) return exec_device(src0, dst, is_max, rows, cols);
    if (src_dev || dst_dev) { std::fprintf(stderr, "ggml-cuda8/reduce: mixed host/device src0/dst unsupported in G11D\n"); return -1; }
    return exec_host(ctx, src0, dst, is_max, rows, cols);
}

int ggml_cuda8_exec_reduce_sum_rows_f32(ggml_cuda8_context * ctx, const ggml_tensor * src0, ggml_tensor * dst) { return exec_common(ctx, src0, dst, 0); }
int ggml_cuda8_exec_reduce_max_rows_f32(ggml_cuda8_context * ctx, const ggml_tensor * src0, ggml_tensor * dst) { return exec_common(ctx, src0, dst, 1); }
