// ggml/src/ggml-cuda8/ggml-cuda8-softmax.cpp
//
// G9B-5B: F32 row-wise softmax dispatcher helper.
//
// G11A-4E:
//   Adds device-resident SOFTMAX_ROWS_F32 path.
//   If src0 and dst tensor->data are inside registered CUDA8
//   ggml_backend_buffer_t wrappers, launch softmax directly on device pointers.
//   Host-backed path remains supported.
//   Mixed host/device src0/dst is rejected.

#include "ggml-cuda8-softmax.h"
#include "ggml-cuda8-backend-buffer.h"
#include "ggml-cuda8-ggml-buffer.h"

#include <cstdio>

extern "C" int ggml_cuda8_softmax_rows_f32_launch(
    const float * src,
    float * dst,
    int rows,
    int cols
);

static int tensor_is_contiguous_f32_matrix_2d(const ggml_tensor * t) {
    if (t == NULL) return 0;
    if (t->type != GGML_TYPE_F32) return 0;
    if (t->data == NULL) return 0;

    if (t->ne[0] <= 0) return 0;
    if (t->ne[1] <= 0) return 0;
    if (t->ne[2] != 1) return 0;
    if (t->ne[3] != 1) return 0;

    if (t->nb[0] != sizeof(float)) return 0;
    if (t->nb[1] != (size_t) t->ne[0] * sizeof(float)) return 0;

    return 1;
}

int ggml_cuda8_supported_softmax_rows_f32(
    const ggml_cuda8_context * ctx,
    const ggml_tensor * src0,
    const ggml_tensor * dst
) {
    (void) ctx;

    if (!tensor_is_contiguous_f32_matrix_2d(src0)) return 0;
    if (!tensor_is_contiguous_f32_matrix_2d(dst)) return 0;

    if (src0->ne[0] != dst->ne[0]) return 0;
    if (src0->ne[1] != dst->ne[1]) return 0;
    if (src0->ne[2] != dst->ne[2]) return 0;
    if (src0->ne[3] != dst->ne[3]) return 0;

    return 1;
}

static int tensor_is_cuda8_resident(
    const ggml_tensor * t,
    size_t bytes
) {
    ggml_backend_buffer_t owner = NULL;
    size_t offset = 0;

    return ggml_cuda8_ggml_tensor_is_device_resident(
        t,
        bytes,
        &owner,
        &offset
    );
}

static int exec_softmax_rows_f32_device_resident(
    const ggml_tensor * src0,
    ggml_tensor * dst,
    int rows,
    int cols
) {
    return ggml_cuda8_softmax_rows_f32_launch(
        (const float *) src0->data,
        (float *) dst->data,
        rows,
        cols
    );
}

static int exec_softmax_rows_f32_host_staging(
    ggml_cuda8_context * ctx,
    const ggml_tensor * src0,
    ggml_tensor * dst,
    int rows,
    int cols
) {
    const size_t bytes =
        (size_t) rows * (size_t) cols * sizeof(float);

    ggml_cuda8_backend_buffer * b_src = NULL;
    ggml_cuda8_backend_buffer * b_dst = NULL;

    if (ggml_cuda8_context_alloc_buffer(ctx, bytes, &b_src) != 0) {
        return -1;
    }

    if (ggml_cuda8_context_alloc_buffer(ctx, bytes, &b_dst) != 0) {
        ggml_cuda8_backend_buffer_free(b_src);
        return -1;
    }

    int rc = 0;

    if (ggml_cuda8_backend_buffer_upload(b_src, 0, src0->data, bytes) != 0) {
        rc = -1;
    }

    if (rc == 0 && ggml_cuda8_backend_buffer_clear(b_dst, 0) != 0) {
        rc = -1;
    }

    if (rc == 0) {
        const float * d_src =
            (const float *) ggml_cuda8_backend_buffer_get_base_const(b_src);

        float * d_dst =
            (float *) ggml_cuda8_backend_buffer_get_base(b_dst);

        rc = ggml_cuda8_softmax_rows_f32_launch(d_src, d_dst, rows, cols);
    }

    if (rc == 0) {
        if (ggml_cuda8_backend_buffer_download(b_dst, 0, dst->data, bytes) != 0) {
            rc = -1;
        }
    }

    ggml_cuda8_backend_buffer_free(b_src);
    ggml_cuda8_backend_buffer_free(b_dst);

    return rc;
}

int ggml_cuda8_exec_softmax_rows_f32(
    ggml_cuda8_context * ctx,
    const ggml_tensor * src0,
    ggml_tensor * dst
) {
    if (!ggml_cuda8_supported_softmax_rows_f32(ctx, src0, dst)) {
        std::fprintf(stderr, "ggml-cuda8/softmax: unsupported layout\n");
        return -1;
    }

    const int cols = (int) src0->ne[0];
    const int rows = (int) src0->ne[1];

    const size_t bytes =
        (size_t) rows * (size_t) cols * sizeof(float);

    const int src_dev = tensor_is_cuda8_resident(src0, bytes);
    const int dst_dev = tensor_is_cuda8_resident(dst,  bytes);

    if (src_dev && dst_dev) {
        return exec_softmax_rows_f32_device_resident(src0, dst, rows, cols);
    }

    if (src_dev || dst_dev) {
        std::fprintf(stderr,
            "ggml-cuda8/softmax: mixed host/device src0/dst unsupported in G11A-4E\n");
        return -1;
    }

    return exec_softmax_rows_f32_host_staging(ctx, src0, dst, rows, cols);
}
