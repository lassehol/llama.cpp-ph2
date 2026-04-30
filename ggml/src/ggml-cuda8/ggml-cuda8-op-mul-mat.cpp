// ggml/src/ggml-cuda8/ggml-cuda8-op-mul-mat.cpp
//
// G4 minimal GGML-tensor-facing MUL_MAT adapter.
//
// Supported only:
//   src0: Q8_0 matrix [cols, rows]
//   src1: F32 vector [cols]
//   dst:  F32 vector [rows]
//
// The function is synchronous and copies host tensor data to/from device.
// This is intentionally conservative for the first GGML-facing op step.

#include "ggml-cuda8-op-mul-mat.h"

#include "buffer.cuh"
#include "ggml-cuda8-mulmat.cuh"
#include "q8_0-mmv.cuh"

#include <cstdio>
#include <stdint.h>

static int ggml_cuda8_op_check_tensor_ptrs(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst
) {
    if (src0 == NULL || src1 == NULL || dst == NULL) {
        std::fprintf(stderr, "ggml-cuda8/op-mul-mat: null tensor pointer\n");
        return -1;
    }

    if (src0->data == NULL || src1->data == NULL || dst->data == NULL) {
        std::fprintf(stderr, "ggml-cuda8/op-mul-mat: null tensor data pointer\n");
        return -1;
    }

    return 0;
}

extern "C" int ggml_cuda8_op_mul_mat_q8_0_f32_supported(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst
) {
    if (ggml_cuda8_op_check_tensor_ptrs(src0, src1, dst) != 0) {
        return 0;
    }

    if (src0->type != GGML_TYPE_Q8_0) {
        return 0;
    }

    if (src1->type != GGML_TYPE_F32) {
        return 0;
    }

    if (dst->type != GGML_TYPE_F32) {
        return 0;
    }

    const int64_t cols = src0->ne[0];
    const int64_t rows = src0->ne[1];

    if (cols <= 0 || rows <= 0) {
        return 0;
    }

    // G4 keeps this strict. Real GGML Q8_0 rows should be quantized in QK=32 blocks.
    if ((cols % GGML_CUDA8_QK8_0) != 0) {
        return 0;
    }

    if (src1->ne[0] != cols) {
        return 0;
    }

    if (dst->ne[0] != rows) {
        return 0;
    }

    // This G4 path is vector-only. Avoid silently accepting batched cases.
    if (src1->ne[1] != 1 || dst->ne[1] != 1) {
        return 0;
    }

    return 1;
}

extern "C" int ggml_cuda8_op_mul_mat_q8_0_f32(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct ggml_tensor * dst
) {
    if (!ggml_cuda8_op_mul_mat_q8_0_f32_supported(src0, src1, dst)) {
        std::fprintf(stderr, "ggml-cuda8/op-mul-mat: unsupported tensor configuration\n");
        return -1;
    }

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

    ggml_cuda8_q8_0_block * d_src0 = NULL;
    float * d_src1 = NULL;
    float * d_dst  = NULL;

    if (ggml_cuda8_buffer_malloc((void **) &d_src0, bytes_src0) != 0) {
        return -1;
    }

    if (ggml_cuda8_buffer_malloc((void **) &d_src1, bytes_src1) != 0) {
        ggml_cuda8_buffer_free(d_src0);
        return -1;
    }

    if (ggml_cuda8_buffer_malloc((void **) &d_dst, bytes_dst) != 0) {
        ggml_cuda8_buffer_free(d_src0);
        ggml_cuda8_buffer_free(d_src1);
        return -1;
    }

    int rc = 0;

    if (ggml_cuda8_buffer_upload(d_src0, src0->data, bytes_src0) != 0) {
        rc = -1;
    }

    if (rc == 0 && ggml_cuda8_buffer_upload(d_src1, src1->data, bytes_src1) != 0) {
        rc = -1;
    }

    if (rc == 0 && ggml_cuda8_buffer_memset(d_dst, 0, bytes_dst) != 0) {
        rc = -1;
    }

    if (rc == 0) {
        rc = ggml_cuda8_mul_mat_q8_0_f32(
            d_src0,
            d_src1,
            d_dst,
            rows,
            cols
        );
    }

    if (rc == 0 && ggml_cuda8_buffer_download(dst->data, d_dst, bytes_dst) != 0) {
        rc = -1;
    }

    ggml_cuda8_buffer_free(d_src0);
    ggml_cuda8_buffer_free(d_src1);
    ggml_cuda8_buffer_free(d_dst);

    return rc;
}
