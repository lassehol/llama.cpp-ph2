// ggml/src/ggml-cuda8/ggml-cuda8-reduce.h
//
// G9B-4/G9B-5A: F32 row-wise reduction dispatcher helpers.
//
// Supported:
//   src0: GGML_TYPE_F32 matrix [cols, rows], contiguous row-major
//   dst : GGML_TYPE_F32 vector [rows]
//
// Computes:
//   REDUCE_SUM_ROWS_F32:
//      dst[row] = sum(src0[row, 0:cols])
//
//   REDUCE_MAX_ROWS_F32:
//      dst[row] = max(src0[row, 0:cols])

#ifndef GGML_CUDA8_REDUCE_H
#define GGML_CUDA8_REDUCE_H

#include "ggml-cuda8-context.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

int ggml_cuda8_supported_reduce_rows_f32(
    const struct ggml_cuda8_context * ctx,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * dst
);

int ggml_cuda8_supported_reduce_sum_rows_f32(
    const struct ggml_cuda8_context * ctx,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * dst
);

int ggml_cuda8_supported_reduce_max_rows_f32(
    const struct ggml_cuda8_context * ctx,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * dst
);

int ggml_cuda8_exec_reduce_sum_rows_f32(
    struct ggml_cuda8_context * ctx,
    const struct ggml_tensor * src0,
    struct ggml_tensor * dst
);

int ggml_cuda8_exec_reduce_max_rows_f32(
    struct ggml_cuda8_context * ctx,
    const struct ggml_tensor * src0,
    struct ggml_tensor * dst
);

#ifdef __cplusplus
}
#endif

#endif // GGML_CUDA8_REDUCE_H
