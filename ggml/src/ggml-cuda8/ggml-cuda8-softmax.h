// ggml/src/ggml-cuda8/ggml-cuda8-softmax.h
//
// G9B-5B: F32 row-wise softmax dispatcher helper.

#ifndef GGML_CUDA8_SOFTMAX_H
#define GGML_CUDA8_SOFTMAX_H

#include "ggml-cuda8-context.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

int ggml_cuda8_supported_softmax_rows_f32(
    const struct ggml_cuda8_context * ctx,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * dst
);

int ggml_cuda8_exec_softmax_rows_f32(
    struct ggml_cuda8_context * ctx,
    const struct ggml_tensor * src0,
    struct ggml_tensor * dst
);

#ifdef __cplusplus
}
#endif

#endif // GGML_CUDA8_SOFTMAX_H
