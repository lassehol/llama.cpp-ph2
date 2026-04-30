// ggml/src/ggml-cuda8/ggml-cuda8-scalar.h
//
// G9B-3: F32 scalar ADD/MUL dispatcher helpers.

#ifndef GGML_CUDA8_SCALAR_H
#define GGML_CUDA8_SCALAR_H

#include "ggml-cuda8-context.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

int ggml_cuda8_supported_scalar_f32(
    const struct ggml_cuda8_context * ctx,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst
);

int ggml_cuda8_exec_add_scalar_f32(
    struct ggml_cuda8_context * ctx,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct ggml_tensor * dst
);

int ggml_cuda8_exec_mul_scalar_f32(
    struct ggml_cuda8_context * ctx,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct ggml_tensor * dst
);

#ifdef __cplusplus
}
#endif

#endif // GGML_CUDA8_SCALAR_H
