// ggml/src/ggml-cuda8/ggml-cuda8-cpy.h
//
// G9B-1: F32 -> F32 CPY dispatcher helper

#ifndef GGML_CUDA8_CPY_H
#define GGML_CUDA8_CPY_H

#include "ggml-cuda8-context.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

int ggml_cuda8_supported_cpy_f32(
    const struct ggml_cuda8_context * ctx,
    const struct ggml_tensor * src,
    const struct ggml_tensor * dst
);

int ggml_cuda8_exec_cpy_f32(
    struct ggml_cuda8_context * ctx,
    const struct ggml_tensor * src,
    struct ggml_tensor * dst
);

#ifdef __cplusplus
}
#endif

#endif // GGML_CUDA8_CPY_H
