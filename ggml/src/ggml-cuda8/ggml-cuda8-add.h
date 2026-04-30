// ggml/src/ggml-cuda8/ggml-cuda8-add.h
//
// G9B-2: F32 vector ADD dispatcher helper

#ifndef GGML_CUDA8_ADD_H
#define GGML_CUDA8_ADD_H

#include "ggml-cuda8-context.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

int ggml_cuda8_supported_add_f32(
    const struct ggml_cuda8_context * ctx,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst
);

int ggml_cuda8_exec_add_f32(
    struct ggml_cuda8_context * ctx,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct ggml_tensor * dst
);

#ifdef __cplusplus
}
#endif



#endif // GGML_CUDA8_ADD_H
