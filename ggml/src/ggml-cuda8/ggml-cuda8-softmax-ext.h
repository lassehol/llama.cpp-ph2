// ggml/src/ggml-cuda8/ggml-cuda8-softmax-ext.h
//
// G41: masked / scaled / ALiBi-biased row-wise softmax dispatcher helper.
//
// Complements ggml-cuda8-softmax.h's plain SOFTMAX_ROWS_F32 path. That path
// (unchanged by G41) requires ggml_cuda8_soft_max_is_plain() - no mask, no
// sinks, scale==1, max_bias==0 - and every existing plain-softmax pipeline
// (G16C, G19A, G32A, ...) still takes it. This header covers the ggml
// ggml_soft_max_ext() forms that is_plain() rejects, EXCEPT attention sinks
// and F16 masks, which remain explicitly refused (see
// ggml_cuda8_soft_max_is_supported_ext() in ggml-cuda8-ggml-backend.h) rather
// than silently mishandled.
#ifndef GGML_CUDA8_SOFTMAX_EXT_H
#define GGML_CUDA8_SOFTMAX_EXT_H

#include "ggml-cuda8-context.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// src1 (mask) may be NULL - a scaled/ALiBi-biased softmax with no mask at all
// is valid input to this path too (is_plain() would already have accepted it
// if scale==1 and max_bias==0, so reaching here with src1==NULL implies
// scale!=1 or max_bias!=0).
int ggml_cuda8_supported_softmax_ext_f32(
    const struct ggml_cuda8_context * ctx,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,   // mask, may be NULL
    const struct ggml_tensor * dst
);

int ggml_cuda8_exec_softmax_ext_f32(
    struct ggml_cuda8_context * ctx,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,   // mask, may be NULL
    struct ggml_tensor * dst
);

#ifdef __cplusplus
}
#endif

#endif // GGML_CUDA8_SOFTMAX_EXT_H
