// ggml/src/ggml-cuda8/ggml-cuda8-mulmat-f32.h
//
// G42: batched F32xF32 MUL_MAT dispatcher helper - the attention matmuls
// (K.Q, probs.V), which operate on activations rather than weights and so
// were not covered by the existing Q8_0/Q4_K/Q6_K quantized-weight MUL_MAT
// paths (ggml-cuda8-dispatch.cpp) or the plain vector kernels in mmv.cu.
//
// NOT the same file as ggml-cuda8-mulmat.cu/.cuh, which implements the F32
// vector matvec path used by ggml-cuda8-context.cpp's Q8_0 helper - this is
// a separate, batched/broadcast-aware kernel for real attention shapes.
//
// Scope: batched over ne02/ne03 with GQA-style broadcast (src1's ne02/ne03
// must be integer multiples of src0's), general strides on dims 1-3 (so
// permuted views - the common case for attention K/Q/V after
// reshape+permute - are handled directly), but dim 0 (the reduction
// dimension) must be contiguous on both src0 and src1: nb[0]==sizeof(float).
// Fully arbitrary dim-0 strides are out of scope here.
#ifndef GGML_CUDA8_MULMAT_F32_H
#define GGML_CUDA8_MULMAT_F32_H

#include "ggml-cuda8-context.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

int ggml_cuda8_supported_mul_mat_f32_f32(
    const struct ggml_cuda8_context * ctx,
    const struct ggml_tensor * src0,   // [ne00, ne01, ne02, ne03] - e.g. K or V
    const struct ggml_tensor * src1,   // [ne00, ne11, ne12, ne13] - e.g. Q or probs
    const struct ggml_tensor * dst     // [ne01, ne11, ne12, ne13]
);

int ggml_cuda8_exec_mul_mat_f32_f32(
    struct ggml_cuda8_context * ctx,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct ggml_tensor * dst
);

#ifdef __cplusplus
}
#endif

#endif // GGML_CUDA8_MULMAT_F32_H
