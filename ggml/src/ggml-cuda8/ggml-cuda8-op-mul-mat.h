// ggml/src/ggml-cuda8/ggml-cuda8-op-mul-mat.h
//
// G4 minimal GGML-tensor-facing MUL_MAT adapter.
//
// This is still not full ggml_backend_t registration.
// It validates real ggml_tensor metadata for one narrow supported case:
//
//   src0: GGML_TYPE_Q8_0 matrix, shape [cols, rows]
//   src1: GGML_TYPE_F32 vector, shape [cols]
//   dst:  GGML_TYPE_F32 vector, shape [rows]
//
// Then it calls the existing CUDA8 Q8_0 mul_mat shim.

#ifndef GGML_CUDA8_OP_MUL_MAT_H
#define GGML_CUDA8_OP_MUL_MAT_H

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

int ggml_cuda8_op_mul_mat_q8_0_f32_supported(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst
);

int ggml_cuda8_op_mul_mat_q8_0_f32(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct ggml_tensor * dst
);

#ifdef __cplusplus
}
#endif

#endif // GGML_CUDA8_OP_MUL_MAT_H
