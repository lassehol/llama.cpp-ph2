// ggml/src/ggml-cuda8/ggml-cuda8-mulmat.cuh
//
// Minimal GGML-shaped CUDA8 mul_mat shim.
//
// This is not full GGML backend registration yet.
// It provides a narrow, explicit API:
//
//   Q8_0 matrix [rows, cols] x F32 vector [cols] -> F32 vector [rows]
//
// Layout:
//   src0_q8_0: row-major Q8_0 blocks
//   src1_f32:  contiguous F32 vector
//   dst_f32:   contiguous F32 output vector

#ifndef GGML_CUDA8_MULMAT_CUH
#define GGML_CUDA8_MULMAT_CUH

#include "q8_0-mmv.cuh"

#ifdef __cplusplus
extern "C" {
#endif

int ggml_cuda8_mul_mat_q8_0_f32(
    const struct ggml_cuda8_q8_0_block * d_src0_q8_0,
    const float * d_src1_f32,
    float * d_dst_f32,
    int rows,
    int cols
);

#ifdef __cplusplus
}
#endif

#endif // GGML_CUDA8_MULMAT_CUH
