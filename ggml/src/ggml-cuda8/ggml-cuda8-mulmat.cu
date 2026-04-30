// ggml/src/ggml-cuda8/ggml-cuda8-mulmat.cu
//
// Minimal GGML-shaped CUDA8 mul_mat shim.
//
// Current supported case:
//   src0: Q8_0 matrix, row-major, shape [rows, cols]
//   src1: F32 vector, shape [cols]
//   dst:  F32 vector, shape [rows]
//
// This wrapper deliberately does not know about ggml_tensor yet.
// It is the bridge between the validated CUDA8 kernels and a future
// GGML_OP_MUL_MAT dispatch path.

#include "ggml-cuda8-mulmat.cuh"

#include <cstdio>

extern "C" int ggml_cuda8_mul_mat_q8_0_f32(
    const ggml_cuda8_q8_0_block * d_src0_q8_0,
    const float * d_src1_f32,
    float * d_dst_f32,
    int rows,
    int cols
) {
    if (d_src0_q8_0 == NULL || d_src1_f32 == NULL || d_dst_f32 == NULL) {
        std::fprintf(stderr, "ggml-cuda8/mulmat: null device pointer\n");
        return -1;
    }

    if (rows <= 0 || cols <= 0) {
        std::fprintf(stderr,
            "ggml-cuda8/mulmat: invalid shape rows=%d cols=%d\n",
            rows, cols);
        return -1;
    }

    // For G0, dispatch directly to the selected Q8_0 MMV implementation.
    return ggml_cuda8_q8_0_mmv_f32(
        d_src0_q8_0,
        d_src1_f32,
        d_dst_f32,
        rows,
        cols
    );
}
