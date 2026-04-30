// ggml/src/ggml-cuda8/q8_0-mmv.cuh
//
// CUDA8/Fermi-safe Q8_0 MMV declarations.
//
// Q8_0 layout used here:
//   block size: 32
//   d:          IEEE-754 fp16 bits stored as uint16_t
//   qs:         32 signed int8 values

#ifndef GGML_CUDA8_Q8_0_MMV_CUH
#define GGML_CUDA8_Q8_0_MMV_CUH

#include <stdint.h>

#define GGML_CUDA8_QK8_0 32

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_cuda8_q8_0_block {
    uint16_t d;
    int8_t   qs[GGML_CUDA8_QK8_0];
};

int ggml_cuda8_q8_0_mmv_f32(
    const struct ggml_cuda8_q8_0_block * d_Aq,
    const float * d_x,
    float * d_y,
    int rows,
    int cols
);

int ggml_cuda8_q8_0_mmv_f32_block(
    const struct ggml_cuda8_q8_0_block * d_Aq,
    const float * d_x,
    float * d_y,
    int rows,
    int cols
);

int ggml_cuda8_q8_0_mmv_f32_qblock(
    const struct ggml_cuda8_q8_0_block * d_Aq,
    const float * d_x,
    float * d_y,
    int rows,
    int cols
);

int ggml_cuda8_q8_0_mmv_f32_col_parallel(
    const struct ggml_cuda8_q8_0_block * d_Aq,
    const float * d_x,
    float * d_y,
    int rows,
    int cols
);

int ggml_cuda8_q8_0_mmv_roundtrip_test(void);

#ifdef __cplusplus
}
#endif

#endif // GGML_CUDA8_Q8_0_MMV_CUH
