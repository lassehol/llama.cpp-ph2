// ggml/src/ggml-cuda8/mmv.cuh
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

int ggml_cuda8_mmv_f32(
    const float * d_A,
    const float * d_x,
    float * d_y,
    int rows,
    int cols
);

int ggml_cuda8_mmv_f32_roundtrip_test(void);

#ifdef __cplusplus
}
#endif
