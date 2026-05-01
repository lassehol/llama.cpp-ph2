// ggml-cuda8-cpy.cu  -  F32 device-to-device copy kernel
// Used by CPY_F32 and CONT_F32 dispatch paths.
#include <cuda_runtime.h>
#include <cstdio>

extern "C" int ggml_cuda8_cpy_f32_d2d(
        const float * src, float * dst, size_t n_bytes) {
    if (src == NULL || dst == NULL || n_bytes == 0) {
        std::fprintf(stderr, "ggml-cuda8/cpy-d2d: invalid args\n");
        return -1;
    }
    cudaError_t err = cudaMemcpy(dst, src, n_bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "ggml-cuda8/cpy-d2d: cudaMemcpy failed: %s\n",
            cudaGetErrorString(err));
        return -1;
    }
    return 0;
}
