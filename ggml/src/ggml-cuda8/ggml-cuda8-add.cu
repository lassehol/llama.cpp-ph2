// ggml/src/ggml-cuda8/ggml-cuda8-add.cu
//
// G9B-2: F32 vector ADD kernel (Fermi-safe)

#include <cuda_runtime.h>

#include "ggml-cuda8-grid.cuh"

static __global__ void kernel_add_f32(
        const float * __restrict__ a,
        const float * __restrict__ b,
        float * __restrict__ c,
        const int n) {
    // G38: grid-stride - the grid is clamped to 65535 blocks on Fermi.
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

extern "C" int ggml_cuda8_add_f32_launch(
        const float * a,
        const float * b,
        float * c,
        int n) {
    const int block_size = 256;
    const int grid_size  = ggml_cuda8_grid_1d(n, block_size);
    kernel_add_f32<<<grid_size, block_size>>>(a, b, c, n);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : -1;
}
