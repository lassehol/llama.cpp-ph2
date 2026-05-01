// ggml-cuda8-diagmask.cu  -  G31A: causal diagonal mask kernel
// Fermi-safe. Matches upstream ggml-cuda/diagmask.cu logic.
#include <cuda_runtime.h>
#include <cstdio>
#include <float.h>

static __global__ void kernel_diag_mask_inf_f32(
        const float * __restrict__ x,
        float       * __restrict__ dst,
        const int ncols,
        const int rows_per_channel,
        const int n_past) {

    const int col = blockDim.y * blockIdx.y + threadIdx.y;
    const int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (col >= ncols) return;

    const int i = row * ncols + col;
    dst[i] = x[i] - (col > n_past + row % rows_per_channel) * FLT_MAX;
}

extern "C" int ggml_cuda8_op_diag_mask_inf_f32(
        const float * x, float * dst,
        int ncols, int nrows, int rows_per_channel, int n_past) {

    if (x == NULL || dst == NULL || ncols <= 0 || nrows <= 0) {
        std::fprintf(stderr, "ggml-cuda8/diagmask: invalid args\n");
        return -1;
    }

    const int block_size = 256;
    const dim3 block_dims(1, block_size, 1);
    const int block_num_x = (ncols + block_size - 1) / block_size;
    const dim3 grid_dims(nrows, block_num_x, 1);

    kernel_diag_mask_inf_f32<<<grid_dims, block_dims>>>(
        x, dst, ncols, rows_per_channel, n_past);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "ggml-cuda8/diagmask: launch failed: %s\n",
            cudaGetErrorString(err));
        return -1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "ggml-cuda8/diagmask: sync failed: %s\n",
            cudaGetErrorString(err));
        return -1;
    }
    return 0;
}
