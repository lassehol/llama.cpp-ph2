// ggml-cuda8-rope.cu  -  G28A: Rotary Positional Embedding kernel
// Basic ROPE (mode=0, no YaRN, no mrope). Fermi-safe.
#include <cuda_runtime.h>
#include <cstdio>
#include <math.h>

#include "ggml-cuda8-grid.cuh"

// One thread per pair of elements across all heads/positions.
// Layout: x[i0 + i1*ne0 + i2*ne0*ne1 + i3*ne0*ne1*ne2]
//   ne0 = head_dim, ne1 = n_heads, ne2 = seq_len, ne3 = batch
//   pos[i2] = position id for sequence index i2
//   Pairs at i0 < n_dims are rotated; i0 >= n_dims pass through.
static __global__ void kernel_rope_f32(
        const float * __restrict__ x,
        float       * __restrict__ dst,
        const int   * __restrict__ pos,
        const int ne0,
        const int ne1,
        const int ne2,
        const int ne3,
        const int n_dims,
        const float theta_scale,
        const float freq_scale) {

    const int pairs_per_row = ne0 / 2;
    const int rows = ne1 * ne2 * ne3;
    const int total_pairs = pairs_per_row * rows;

    // G38: grid-stride - the grid is clamped to 65535 blocks on Fermi.
    const int stride = blockDim.x * gridDim.x;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_pairs;
         idx += stride) {

        const int pair = idx % pairs_per_row;
        const int row  = idx / pairs_per_row;

        const int i1 = row % ne1;
        const int i2 = (row / ne1) % ne2;
        const int i3 = row / (ne1 * ne2);

        const int i0 = pair * 2;
        const int offset = i0 + i1 * ne0
                              + i2 * ne0 * ne1
                              + i3 * ne0 * ne1 * ne2;

        if (i0 >= n_dims) {
            // Beyond rotary dims - pass through unchanged
            dst[offset]     = x[offset];
            dst[offset + 1] = x[offset + 1];
            continue;
        }

        const int p = pos[i2];
        float theta = (float)p * powf(theta_scale, (float)pair) * freq_scale;
        float cos_t = cosf(theta);
        float sin_t = sinf(theta);

        float x0 = x[offset];
        float x1 = x[offset + 1];

        dst[offset]     = x0 * cos_t - x1 * sin_t;
        dst[offset + 1] = x0 * sin_t + x1 * cos_t;
    }
}

extern "C" int ggml_cuda8_op_rope_f32(
        const float * x,
        float * dst,
        const int * pos,
        int ne0, int ne1, int ne2, int ne3,
        int n_dims,
        float freq_base,
        float freq_scale) {

    if (x == NULL || dst == NULL || pos == NULL || ne0 < 2) {
        std::fprintf(stderr, "ggml-cuda8/rope: invalid args\n");
        return -1;
    }

    const float theta_scale = powf(freq_base, -2.0f / (float)n_dims);
    const int total_pairs = (ne0 / 2) * ne1 * ne2 * ne3;
    const int block = 256;
    const int grid  = ggml_cuda8_grid_1d(total_pairs, block);

    kernel_rope_f32<<<grid, block>>>(
        x, dst, pos, ne0, ne1, ne2, ne3,
        n_dims, theta_scale, freq_scale);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "ggml-cuda8/rope: launch failed: %s\n",
            cudaGetErrorString(err));
        return -1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "ggml-cuda8/rope: sync failed: %s\n",
            cudaGetErrorString(err));
        return -1;
    }
    return 0;
}
