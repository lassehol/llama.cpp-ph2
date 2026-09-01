// ggml-cuda8-getrows.cu  -  G33A: embedding lookup / GET_ROWS kernel
// Fermi-safe. Simple 2D case: dst[i,:] = src0[src1[i],:]
// src0: [ne00, ne01] F32 embedding table (ne00=embd, ne01=vocab)
// src1: [n_tokens]   I32 token indices
// dst:  [ne00, n_tokens] F32 output
#include <cuda_runtime.h>
#include <cstdio>

#include "ggml-cuda8-grid.cuh"

static __global__ void kernel_get_rows_f32(
        const float * __restrict__ src0,
        const int   * __restrict__ src1,
        float       * __restrict__ dst,
        const int ne00,
        const int n_tokens) {

    // G38: row-stride - the grid is clamped to 65535 blocks on Fermi.
    // No shared memory here, so no __syncthreads() to keep collective.
    for (int token = blockIdx.x; token < n_tokens; token += gridDim.x) {
        const int row_idx = src1[token];
        const float * src_row = src0 + (size_t)row_idx * ne00;
        float       * dst_row = dst  + (size_t)token   * ne00;

        for (int col = threadIdx.x; col < ne00; col += blockDim.x) {
            dst_row[col] = src_row[col];
        }
    }
}

extern "C" int ggml_cuda8_op_get_rows_f32(
        const float * src0,
        const int   * src1,
        float       * dst,
        int ne00,
        int n_tokens) {

    if (src0 == NULL || src1 == NULL || dst == NULL || ne00 <= 0 || n_tokens <= 0) {
        std::fprintf(stderr, "ggml-cuda8/getrows: invalid args\n");
        return -1;
    }

    const int block_size = 256;
    kernel_get_rows_f32<<<ggml_cuda8_grid_rows(n_tokens), block_size>>>(
        src0, src1, dst, ne00, n_tokens);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "ggml-cuda8/getrows: launch failed: %s\n",
            cudaGetErrorString(err));
        return -1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "ggml-cuda8/getrows: sync failed: %s\n",
            cudaGetErrorString(err));
        return -1;
    }
    return 0;
}
