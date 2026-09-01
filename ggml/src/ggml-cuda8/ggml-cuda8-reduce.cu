// ggml/src/ggml-cuda8/ggml-cuda8-reduce.cu
//
// G9B-4/G9B-5A: F32 row-wise reduce sum/max kernels.
// CUDA8/Fermi-safe:
//   - no warp shuffle
//   - no cooperative groups
//   - shared-memory tree reduction

#include <cuda_runtime.h>
#include <cstdio>
#include <float.h>

#include "ggml-cuda8-grid.cuh"

static const int GGML_CUDA8_REDUCE_BLOCK_SIZE = 128;

static __device__ void ggml_cuda8_reduce_sum_128(float * partial, int tid) {
    if (tid < 64) { partial[tid] += partial[tid + 64]; }
    __syncthreads();

    if (tid < 32) { partial[tid] += partial[tid + 32]; }
    __syncthreads();

    if (tid < 16) { partial[tid] += partial[tid + 16]; }
    __syncthreads();

    if (tid < 8) { partial[tid] += partial[tid + 8]; }
    __syncthreads();

    if (tid < 4) { partial[tid] += partial[tid + 4]; }
    __syncthreads();

    if (tid < 2) { partial[tid] += partial[tid + 2]; }
    __syncthreads();

    if (tid < 1) { partial[tid] += partial[tid + 1]; }
    __syncthreads();
}

static __device__ void ggml_cuda8_reduce_max_128(float * partial, int tid) {
    if (tid < 64) { partial[tid] = partial[tid] > partial[tid + 64] ? partial[tid] : partial[tid + 64]; }
    __syncthreads();

    if (tid < 32) { partial[tid] = partial[tid] > partial[tid + 32] ? partial[tid] : partial[tid + 32]; }
    __syncthreads();

    if (tid < 16) { partial[tid] = partial[tid] > partial[tid + 16] ? partial[tid] : partial[tid + 16]; }
    __syncthreads();

    if (tid < 8) { partial[tid] = partial[tid] > partial[tid + 8] ? partial[tid] : partial[tid + 8]; }
    __syncthreads();

    if (tid < 4) { partial[tid] = partial[tid] > partial[tid + 4] ? partial[tid] : partial[tid + 4]; }
    __syncthreads();

    if (tid < 2) { partial[tid] = partial[tid] > partial[tid + 2] ? partial[tid] : partial[tid + 2]; }
    __syncthreads();

    if (tid < 1) { partial[tid] = partial[tid] > partial[tid + 1] ? partial[tid] : partial[tid + 1]; }
    __syncthreads();
}

static __global__ void ggml_cuda8_reduce_sum_rows_f32_kernel(
    const float * src,
    float * dst,
    int rows,
    int cols
) {
    const int tid = threadIdx.x;

    __shared__ float partial[GGML_CUDA8_REDUCE_BLOCK_SIZE];

    // G38: row-stride - the grid is clamped to 65535 blocks on Fermi.
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const float * row_ptr = src + (size_t) row * cols;

        float sum = 0.0f;

        for (int c = tid; c < cols; c += GGML_CUDA8_REDUCE_BLOCK_SIZE) {
            sum += row_ptr[c];
        }

        partial[tid] = sum;
        __syncthreads();

        ggml_cuda8_reduce_sum_128(partial, tid);

        if (tid == 0) {
            dst[row] = partial[0];
        }

        // Guard the next iteration's partial[tid] write.
        __syncthreads();
    }
}

static __global__ void ggml_cuda8_reduce_max_rows_f32_kernel(
    const float * src,
    float * dst,
    int rows,
    int cols
) {
    const int tid = threadIdx.x;

    __shared__ float partial[GGML_CUDA8_REDUCE_BLOCK_SIZE];

    // G38: row-stride - the grid is clamped to 65535 blocks on Fermi.
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const float * row_ptr = src + (size_t) row * cols;

        float vmax = -FLT_MAX;

        for (int c = tid; c < cols; c += GGML_CUDA8_REDUCE_BLOCK_SIZE) {
            const float x = row_ptr[c];
            vmax = vmax > x ? vmax : x;
        }

        partial[tid] = vmax;
        __syncthreads();

        ggml_cuda8_reduce_max_128(partial, tid);

        if (tid == 0) {
            dst[row] = partial[0];
        }

        // Guard the next iteration's partial[tid] write.
        __syncthreads();
    }
}

extern "C" int ggml_cuda8_reduce_sum_rows_f32_launch(
    const float * src,
    float * dst,
    int rows,
    int cols
) {
    if (src == NULL || dst == NULL || rows <= 0 || cols <= 0) {
        std::fprintf(stderr,
            "ggml-cuda8/reduce-sum-rows: invalid args rows=%d cols=%d\n",
            rows, cols);
        return -1;
    }

    ggml_cuda8_reduce_sum_rows_f32_kernel<<<ggml_cuda8_grid_rows(rows), GGML_CUDA8_REDUCE_BLOCK_SIZE>>>(
        src,
        dst,
        rows,
        cols
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr,
            "ggml-cuda8/reduce-sum-rows: kernel launch failed: %s (%d)\n",
            cudaGetErrorString(err), (int) err);
        return -1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr,
            "ggml-cuda8/reduce-sum-rows: sync failed: %s (%d)\n",
            cudaGetErrorString(err), (int) err);
        return -1;
    }

    return 0;
}

extern "C" int ggml_cuda8_reduce_max_rows_f32_launch(
    const float * src,
    float * dst,
    int rows,
    int cols
) {
    if (src == NULL || dst == NULL || rows <= 0 || cols <= 0) {
        std::fprintf(stderr,
            "ggml-cuda8/reduce-max-rows: invalid args rows=%d cols=%d\n",
            rows, cols);
        return -1;
    }

    ggml_cuda8_reduce_max_rows_f32_kernel<<<ggml_cuda8_grid_rows(rows), GGML_CUDA8_REDUCE_BLOCK_SIZE>>>(
        src,
        dst,
        rows,
        cols
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr,
            "ggml-cuda8/reduce-max-rows: kernel launch failed: %s (%d)\n",
            cudaGetErrorString(err), (int) err);
        return -1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr,
            "ggml-cuda8/reduce-max-rows: sync failed: %s (%d)\n",
            cudaGetErrorString(err), (int) err);
        return -1;
    }

    return 0;
}
