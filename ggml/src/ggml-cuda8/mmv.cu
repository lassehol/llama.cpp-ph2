// ggml/src/ggml-cuda8/mmv.cu
//
// Minimal legacy CUDA8 / Fermi float matrix-vector multiply.
//
// Provides:
//   1) ggml_cuda8_mmv_f32_naive()
//      - one CUDA thread computes one output row
//
//   2) ggml_cuda8_mmv_f32_block()
//      - one CUDA block computes one output row
//      - threads cooperate over columns
//      - shared-memory reduction
//
//   3) ggml_cuda8_mmv_f32()
//      - selector wrapper
//      - uses naive kernel for small cols
//      - uses block kernel for larger cols
//
// A is row-major with shape [rows, cols].
// x has shape [cols].
// y has shape [rows].
//
// CUDA8/Fermi-safe:
//   - no cooperative groups
//   - no warp shuffle
//   - no tensor cores / WMMA
//   - no C++14/C++17 assumptions

#include <cstdio>
#include <cuda_runtime.h>

#include "ggml-cuda8-grid.cuh"

#define GGML_CUDA8_MMV_CHECK(call)                                                 \
    do {                                                                           \
        cudaError_t err__ = (call);                                                \
        if (err__ != cudaSuccess) {                                                \
            std::fprintf(stderr,                                                   \
                "ggml-cuda8/mmv: CUDA error %s (%d) at %s:%d\n",                   \
                cudaGetErrorString(err__), (int) err__, __FILE__, __LINE__);       \
            return -1;                                                             \
        }                                                                          \
    } while (0)

static const int GGML_CUDA8_MMV_BLOCK_SIZE = 128;

// Keep selector conservative.
// For small cols, one-thread-per-row avoids block/reduction overhead.
// For larger cols, block-per-row gives parallelism across the dot product.
static const int GGML_CUDA8_MMV_BLOCK_COLS_THRESHOLD = 32;

// -----------------------------------------------------------------------------
// Naive kernel:
//   one CUDA thread = one output row
// -----------------------------------------------------------------------------

static __global__ void ggml_cuda8_mmv_f32_naive_kernel(
    const float * A,
    const float * x,
    float * y,
    int rows,
    int cols
) {
    // G38: grid-stride - the grid is clamped to 65535 blocks on Fermi.
    const int stride = blockDim.x * gridDim.x;

    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < rows; row += stride) {
        const float * Arow = A + (size_t) row * cols;

        float sum = 0.0f;

        for (int j = 0; j < cols; ++j) {
            sum += Arow[j] * x[j];
        }

        y[row] = sum;
    }
}

extern "C" int ggml_cuda8_mmv_f32_naive(
    const float * d_A,
    const float * d_x,
    float * d_y,
    int rows,
    int cols
) {
    if (d_A == NULL || d_x == NULL || d_y == NULL) {
        std::fprintf(stderr, "ggml-cuda8/mmv naive: null device pointer\n");
        return -1;
    }

    if (rows <= 0 || cols <= 0) {
        std::fprintf(stderr, "ggml-cuda8/mmv naive: invalid shape rows=%d cols=%d\n", rows, cols);
        return -1;
    }

    const int block_size = 128;
    const int grid_size  = ggml_cuda8_grid_1d(rows, block_size);

    ggml_cuda8_mmv_f32_naive_kernel<<<grid_size, block_size>>>(d_A, d_x, d_y, rows, cols);
    GGML_CUDA8_MMV_CHECK(cudaGetLastError());
    GGML_CUDA8_MMV_CHECK(cudaDeviceSynchronize());

    return 0;
}

// -----------------------------------------------------------------------------
// Block-per-row kernel:
//   one CUDA block = one output row
//   blockDim.x threads cooperate over columns
// -----------------------------------------------------------------------------

static __global__ void ggml_cuda8_mmv_f32_block_kernel(
    const float * A,
    const float * x,
    float * y,
    int rows,
    int cols
) {
    const int tid = threadIdx.x;

    __shared__ float partial[GGML_CUDA8_MMV_BLOCK_SIZE];

    // G38: row-stride - the grid is clamped to 65535 blocks on Fermi.
    // The loop bound is uniform across the block (row comes from blockIdx.x
    // plus multiples of gridDim.x), so every thread runs the same number of
    // iterations and the __syncthreads() below stay collective.
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const float * Arow = A + (size_t) row * cols;

        float sum = 0.0f;

        for (int j = tid; j < cols; j += GGML_CUDA8_MMV_BLOCK_SIZE) {
            sum += Arow[j] * x[j];
        }

        partial[tid] = sum;
        __syncthreads();

        // Shared-memory tree reduction.
        // Block size is fixed at 128.
        if (tid < 64) {
            partial[tid] += partial[tid + 64];
        }
        __syncthreads();

        if (tid < 32) {
            partial[tid] += partial[tid + 32];
        }
        __syncthreads();

        if (tid < 16) {
            partial[tid] += partial[tid + 16];
        }
        __syncthreads();

        if (tid < 8) {
            partial[tid] += partial[tid + 8];
        }
        __syncthreads();

        if (tid < 4) {
            partial[tid] += partial[tid + 4];
        }
        __syncthreads();

        if (tid < 2) {
            partial[tid] += partial[tid + 2];
        }
        __syncthreads();

        if (tid < 1) {
            partial[tid] += partial[tid + 1];
        }
        __syncthreads();

        if (tid == 0) {
            y[row] = partial[0];
        }

        // Guard the next iteration's partial[tid] write against threads still
        // reading partial[0] above.
        __syncthreads();
    }
}

extern "C" int ggml_cuda8_mmv_f32_block(
    const float * d_A,
    const float * d_x,
    float * d_y,
    int rows,
    int cols
) {
    if (d_A == NULL || d_x == NULL || d_y == NULL) {
        std::fprintf(stderr, "ggml-cuda8/mmv block: null device pointer\n");
        return -1;
    }

    if (rows <= 0 || cols <= 0) {
        std::fprintf(stderr, "ggml-cuda8/mmv block: invalid shape rows=%d cols=%d\n", rows, cols);
        return -1;
    }

    const int grid_size  = ggml_cuda8_grid_rows(rows);
    const int block_size = GGML_CUDA8_MMV_BLOCK_SIZE;

    ggml_cuda8_mmv_f32_block_kernel<<<grid_size, block_size>>>(d_A, d_x, d_y, rows, cols);
    GGML_CUDA8_MMV_CHECK(cudaGetLastError());
    GGML_CUDA8_MMV_CHECK(cudaDeviceSynchronize());

    return 0;
}

// -----------------------------------------------------------------------------
// Public selector wrapper.
// -----------------------------------------------------------------------------

extern "C" int ggml_cuda8_mmv_f32(
    const float * d_A,
    const float * d_x,
    float * d_y,
    int rows,
    int cols
) {
    if (cols >= GGML_CUDA8_MMV_BLOCK_COLS_THRESHOLD) {
        return ggml_cuda8_mmv_f32_block(d_A, d_x, d_y, rows, cols);
    }

    return ggml_cuda8_mmv_f32_naive(d_A, d_x, d_y, rows, cols);
}

// -----------------------------------------------------------------------------
// Existing smoke-test helper.
// Uses the selector wrapper.
// -----------------------------------------------------------------------------

extern "C" int ggml_cuda8_mmv_f32_roundtrip_test(void) {
    const int rows = 4;
    const int cols = 4;

    const size_t bytes_A = rows * cols * sizeof(float);
    const size_t bytes_x = cols * sizeof(float);
    const size_t bytes_y = rows * sizeof(float);

    float h_A[rows * cols] = {
        1.f,  2.f,  3.f,  4.f,
        5.f,  6.f,  7.f,  8.f,
        9.f, 10.f, 11.f, 12.f,
        13.f,14.f, 15.f, 16.f
    };

    float h_x[cols] = {
        1.f, 1.f, 1.f, 1.f
    };

    float h_y[rows] = {
        0.f, 0.f, 0.f, 0.f
    };

    float * d_A = NULL;
    float * d_x = NULL;
    float * d_y = NULL;

    GGML_CUDA8_MMV_CHECK(cudaMalloc((void **) &d_A, bytes_A));
    GGML_CUDA8_MMV_CHECK(cudaMalloc((void **) &d_x, bytes_x));
    GGML_CUDA8_MMV_CHECK(cudaMalloc((void **) &d_y, bytes_y));

    GGML_CUDA8_MMV_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    GGML_CUDA8_MMV_CHECK(cudaMemcpy(d_x, h_x, bytes_x, cudaMemcpyHostToDevice));
    GGML_CUDA8_MMV_CHECK(cudaMemset(d_y, 0, bytes_y));

    if (ggml_cuda8_mmv_f32(d_A, d_x, d_y, rows, cols) != 0) {
        cudaFree(d_A);
        cudaFree(d_x);
        cudaFree(d_y);
        return -1;
    }

    GGML_CUDA8_MMV_CHECK(cudaMemcpy(h_y, d_y, bytes_y, cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);

    const float expected[rows] = {
        10.f, 26.f, 42.f, 58.f
    };

    for (int i = 0; i < rows; ++i) {
        if (h_y[i] != expected[i]) {
            std::fprintf(stderr,
                "ggml-cuda8/mmv: mismatch at row %d: got=%f expected=%f\n",
                i, h_y[i], expected[i]);
            return -1;
        }
    }

    std::printf("ggml-cuda8/mmv: roundtrip GEMV OK\n");
    return 0;
}
