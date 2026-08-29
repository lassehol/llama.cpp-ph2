// ggml/src/ggml-cuda8/q8_0-mmv.cu
//
// CUDA8/Fermi-safe Q8_0 matrix-vector multiply, v2.
//
// Provides:
//   - ggml_cuda8_q8_0_mmv_f32_qblock()
//   - ggml_cuda8_q8_0_mmv_f32_col_parallel()
//   - ggml_cuda8_q8_0_mmv_f32()
//   - ggml_cuda8_q8_0_mmv_f32_block()
//
// CUDA8/Fermi-safe:
//   - no cooperative groups
//   - no warp shuffle
//   - no tensor cores / WMMA
//   - no CUDA half arithmetic
//   - manual fp16-bits to float conversion

#include "q8_0-mmv.cuh"

#include <cstdio>
#include <cuda_runtime.h>

#define GGML_CUDA8_Q8_0_CHECK(call)                                                \
    do {                                                                           \
        cudaError_t err__ = (call);                                                \
        if (err__ != cudaSuccess) {                                                \
            std::fprintf(stderr,                                                   \
                "ggml-cuda8/q8_0-mmv: CUDA error %s (%d) at %s:%d\n",              \
                cudaGetErrorString(err__), (int) err__, __FILE__, __LINE__);       \
            return -1;                                                             \
        }                                                                          \
    } while (0)

static const int GGML_CUDA8_Q8_0_MMV_BLOCK_SIZE = 128;
static const int GGML_CUDA8_Q8_0_COL_PARALLEL_THRESHOLD = 32;

static __device__ float ggml_cuda8_fp16_bits_to_float(uint16_t h) {
    const unsigned int h_exp  = (h & 0x7C00u);
    const unsigned int h_sig  = (h & 0x03FFu);
    const unsigned int h_sign = (h & 0x8000u);

    const unsigned int f_sign = h_sign << 16;
    unsigned int f_exp;
    unsigned int f_sig;

    if (h_exp == 0) {
        if (h_sig == 0) {
            return __uint_as_float(f_sign);
        }

        unsigned int sig = h_sig;
        int exp = -14;

        while ((sig & 0x0400u) == 0) {
            sig <<= 1;
            --exp;
        }

        sig &= 0x03FFu;
        f_exp = (unsigned int) (exp + 127) << 23;
        f_sig = sig << 13;

        return __uint_as_float(f_sign | f_exp | f_sig);
    }

    if (h_exp == 0x7C00u) {
        f_exp = 0xFFu << 23;
        f_sig = h_sig << 13;
        return __uint_as_float(f_sign | f_exp | f_sig);
    }

    const int exp = (int) (h_exp >> 10) - 15;
    f_exp = (unsigned int) (exp + 127) << 23;
    f_sig = h_sig << 13;

    return __uint_as_float(f_sign | f_exp | f_sig);
}

static __device__ void ggml_cuda8_reduce_128(float * partial, int tid) {
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

// -----------------------------------------------------------------------------
// Kernel A: qblock baseline.
// One CUDA thread processes one Q8_0 block of 32 values.
// -----------------------------------------------------------------------------

static __global__ void ggml_cuda8_q8_0_mmv_f32_qblock_kernel(
    const ggml_cuda8_q8_0_block * Aq,
    const float * x,
    float * y,
    int rows,
    int cols,
    int blocks_per_row
) {
    // 2D grid for Fermi (max grid.x=65535)
    const int row = blockIdx.x + blockIdx.y * gridDim.x;
    const int tid = threadIdx.x;

    __shared__ float partial[GGML_CUDA8_Q8_0_MMV_BLOCK_SIZE];

    float sum = 0.0f;

    if (row < rows) {
        const ggml_cuda8_q8_0_block * row_blocks =
            Aq + (size_t) row * blocks_per_row;

        for (int ib = tid; ib < blocks_per_row; ib += GGML_CUDA8_Q8_0_MMV_BLOCK_SIZE) {
            const ggml_cuda8_q8_0_block b = row_blocks[ib];
            const float d = ggml_cuda8_fp16_bits_to_float(b.d);

            const int base_col = ib * GGML_CUDA8_QK8_0;
            float block_sum = 0.0f;

            #pragma unroll
            for (int k = 0; k < GGML_CUDA8_QK8_0; ++k) {
                const int col = base_col + k;
                if (col < cols) {
                    block_sum += ((float) b.qs[k]) * x[col];
                }
            }

            sum += d * block_sum;
        }
    }

    partial[tid] = sum;
    __syncthreads();

    ggml_cuda8_reduce_128(partial, tid);

    if (tid == 0 && row < rows) {
        y[row] = partial[0];
    }
}

extern "C" int ggml_cuda8_q8_0_mmv_f32_qblock(
    const ggml_cuda8_q8_0_block * d_Aq,
    const float * d_x,
    float * d_y,
    int rows,
    int cols
) {
    if (d_Aq == NULL || d_x == NULL || d_y == NULL) {
        std::fprintf(stderr, "ggml-cuda8/q8_0-mmv qblock: null device pointer\n");
        return -1;
    }

    if (rows <= 0 || cols <= 0) {
        std::fprintf(stderr,
            "ggml-cuda8/q8_0-mmv qblock: invalid shape rows=%d cols=%d\n",
            rows, cols);
        return -1;
    }

    const int blocks_per_row =
        (cols + GGML_CUDA8_QK8_0 - 1) / GGML_CUDA8_QK8_0;

    ggml_cuda8_q8_0_mmv_f32_qblock_kernel<<<dim3(rows > 65535 ? 65535 : rows, (rows + 65534) / 65535), GGML_CUDA8_Q8_0_MMV_BLOCK_SIZE>>>(
        d_Aq, d_x, d_y, rows, cols, blocks_per_row
    );

    GGML_CUDA8_Q8_0_CHECK(cudaGetLastError());
    GGML_CUDA8_Q8_0_CHECK(cudaDeviceSynchronize());

    return 0;
}

// -----------------------------------------------------------------------------
// Kernel B: column-parallel v2.
// All 128 threads cooperate over individual columns.
// -----------------------------------------------------------------------------

static __global__ void ggml_cuda8_q8_0_mmv_f32_col_parallel_kernel(
    const ggml_cuda8_q8_0_block * Aq,
    const float * x,
    float * y,
    int rows,
    int cols,
    int blocks_per_row
) {
    // 2D grid for Fermi (max grid.x=65535)
    const int row = blockIdx.x + blockIdx.y * gridDim.x;
    const int tid = threadIdx.x;

    __shared__ float partial[GGML_CUDA8_Q8_0_MMV_BLOCK_SIZE];

    float sum = 0.0f;

    if (row < rows) {
        const ggml_cuda8_q8_0_block * row_blocks =
            Aq + (size_t) row * blocks_per_row;

        for (int col = tid; col < cols; col += GGML_CUDA8_Q8_0_MMV_BLOCK_SIZE) {
            const int ib = col / GGML_CUDA8_QK8_0;
            const int k  = col - ib * GGML_CUDA8_QK8_0;

            const ggml_cuda8_q8_0_block * b = row_blocks + ib;

            const float d = ggml_cuda8_fp16_bits_to_float(b->d);
            const int q = (int) b->qs[k];

            sum += d * ((float) q) * x[col];
        }
    }

    partial[tid] = sum;
    __syncthreads();

    ggml_cuda8_reduce_128(partial, tid);

    if (tid == 0 && row < rows) {
        y[row] = partial[0];
    }
}

extern "C" int ggml_cuda8_q8_0_mmv_f32_col_parallel(
    const ggml_cuda8_q8_0_block * d_Aq,
    const float * d_x,
    float * d_y,
    int rows,
    int cols
) {
    if (d_Aq == NULL || d_x == NULL || d_y == NULL) {
        std::fprintf(stderr, "ggml-cuda8/q8_0-mmv col_parallel: null device pointer\n");
        return -1;
    }

    if (rows <= 0 || cols <= 0) {
        std::fprintf(stderr,
            "ggml-cuda8/q8_0-mmv col_parallel: invalid shape rows=%d cols=%d\n",
            rows, cols);
        return -1;
    }

    const int blocks_per_row =
        (cols + GGML_CUDA8_QK8_0 - 1) / GGML_CUDA8_QK8_0;

    ggml_cuda8_q8_0_mmv_f32_col_parallel_kernel<<<dim3(rows > 65535 ? 65535 : rows, (rows + 65534) / 65535), GGML_CUDA8_Q8_0_MMV_BLOCK_SIZE>>>(
        d_Aq, d_x, d_y, rows, cols, blocks_per_row
    );

    GGML_CUDA8_Q8_0_CHECK(cudaGetLastError());
    GGML_CUDA8_Q8_0_CHECK(cudaDeviceSynchronize());

    return 0;
}

// Compatibility alias for previous function name.
extern "C" int ggml_cuda8_q8_0_mmv_f32_block(
    const ggml_cuda8_q8_0_block * d_Aq,
    const float * d_x,
    float * d_y,
    int rows,
    int cols
) {
    return ggml_cuda8_q8_0_mmv_f32_qblock(d_Aq, d_x, d_y, rows, cols);
}

// Public selector wrapper.
extern "C" int ggml_cuda8_q8_0_mmv_f32(
    const ggml_cuda8_q8_0_block * d_Aq,
    const float * d_x,
    float * d_y,
    int rows,
    int cols
) {
    if (cols >= GGML_CUDA8_Q8_0_COL_PARALLEL_THRESHOLD) {
        return ggml_cuda8_q8_0_mmv_f32_col_parallel(d_Aq, d_x, d_y, rows, cols);
    }

    return ggml_cuda8_q8_0_mmv_f32_qblock(d_Aq, d_x, d_y, rows, cols);
}

extern "C" int ggml_cuda8_q8_0_mmv_roundtrip_test(void) {
    const int rows = 2;
    const int cols = 4;
    const int blocks_per_row = 1;

    ggml_cuda8_q8_0_block h_Aq[rows * blocks_per_row];

    for (int i = 0; i < rows * blocks_per_row; ++i) {
        h_Aq[i].d = 0x3C00u;

        for (int k = 0; k < GGML_CUDA8_QK8_0; ++k) {
            h_Aq[i].qs[k] = 0;
        }
    }

    h_Aq[0].qs[0] = 1;
    h_Aq[0].qs[1] = 2;
    h_Aq[0].qs[2] = 3;
    h_Aq[0].qs[3] = 4;

    h_Aq[1].qs[0] = 5;
    h_Aq[1].qs[1] = 6;
    h_Aq[1].qs[2] = 7;
    h_Aq[1].qs[3] = 8;

    float h_x[cols] = { 1.0f, 1.0f, 1.0f, 1.0f };
    float h_y[rows] = { 0.0f, 0.0f };

    ggml_cuda8_q8_0_block * d_Aq = NULL;
    float * d_x = NULL;
    float * d_y = NULL;

    const size_t bytes_Aq = sizeof(h_Aq);
    const size_t bytes_x  = sizeof(h_x);
    const size_t bytes_y  = sizeof(h_y);

    GGML_CUDA8_Q8_0_CHECK(cudaMalloc((void **) &d_Aq, bytes_Aq));
    GGML_CUDA8_Q8_0_CHECK(cudaMalloc((void **) &d_x,  bytes_x));
    GGML_CUDA8_Q8_0_CHECK(cudaMalloc((void **) &d_y,  bytes_y));

    GGML_CUDA8_Q8_0_CHECK(cudaMemcpy(d_Aq, h_Aq, bytes_Aq, cudaMemcpyHostToDevice));
    GGML_CUDA8_Q8_0_CHECK(cudaMemcpy(d_x,  h_x,  bytes_x,  cudaMemcpyHostToDevice));
    GGML_CUDA8_Q8_0_CHECK(cudaMemset(d_y, 0, bytes_y));

    if (ggml_cuda8_q8_0_mmv_f32(d_Aq, d_x, d_y, rows, cols) != 0) {
        cudaFree(d_Aq);
        cudaFree(d_x);
        cudaFree(d_y);
        return -1;
    }

    GGML_CUDA8_Q8_0_CHECK(cudaMemcpy(h_y, d_y, bytes_y, cudaMemcpyDeviceToHost));

    cudaFree(d_Aq);
    cudaFree(d_x);
    cudaFree(d_y);

    const float expected[rows] = { 10.0f, 26.0f };

    for (int i = 0; i < rows; ++i) {
        const float diff = h_y[i] - expected[i];
        const float abs_diff = diff < 0.0f ? -diff : diff;

        if (abs_diff > 1e-5f) {
            std::fprintf(stderr,
                "ggml-cuda8/q8_0-mmv: mismatch row=%d got=%f expected=%f\n",
                i, h_y[i], expected[i]);
            return -1;
        }
    }

    std::printf("ggml-cuda8/q8_0-mmv: roundtrip OK\n");
    return 0;
}
