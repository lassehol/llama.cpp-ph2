// ggml-cuda8-q4k. Q4_K MUL_MAT and GET_ROWS kernels for Fermi/Keplercu 
// CUDA 8, compute capability 2.x-3.x (GTX 560)
// No __shfl_sync, no __half arithmetic, no __ldg on cc 2.x

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <stdint.h>

#include "ggml-cuda8-grid.cuh"

#define QK_K 256
#define K_SCALE_SIZE 12

// Q4_K block: 144 bytes -> 256 values
// Must match ggml-common.h layout exactly
struct block_q4_K {
    uint16_t d;          // FP16 as bits (super-block scale)
    uint16_t dmin;       // FP16 as bits (super-block minimum)
    uint8_t  scales[K_SCALE_SIZE]; // 12 bytes: packed 6-bit scales+mins
    uint8_t  qs[QK_K/2];          // 128 bytes: 4-bit quants
};

// Fermi-safe FP16 -> float (no __half arithmetic)
__device__ __forceinline__ float fp16_to_f32(uint16_t h) {
    __half tmp;
    memcpy(&tmp, &h, sizeof(tmp));
    return __half2float(tmp);
}

// Unpack 6-bit scale and min for sub-block j (0..7)
__device__ __forceinline__ void get_scale_min_k4(
        int j, const uint8_t * __restrict__ scales,
        uint8_t &sc, uint8_t &mn) {
    if (j < 4) {
        sc = scales[j]     & 63;
        mn = scales[j + 4] & 63;
    } else {
        sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        mn = (scales[j + 4] >> 4)  | ((scales[j]     >> 6) << 4);
    }
}

// Shared memory tree reduction (Fermi-safe, no warp shuffles)
__device__ __forceinline__ void block_reduce_sum_inplace(float * smem, int tid) {
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
}

// =====================================================================
// GET_ROWS Q4_K: dequantize selected rows
// Grid:  (n_tokens, 1)
// Block: (256, 1)
// =====================================================================
__global__ void kernel_get_rows_q4k(
        const void  * __restrict__ src0,
        const int   * __restrict__ src1,
        float       * __restrict__ dst,
        const int ne00,
        const int n_tokens,
        const int nb01,
        const int nb1) {

    const int tid = threadIdx.x;

    // G38: row-stride - the grid is clamped to 65535 blocks on Fermi.
    for (int token = blockIdx.x; token < n_tokens; token += gridDim.x) {
        const int row_idx = src1[token];

        const int nb = ne00 / QK_K;  // Q4_K blocks per row
        const block_q4_K * row = (const block_q4_K *)((const char *)src0 + (size_t)row_idx * nb01);
        float * out = (float *)((char *)dst + (size_t)token * nb1);

        // Each thread processes full blocks with stride
        for (int b = tid; b < nb; b += blockDim.x) {
            const block_q4_K * blk = &row[b];
            const float d    = fp16_to_f32(blk->d);
            const float dmin = fp16_to_f32(blk->dmin);

            for (int j = 0; j < 8; j++) {
                uint8_t sc, mn;
                get_scale_min_k4(j, blk->scales, sc, mn);
                const float scale = d * (float)sc;
                const float min   = dmin * (float)mn;
                const int g = j / 2;
                const int is_high = j & 1;

                for (int l = 0; l < 32; l++) {
                    uint8_t qbyte = blk->qs[g * 32 + l];
                    uint8_t q = is_high ? (qbyte >> 4) : (qbyte & 0xF);
                    out[b * QK_K + j * 32 + l] = scale * (float)q - min;
                }
            }
        }
    }
}

// =====================================================================
// MUL_MAT Q4_K x F32 -> F32
// Grid:  (ne01,  one block per output elementne11) 
// Block: (256, 1)
// =====================================================================
__global__ void kernel_mul_mat_q4k_f32(
        const void * __restrict__ src0, const float * __restrict__ src1, float * __restrict__ dst,
        const int ne00, const int ne01, const int ne11, const int nb01, const int nb11) {
    const int idx = blockIdx.x + blockIdx.y * gridDim.x;
    if (idx >= ne01 * ne11) return;
    const int row = idx % ne01;
    const int col = idx / ne01;
    const int tid = threadIdx.x;            // 0..255
    const int nb  = ne00 / QK_K;
    const block_q4_K * src0_row = (const block_q4_K *)((const char *)src0 + (size_t)row * nb01);
    const float * src1_col = (const float *)((const char *)src1 + (size_t)col * nb11);

    // G59: every thread handles ONE of the 256 values in each block.
    // tid = j*32 + l   (j = sub-block 0..7, l = position 0..31)
    const int j = tid >> 5;          // tid / 32  -> 0..7
    const int l = tid & 31;          // tid % 32  -> 0..31
    const int g = j >> 1;            // j / 2
    const int is_high = j & 1;

    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {                        // ← ALL 256 threads, every block
        const block_q4_K * blk = &src0_row[b];
        const float d    = fp16_to_f32(blk->d);
        const float dmin = fp16_to_f32(blk->dmin);
        uint8_t sc, mn;
        get_scale_min_k4(j, blk->scales, sc, mn);
        const float scale = d * (float) sc;
        const float mmin  = dmin * (float) mn;
        const uint8_t qbyte = blk->qs[g * 32 + l];
        const uint8_t q = is_high ? (qbyte >> 4) : (qbyte & 0xF);
        const float w = scale * (float) q - mmin;
        sum += w * src1_col[b * QK_K + j * 32 + l];
    }
    __shared__ float smem[256];
    smem[tid] = sum;
    block_reduce_sum_inplace(smem, tid);
    if (tid == 0) dst[col * ne01 + row] = smem[0];
}

// =====================================================================
// Host wrappers (C linkage)
// =====================================================================

extern "C" {

int ggml_cuda8_op_get_rows_q4k(
        const void * src0, const int * src1, float * dst,
        int ne00, int n_tokens, int nb01, int nb1) {
    kernel_get_rows_q4k<<<ggml_cuda8_grid_rows(n_tokens), 256>>>(
        src0, src1, dst, ne00, n_tokens, nb01, nb1);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ggml-cuda8/q4k get_rows: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

int ggml_cuda8_op_mul_mat_q4k_f32(
        const void * src0, const float * src1, float * dst,
        int ne00, int ne01, int ne11,
        int nb01, int nb11) {
    int total_q4k = ne01 * ne11;
    dim3 grid(total_q4k > 65535 ? 65535 : total_q4k, (total_q4k + 65534) / 65535);
    kernel_mul_mat_q4k_f32<<<grid, 256>>>(src0, src1, dst, ne00, ne01, ne11, nb01, nb11);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ggml-cuda8/q4k mul_mat: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

} // extern "C"
