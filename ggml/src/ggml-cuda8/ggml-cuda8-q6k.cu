// ggml-cuda8-q6k.cu  Q6_K MUL_MAT and GET_ROWS kernels for Fermi/Kepler
// Required for Q4_K_M models (~25% of layers use Q6_K)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <stdint.h>
#include "ggml-cuda8-grid.cuh"

#define QK_K 256

// G57: warp-per-row MUL_MAT tiling.
#define Q6K_WARP 32
#define Q6K_ROWS_PER_BLOCK 8
#define Q6K_BLOCK_THREADS (Q6K_WARP * Q6K_ROWS_PER_BLOCK)   // 256

// Q6_K block: 210 bytes -> 256 values
struct block_q6_K {
    uint8_t ql[QK_K/2];     // 128 bytes: lower 4 bits
    uint8_t qh[QK_K/4];     //  64 bytes: upper 2 bits
    int8_t  scales[QK_K/16]; //  16 bytes: 8-bit scales
    uint16_t d;              //   2 bytes: FP16 super-block scale
};

__device__ __forceinline__ float fp16_to_f32(uint16_t h) {
    __half tmp;
    memcpy(&tmp, &h, sizeof(tmp));
    return __half2float(tmp);
}

__device__ __forceinline__ void block_reduce_sum_inplace_q6(float * smem, int tid) {
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
}

// Dequantize one Q6_K value at position idx (0..255) within a block
__device__ __forceinline__ float dequant_q6k_val(
        const block_q6_K * blk, int idx, float d) {
    const int half  = idx / 128;         // 0 or 1
    const int local = idx - half * 128;  // 0..127 within half
    const uint8_t * ql = blk->ql + half * 64;
    const uint8_t * qh = blk->qh + half * 32;
    const int8_t  * sc = blk->scales + half * 8;
    const int sub  = local / 16;  // sub-block 0..7
    const int pos  = local % 16;  // position within sub-block
    const int flat = sub * 16 + pos;  // 0..127
    // Lower 4 bits from ql: 2 values per byte
    uint8_t ql_val = (ql[flat / 2] >> ((flat % 2) * 4)) & 0xF;
    // Upper 2 bits from qh: 4 values per byte
    uint8_t qh_val = (qh[flat / 4] >> ((flat % 4) * 2)) & 0x3;
    // 6-bit quant, offset by 32
    int8_t q6 = (int8_t)(ql_val | (qh_val << 4)) - 32;
    return d * (float)sc[sub] * (float)q6;
}

// =====================================================================
// GET_ROWS Q6_K
// =====================================================================
__global__ void kernel_get_rows_q6k(
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
        const int nb = ne00 / QK_K;
        const block_q6_K * row = (const block_q6_K *)((const char *)src0 + (size_t)row_idx * nb01);
        float * out = (float *)((char *)dst + (size_t)token * nb1);
        for (int b = tid; b < nb; b += blockDim.x) {
            const block_q6_K * blk = &row[b];
            const float d = fp16_to_f32(blk->d);
            for (int i = 0; i < QK_K; i++) {
                out[b * QK_K + i] = dequant_q6k_val(blk, i, d);
            }
        }
    }
}

// =====================================================================
// MUL_MAT Q6_K x F32 -> F32
// =====================================================================
// G57: warp-per-row. 8 warps/block = 8 output rows/block; lane = l (0..31),
// each thread does 2 halves x 4 quadrants = 8 values per superblock;
// 32-wide warp-synchronous reduction. Same transform as the Q4_K warp-per-row
// kernel, adapted to Q6_K's ql/qh/quadrant layout. Replaces the G60
// one-term-per-thread kernel + 256-wide __syncthreads reduction; the
// __launch_bounds__(256,2) cap is dropped so the scheduler can pick
// occupancy freely.
__global__ void kernel_mul_mat_q6k_f32(
        const void * __restrict__ src0, const float * __restrict__ src1, float * __restrict__ dst,
        const int ne00, const int ne01, const int ne11, const int nb01, const int nb11) {
    const int tid     = threadIdx.x;            // 0..255
    const int lane    = tid & (Q6K_WARP - 1);   // 0..31  == l
    const int warp_id = tid >> 5;               // 0..7

    const int row = blockIdx.x * Q6K_ROWS_PER_BLOCK + warp_id;
    const int col = blockIdx.y;
    // row/col are uniform within a warp, so an out-of-range warp returns as a
    // whole - safe for the warp-synchronous reduction below.
    if (row >= ne01 || col >= ne11) return;

    const int nb = ne00 / QK_K;
    const block_q6_K * src0_row = (const block_q6_K *)((const char *)src0 + (size_t)row * nb01);
    const float * src1_col = (const float *)((const char *)src1 + (size_t)col * nb11);

    const int l  = lane;         // 0..31
    const int is = l >> 4;       // l / 16 -> 0..1

    float sum = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const block_q6_K * blk = &src0_row[b];
        const float d = fp16_to_f32(blk->d);
        const float * s1b = src1_col + b * QK_K;
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const uint8_t * ql = blk->ql + half * 64;
            const uint8_t * qh = blk->qh + half * 32;
            const int8_t  * sc = blk->scales + half * 8;
            const float   * s1h = s1b + half * 128;

            // The four quadrants, exactly as the G60 serial q1..q4 enumeration:
            //  q1: ql[l]    low , qh bits 0-1, scale is+0, s1h[l+ 0]
            //  q2: ql[l+32] low , qh bits 2-3, scale is+2, s1h[l+32]
            //  q3: ql[l]    high, qh bits 4-5, scale is+4, s1h[l+64]
            //  q4: ql[l+32] high, qh bits 6-7, scale is+6, s1h[l+96]
            const int8_t q1 = (int8_t)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((ql[l]      >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;

            sum += d * (float) sc[is + 0] * (float) q1 * s1h[l +  0];
            sum += d * (float) sc[is + 2] * (float) q2 * s1h[l + 32];
            sum += d * (float) sc[is + 4] * (float) q3 * s1h[l + 64];
            sum += d * (float) sc[is + 6] * (float) q4 * s1h[l + 96];
        }
    }

    // 32-wide warp-synchronous reduction (Fermi lockstep warp, volatile
    // shared, no __syncthreads within a warp). Each warp reduces its own 32
    // slots; warps are independent.
    __shared__ volatile float smem[Q6K_BLOCK_THREADS];
    smem[tid] = sum;
    if (lane < 16) smem[tid] += smem[tid + 16];
    if (lane <  8) smem[tid] += smem[tid +  8];
    if (lane <  4) smem[tid] += smem[tid +  4];
    if (lane <  2) smem[tid] += smem[tid +  2];
    if (lane <  1) smem[tid] += smem[tid +  1];
    if (lane == 0) {
        dst[(size_t) col * ne01 + row] = smem[tid];   // G53 layout: col*ne01 + row
    }
}

// =====================================================================
// Host wrappers
// =====================================================================
extern "C" {

int ggml_cuda8_op_get_rows_q6k(
        const void * src0, const int * src1, float * dst,
        int ne00, int n_tokens, int nb01, int nb1) {
    kernel_get_rows_q6k<<<ggml_cuda8_grid_rows(n_tokens), 256>>>(
        src0, src1, dst, ne00, n_tokens, nb01, nb1);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ggml-cuda8/q6k get_rows: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

int ggml_cuda8_op_mul_mat_q6k_f32(
        const void * src0, const float * src1, float * dst,
        int ne00, int ne01, int ne11,
        int nb01, int nb11) {
    // G57: grid = (ceil(ne01 / rows_per_block), ne11). ceil(ne01/8) stays well
    // under the 65535 grid.x limit for any realistic model.
    dim3 grid((ne01 + Q6K_ROWS_PER_BLOCK - 1) / Q6K_ROWS_PER_BLOCK,
              ne11 > 0 ? ne11 : 1);
    kernel_mul_mat_q6k_f32<<<grid, Q6K_BLOCK_THREADS>>>(
        src0, src1, dst, ne00, ne01, ne11, nb01, nb11);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ggml-cuda8/q6k mul_mat: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

} // extern "C"
