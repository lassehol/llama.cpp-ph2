// ggml-cuda8-q6k. Q6_K MUL_MAT and GET_ROWS kernels for Fermi/Keplercu 
// Required for Q4_K_M models (~25% of layers use Q6_K)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <stdint.h>

#include "ggml-cuda8-grid.cuh"

#define QK_K 256

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
__global__ void __launch_bounds__(256, 2) kernel_mul_mat_q6k_f32(
        const void * __restrict__ src0, const float * __restrict__ src1, float * __restrict__ dst,
        const int ne00, const int ne01, const int ne11, const int nb01, const int nb11) {
    const int idx = blockIdx.x + blockIdx.y * gridDim.x;
    if (idx >= ne01 * ne11) return;
    const int row = idx % ne01;
    const int col = idx / ne01;
    const int tid = threadIdx.x;            // 0..255
    const int nb  = ne00 / QK_K;
    const block_q6_K * src0_row = (const block_q6_K *)((const char *)src0 + (size_t)row * nb01);
    const float * src1_col = (const float *)((const char *)src1 + (size_t)col * nb11);

    // G60: decode tid -> (half, l, k), the same (half,l,q1..q4) enumeration
    // the serial loop used, one term per thread. 2*32*4 = 256 exactly.
    const int half = tid >> 7;              // tid / 128 -> 0..1
    const int rem  = tid & 127;             // 0..127
    const int l    = rem >> 2;              // 0..31
    const int k    = rem & 3;               // 0..3  (which of q1..q4)
    const int is   = l >> 4;                // 0..1

    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {                        // ALL 256 threads, every block
        const block_q6_K * blk = &src0_row[b];
        const float d = fp16_to_f32(blk->d);
        const uint8_t * ql = blk->ql + half*64;
        const uint8_t * qh = blk->qh + half*32;
        const int8_t  * sc = blk->scales + half*8;
        const float   * s1h = src1_col + b*QK_K + half*128;

        // k selects the quadrant exactly as q1..q4 in the serial version:
        //  k=0: ql[l]   low , qh bits 0-1, scale is+0, s1h[l+ 0]
        //  k=1: ql[l+32]low , qh bits 2-3, scale is+2, s1h[l+32]
        //  k=2: ql[l]   high, qh bits 4-5, scale is+4, s1h[l+64]
        //  k=3: ql[l+32]high, qh bits 6-7, scale is+6, s1h[l+96]
        int ql_idx, sh, sc_off, s_off;
        bool high;
        switch (k) {
            case 0: ql_idx = l;      sh = 0; sc_off = 0; s_off =  0; high = false; break;
            case 1: ql_idx = l + 32; sh = 2; sc_off = 2; s_off = 32; high = false; break;
            case 2: ql_idx = l;      sh = 4; sc_off = 4; s_off = 64; high = true;  break;
            default:ql_idx = l + 32; sh = 6; sc_off = 6; s_off = 96; high = true;  break;
        }
        const uint8_t qlv = high ? (ql[ql_idx] >> 4) : (ql[ql_idx] & 0xF);
        const int8_t  q   = (int8_t)(qlv | (((qh[l] >> sh) & 3) << 4)) - 32;
        sum += d * (float) sc[is + sc_off] * (float) q * s1h[l + s_off];
    }
    __shared__ float smem[256];
    smem[tid] = sum;
    block_reduce_sum_inplace_q6(smem, tid);
    if (tid == 0) dst[col * ne01 + row] = smem[0];
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
    int total_q6k = ne01 * ne11;
    dim3 grid(total_q6k > 65535 ? 65535 : total_q6k, (total_q6k + 65534) / 65535);
    kernel_mul_mat_q6k_f32<<<grid, 256>>>(src0, src1, dst, ne00, ne01, ne11, nb01, nb11);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ggml-cuda8/q6k mul_mat: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

} // extern "C"
