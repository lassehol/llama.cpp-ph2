// ggml/src/ggml-cuda8/ggml-cuda8-mulmat-f16.cu
//
// G49 (increment 2): F16-src0 x F32-src1 -> F32 attention matmul.
//
// The F16 read path for an F16 KV cache. When the cache is stored F16 (G49,
// half the bytes -> larger context fits in 1 GB), the attention matmuls read
// K / V as F16 and Q / probs as F32. Fermi has no fp16 ARITHMETIC, but
// __half2float is a single cvt.f32.f16 hardware instruction at sm_21 (outside
// the >=530 arithmetic guards - backport doc section 7), so this reads F16,
// converts per element, and accumulates in F32.
//
// Deliberately a SEPARATE kernel from ggml-cuda8-mulmat-f32.cu: that F32 hot
// path (0.13 ms/call after the G58 warp rewrite) is left completely untouched.
// This variant is identical in structure - warp-per-output-element, 8 warps/
// block, 32-wide warp-synchronous reduction, GQA broadcast, permuted-view
// dim-1-3 strides - the ONLY difference is src0 is __half and each read is
// converted to float in the dot loop.
//
// Semantics match ggml_mul_mat(a=src0/F16, b=src1/F32):
//   dst[i01,i11,i12,i13] = sum_c float(src0[c,i01,i12/r2,i13/r3]) * src1[c,i11,i12,i13]
//
// src0 dim 0 stride is implicitly sizeof(__half) = 2 bytes (contiguous
// reduction dim); nb01/nb02/nb03 are the F16 tensor's real byte strides.
// src1 dim 0 stride is implicitly sizeof(float). Both enforced by the caller.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

#define MMF16_WARP 32
#define MMF16_WARPS_PER_BLOCK 8
#define MMF16_BLOCK_THREADS (MMF16_WARP * MMF16_WARPS_PER_BLOCK)   // 256

static __global__ void ggml_cuda8_mul_mat_f16_f32_kernel(
    const __half * __restrict__ src0,   // F16 (KV cache)
    const float  * __restrict__ src1,   // F32 (Q / probs)
    float        * __restrict__ dst,
    int ne00,
    int ne01, int ne02, int ne03,
    int ne11, int ne12, int ne13,
    size_t nb01, size_t nb02, size_t nb03,   // src0 strides (bytes), dim0 implicitly sizeof(__half)
    size_t nb11, size_t nb12, size_t nb13,   // src1 strides (bytes), dim0 implicitly sizeof(float)
    size_t nb1,  size_t nb2,  size_t nb3,    // dst strides (bytes); dst dim0 implicitly sizeof(float)
    int r2, int r3,
    long long total_rows
) {
    const int tid     = threadIdx.x;
    const int lane    = tid & (MMF16_WARP - 1);   // 0..31
    const int warp_id = tid >> 5;                 // 0..7

    const long long block_idx = (long long) blockIdx.x + (long long) blockIdx.y * gridDim.x;
    const long long idx = block_idx * MMF16_WARPS_PER_BLOCK + warp_id;
    // idx uniform within a warp -> whole warp returns together (safe reduction).
    if (idx >= total_rows) {
        return;
    }

    long long tmp = idx;
    const int i01 = (int) (tmp % ne01); tmp /= ne01;
    const int i11 = (int) (tmp % ne11); tmp /= ne11;
    const int i12 = (int) (tmp % ne12); tmp /= ne12;
    const int i13 = (int) tmp;
    const int i02 = i12 / r2;
    const int i03 = i13 / r3;

    const __half * row0 =
        (const __half *) ((const char *) src0 + (size_t) i01 * nb01
                                              + (size_t) i02 * nb02
                                              + (size_t) i03 * nb03);
    const float * row1 =
        (const float *) ((const char *) src1 + (size_t) i11 * nb11
                                              + (size_t) i12 * nb12
                                              + (size_t) i13 * nb13);

    // The 32 lanes split the reduction dim; consecutive lanes read consecutive
    // halves (row0[lane], row0[lane+32], ...) -> coalesced 2-byte reads.
    float sum = 0.0f;
    for (int c = lane; c < ne00; c += MMF16_WARP) {
        sum += __half2float(row0[c]) * row1[c];   // F16 -> F32 convert, F32 accumulate
    }

    // 32-wide warp-synchronous reduction (Fermi lockstep warp, volatile shared).
    __shared__ volatile float smem[MMF16_BLOCK_THREADS];
    smem[tid] = sum;
    if (lane < 16) smem[tid] += smem[tid + 16];
    if (lane <  8) smem[tid] += smem[tid +  8];
    if (lane <  4) smem[tid] += smem[tid +  4];
    if (lane <  2) smem[tid] += smem[tid +  2];
    if (lane <  1) smem[tid] += smem[tid +  1];

    if (lane == 0) {
        float * dst_row =
            (float *) ((char *) dst + (size_t) i11 * nb1
                                     + (size_t) i12 * nb2
                                     + (size_t) i13 * nb3);
        dst_row[i01] = smem[tid];
    }
}

extern "C" int ggml_cuda8_mul_mat_f16_f32_launch(
    const void  * src0,    // F16 tensor
    const float * src1,    // F32 tensor
    float * dst,
    int ne00,
    int ne01, int ne02, int ne03,
    int ne11, int ne12, int ne13,
    size_t nb01, size_t nb02, size_t nb03,
    size_t nb11, size_t nb12, size_t nb13,
    size_t nb1,  size_t nb2,  size_t nb3
) {
    if (src0 == NULL || src1 == NULL || dst == NULL ||
        ne00 <= 0 || ne01 < 0 || ne02 <= 0 || ne03 <= 0 ||
        ne11 < 0 || ne12 <= 0 || ne13 <= 0) {
        std::fprintf(stderr,
            "ggml-cuda8/mulmat-f32: invalid args ne00=%d ne01=%d ne02=%d ne03=%d "
            "ne11=%d ne12=%d ne13=%d\n",
            ne00, ne01, ne02, ne03, ne11, ne12, ne13);
        return -1;
    }
    // ne01/ne11 == 0: a zero-sized sub-batch row/token count (e.g. a batch
    // split that leaves no positions for this graph fragment) - not an
    // error. Nothing to compute; the caller's dst allocation for the zero
    // dimension is itself zero-sized, so there is nothing to write either.
    if (ne01 == 0 || ne11 == 0) {
        return 0;
    }
    if (ne12 % ne02 != 0 || ne13 % ne03 != 0) {
        std::fprintf(stderr,
            "ggml-cuda8/mulmat-f16: broadcast mismatch ne02=%d ne12=%d ne03=%d ne13=%d\n",
            ne02, ne12, ne03, ne13);
        return -1;
    }
    const int r2 = ne12 / ne02;
    const int r3 = ne13 / ne03;

    const long long total_rows = (long long) ne01 * ne11 * ne12 * ne13;
    // total_rows == 0 is unreachable here (ne01/ne11 == 0 already returned
    // above; ne02/ne03/ne12/ne13 are guarded > 0), but keep the guard exact
    // rather than assuming - only genuinely negative/overflowing is an error.
    if (total_rows < 0 || total_rows > 0x7fffffffLL) {

        std::fprintf(stderr,
            "ggml-cuda8/mulmat-f16: row count out of range (%lld)\n", total_rows);
        return -1;
    }

    // Same grid as the F32 variant: one warp per output element, 8 warps/block.
    const long long nblocks =
        (total_rows + MMF16_WARPS_PER_BLOCK - 1) / MMF16_WARPS_PER_BLOCK;
    const int gx = (int) (nblocks > 65535 ? 65535 : nblocks);
    const int gy = (int) ((nblocks + 65534) / 65535);
    dim3 grid(gx, gy);
    ggml_cuda8_mul_mat_f16_f32_kernel<<<grid, MMF16_BLOCK_THREADS>>>(
        (const __half *) src0, src1, dst,
        ne00, ne01, ne02, ne03, ne11, ne12, ne13,
        nb01, nb02, nb03,
        nb11, nb12, nb13,
        nb1, nb2, nb3,
        r2, r3,
        total_rows
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "ggml-cuda8/mulmat-f16: launch failed: %s (%d)\n", cudaGetErrorString(err), (int) err);
        return -1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "ggml-cuda8/mulmat-f16: sync failed: %s (%d)\n", cudaGetErrorString(err), (int) err);
        return -1;
    }
    return 0;
}
