// ggml/src/ggml-cuda8/ggml-cuda8-mulmat-f32.cu
//
// G42: batched, broadcast-aware F32xF32 matrix multiply kernel - the
// attention matmuls (K.Q, probs.V).
//
// G58: warp-per-output-element rewrite. The G42 kernel used one 256-thread
// block per output element with a 256-wide __syncthreads tree reduction.
// For attention, ne00 (the reduction dim = head_dim, ~128, or n_kv) is small,
// so a 256-thread block left half its threads idle and spent most of its time
// in the reduction, not the dot product. Once the K-quant weight matmuls were
// warp-optimized (G56/G57), this kernel became ~52% of all GPU time at
// 0.92 ms/call - 5x slower than the tuned K-quant matmuls. This rewrite gives
// it the same treatment: one warp (32 threads) per output element, 8 warps
// (256 threads) per block = 8 elements/block, the 32 lanes split the ne00
// reduction, and a 32-wide warp-synchronous reduction (volatile shared, no
// __syncthreads within a warp on Fermi's lockstep warps) replaces the
// 256-wide tree.
//
// Semantics unchanged from G42, match ggml_mul_mat(a=src0, b=src1):
//   dst[i01, i11, i12, i13] = sum_c src0[c, i01, i12/r2, i13/r3] * src1[c, i11, i12, i13]
//   where r2 = ne12/ne02, r3 = ne13/ne03 (GQA-style head broadcast).
//
// dim 0 (the reduction dimension) is required contiguous on both src0 and
// src1 (enforced by the caller in ggml-cuda8-mulmat-f32.cpp). Dims 1-3 take
// explicit byte strides, so permuted attention views are handled directly.
#include <cuda_runtime.h>
#include <cstdio>

#define MMF32_WARP 32
#define MMF32_WARPS_PER_BLOCK 8
#define MMF32_BLOCK_THREADS (MMF32_WARP * MMF32_WARPS_PER_BLOCK)   // 256

static __global__ void ggml_cuda8_mul_mat_f32_f32_kernel(
    const float * __restrict__ src0,
    const float * __restrict__ src1,
    float * __restrict__ dst,
    int ne00,
    int ne01, int ne02, int ne03,
    int ne11, int ne12, int ne13,
    size_t nb01, size_t nb02, size_t nb03,   // src0 strides (bytes), dim0 implicitly sizeof(float)
    size_t nb11, size_t nb12, size_t nb13,   // src1 strides (bytes)
    size_t nb1,  size_t nb2,  size_t nb3,    // dst strides (bytes); dst dim0 implicitly sizeof(float)
    int r2, int r3,
    long long total_rows
) {
    const int tid     = threadIdx.x;
    const int lane    = tid & (MMF32_WARP - 1);   // 0..31
    const int warp_id = tid >> 5;                 // 0..7

    // One warp per output element. block_idx counts blocks (2D grid for the
    // Fermi 65535 limit); each block owns 8 consecutive output elements.
    const long long block_idx = (long long) blockIdx.x + (long long) blockIdx.y * gridDim.x;
    const long long idx = block_idx * MMF32_WARPS_PER_BLOCK + warp_id;
    // idx is uniform within a warp, so an out-of-range warp returns as a whole
    // - safe for the warp-synchronous reduction below (no thread desyncs).
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

    const float * row0 =
        (const float *) ((const char *) src0 + (size_t) i01 * nb01
                                              + (size_t) i02 * nb02
                                              + (size_t) i03 * nb03);
    const float * row1 =
        (const float *) ((const char *) src1 + (size_t) i11 * nb11
                                              + (size_t) i12 * nb12
                                              + (size_t) i13 * nb13);

    // The 32 lanes split the reduction dim; consecutive lanes read consecutive
    // floats (row0[lane], row0[lane+32], ...) -> coalesced within the warp.
    float sum = 0.0f;
    for (int c = lane; c < ne00; c += MMF32_WARP) {
        sum += row0[c] * row1[c];
    }

    // 32-wide warp-synchronous reduction. Each warp occupies smem[warp_id*32 ..
    // warp_id*32+31] and reduces only its own 32 slots; warps are independent.
    __shared__ volatile float smem[MMF32_BLOCK_THREADS];
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

extern "C" int ggml_cuda8_mul_mat_f32_f32_launch(
    const float * src0,
    const float * src1,
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
            "ggml-cuda8/mulmat-f32: broadcast mismatch ne02=%d ne12=%d ne03=%d ne13=%d\n",
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
            "ggml-cuda8/mulmat-f32: row count out of range (%lld)\n", total_rows);
        return -1;
    }

    // G58: one warp per output element, 8 warps/block -> ceil(total_rows / 8)
    // blocks. 2D grid keeps each dim under the Fermi 65535 limit.
    const long long nblocks =
        (total_rows + MMF32_WARPS_PER_BLOCK - 1) / MMF32_WARPS_PER_BLOCK;
    const int gx = (int) (nblocks > 65535 ? 65535 : nblocks);
    const int gy = (int) ((nblocks + 65534) / 65535);
    dim3 grid(gx, gy);
    ggml_cuda8_mul_mat_f32_f32_kernel<<<grid, MMF32_BLOCK_THREADS>>>(
        src0, src1, dst,
        ne00, ne01, ne02, ne03, ne11, ne12, ne13,
        nb01, nb02, nb03,
        nb11, nb12, nb13,
        nb1, nb2, nb3,
        r2, r3,
        total_rows
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "ggml-cuda8/mulmat-f32: launch failed: %s (%d)\n", cudaGetErrorString(err), (int) err);
        return -1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "ggml-cuda8/mulmat-f32: sync failed: %s (%d)\n", cudaGetErrorString(err), (int) err);
        return -1;
    }
    return 0;
}
