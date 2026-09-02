// ggml/src/ggml-cuda8/ggml-cuda8-mulmat-f32.cu
//
// G42: batched, broadcast-aware F32xF32 matrix multiply kernel - the
// attention matmuls (K.Q, probs.V). CUDA8/Fermi-safe: one block per output
// element (i01,i11,i12,i13), shared-memory tree reduction (no warp
// shuffle), same style as ggml-cuda8-q4k.cu / ggml-cuda8-q6k.cu.
//
// Semantics match ggml_mul_mat(a=src0, b=src1):
//   dst[i01, i11, i12, i13] = sum_c src0[c, i01, i12/r2, i13/r3] * src1[c, i11, i12, i13]
//   where r2 = ne12/ne02, r3 = ne13/ne03 (GQA-style head broadcast: src0
//   repeats to match src1's larger head/batch count).
//
// dim 0 (the reduction dimension) is required contiguous on both src0 and
// src1 (nb00/nb10 == sizeof(float), enforced by the caller in
// ggml-cuda8-mulmat-f32.cpp, not re-checked here). Dims 1-3 take explicit
// byte strides rather than being assumed packed, so permuted views (the
// common case for attention K/Q/V after reshape+permute, where the
// permute reorders head/token/batch dims but leaves dim 0 alone) are
// handled directly without a separate CONT copy.
//
// Grid: one block per output element via an explicit 2D dim3 grid
// (blockIdx.x, blockIdx.y), matching the existing Q4_K/Q6_K MUL_MAT kernels'
// grid construction exactly (NOT the ggml_cuda8_grid_rows() +
// grid-stride-loop pattern used by GET_ROWS/softmax/reduce - that pattern
// requires a per-block loop, which this kernel does not have). total_rows
// can safely exceed 65535 via grid.y, and an early return is safe here
// (unlike the stride-loop kernels) because each block is assigned exactly
// one output element with no per-block loop, so no thread can desync
// __syncthreads() from a block-mate by returning early.
#include <cuda_runtime.h>
#include <cstdio>

static const int GGML_CUDA8_MULMAT_F32_BLOCK_SIZE = 256;

static __device__ void ggml_cuda8_mulmat_f32_reduce_sum(float * partial, int tid) {
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial[tid] += partial[tid + s];
        }
        __syncthreads();
    }
}

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
    const int tid = threadIdx.x;
    __shared__ float partial[GGML_CUDA8_MULMAT_F32_BLOCK_SIZE];

    const long long idx = (long long) blockIdx.x + (long long) blockIdx.y * gridDim.x;
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

    float sum = 0.0f;
    for (int c = tid; c < ne00; c += GGML_CUDA8_MULMAT_F32_BLOCK_SIZE) {
        sum += row0[c] * row1[c];
    }
    partial[tid] = sum;
    ggml_cuda8_mulmat_f32_reduce_sum(partial, tid);

    if (tid == 0) {
        float * dst_row =
            (float *) ((char *) dst + (size_t) i11 * nb1
                                     + (size_t) i12 * nb2
                                     + (size_t) i13 * nb3);
        dst_row[i01] = partial[0];
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
        ne00 <= 0 || ne01 <= 0 || ne02 <= 0 || ne03 <= 0 ||
        ne11 <= 0 || ne12 <= 0 || ne13 <= 0) {
        std::fprintf(stderr,
            "ggml-cuda8/mulmat-f32: invalid args ne00=%d ne01=%d ne02=%d ne03=%d "
            "ne11=%d ne12=%d ne13=%d\n",
            ne00, ne01, ne02, ne03, ne11, ne12, ne13);
        return -1;
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
    if (total_rows <= 0 || total_rows > 0x7fffffffLL) {
        std::fprintf(stderr,
            "ggml-cuda8/mulmat-f32: row count out of range (%lld)\n", total_rows);
        return -1;
    }

    // Explicit 2D grid, matching kernel_mul_mat_q4k_f32 / kernel_mul_mat_q6k_f32
    // exactly: grid.x carries up to 65535 blocks, grid.y covers the rest.
    // total_rows is already bounds-checked above to fit safely within
    // grid.x * grid.y's int32 range for any realistic attention shape.
    const int total_rows_i = (int) total_rows;
    dim3 grid(total_rows_i > 65535 ? 65535 : total_rows_i,
               (total_rows_i + 65534) / 65535);
    ggml_cuda8_mul_mat_f32_f32_kernel<<<grid, GGML_CUDA8_MULMAT_F32_BLOCK_SIZE>>>(
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
