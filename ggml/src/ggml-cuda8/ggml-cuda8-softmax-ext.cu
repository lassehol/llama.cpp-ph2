// ggml/src/ggml-cuda8/ggml-cuda8-softmax-ext.cu
//
// G41: masked / scaled / ALiBi-biased row-wise softmax kernel.
// CUDA8/Fermi-safe: one block per (i01,i02,i03) row, 128 threads,
// shared-memory reductions (no warp shuffle, same pattern as
// ggml-cuda8-softmax.cu's plain kernel).
//
// Computes, per row:
//     v[c]   = src[c] * scale + (mask ? slope(head) * mask[c] : 0)
//     dst[c] = softmax(v)[c]
//
// slope(head) is the standard ALiBi per-head slope (mirrors
// ggml_compute_forward_soft_max_f32 in ggml-cpu/ops.cpp):
//     n_head_log2 = largest power of two <= n_head
//     m0 = 2^(-max_bias / n_head_log2)
//     m1 = 2^(-max_bias/2 / n_head_log2)
//     slope(h) = max_bias <= 0 ? 1
//              : h < n_head_log2 ? m0^(h+1)
//                                : m1^(2*(h-n_head_log2)+1)
// m0/m1/n_head_log2 are computed once on the host (they only depend on
// n_head and max_bias, not on the row), matching how upstream backends
// precompute them rather than recomputing powf() per row.
//
// Explicitly NOT implemented, refused upstream in supports_op /
// ggml_cuda8_soft_max_is_supported_ext() before this kernel is ever reached:
//   - attention sinks (src[2])
//   - F16 mask
// Both would silently produce wrong attention weights if accepted without
// being applied - the same failure mode G37 fixed for the unguarded plain
// kernel. This kernel does not read a "sinks" pointer or an F16 mask at all,
// so there is nothing here that could silently ignore them; the guard lives
// entirely in the caller.
#include <cuda_runtime.h>
#include <cstdio>
#include <float.h>
#include <math.h>
#include "ggml-cuda8-grid.cuh"

static const int GGML_CUDA8_SOFTMAX_EXT_BLOCK_SIZE = 128;

static __device__ void ggml_cuda8_ext_reduce_max_128(float * partial, int tid) {
    if (tid < 64) { partial[tid] = partial[tid] > partial[tid + 64] ? partial[tid] : partial[tid + 64]; } __syncthreads();
    if (tid < 32) { partial[tid] = partial[tid] > partial[tid + 32] ? partial[tid] : partial[tid + 32]; } __syncthreads();
    if (tid < 16) { partial[tid] = partial[tid] > partial[tid + 16] ? partial[tid] : partial[tid + 16]; } __syncthreads();
    if (tid < 8)  { partial[tid] = partial[tid] > partial[tid + 8]  ? partial[tid] : partial[tid + 8];  } __syncthreads();
    if (tid < 4)  { partial[tid] = partial[tid] > partial[tid + 4]  ? partial[tid] : partial[tid + 4];  } __syncthreads();
    if (tid < 2)  { partial[tid] = partial[tid] > partial[tid + 2]  ? partial[tid] : partial[tid + 2];  } __syncthreads();
    if (tid < 1)  { partial[tid] = partial[tid] > partial[tid + 1]  ? partial[tid] : partial[tid + 1];  } __syncthreads();
}
static __device__ void ggml_cuda8_ext_reduce_sum_128(float * partial, int tid) {
    if (tid < 64) { partial[tid] += partial[tid + 64]; } __syncthreads();
    if (tid < 32) { partial[tid] += partial[tid + 32]; } __syncthreads();
    if (tid < 16) { partial[tid] += partial[tid + 16]; } __syncthreads();
    if (tid < 8)  { partial[tid] += partial[tid + 8];  } __syncthreads();
    if (tid < 4)  { partial[tid] += partial[tid + 4];  } __syncthreads();
    if (tid < 2)  { partial[tid] += partial[tid + 2];  } __syncthreads();
    if (tid < 1)  { partial[tid] += partial[tid + 1];  } __syncthreads();
}

// Row value at column c, recomputed on each of the three passes (max / sum /
// write) rather than cached, exactly like the plain kernel already does -
// cols can be large (n_kv can be thousands), so there is no shared-memory
// budget to hold a full row.
static __device__ __forceinline__ float ggml_cuda8_ext_row_value(
        const float * row_src, const float * row_mask, int c,
        float scale, float slope) {
    float v = row_src[c] * scale;
    if (row_mask != NULL) {
        v += slope * row_mask[c];
    }
    return v;
}

static __global__ void ggml_cuda8_softmax_ext_f32_kernel(
    const float * __restrict__ src,
    const float * __restrict__ mask,   // may be NULL
    float * __restrict__ dst,
    int ne00, int ne01, int ne02, int ne03,
    size_t nb01, size_t nb02, size_t nb03,             // src0 strides (bytes)
    size_t dst_nb1, size_t dst_nb2, size_t dst_nb3,    // dst strides (bytes)
    int mask_ne1, int mask_ne2, int mask_ne3,          // ignored if mask==NULL
    size_t mask_nb1, size_t mask_nb2, size_t mask_nb3, // mask strides (bytes)
    float scale, float max_bias,
    int n_head_log2, float m0, float m1,
    int total_rows
) {
    const int tid = threadIdx.x;
    __shared__ float partial[GGML_CUDA8_SOFTMAX_EXT_BLOCK_SIZE];
    // G38-style row-stride: grid is clamped to 65535 blocks on Fermi. No
    // early return inside the loop (same reasoning as the plain kernel's
    // G38 fix) - an early-returning thread would desync __syncthreads()
    // from its block-mates.
    for (int row = blockIdx.x; row < total_rows; row += gridDim.x) {
        const int i01 = row % ne01;
        const int tmp = row / ne01;
        const int i02 = tmp % ne02;
        const int i03 = tmp / ne02;

        const float * row_src =
            (const float *) ((const char *) src + (size_t) i01 * nb01
                                                 + (size_t) i02 * nb02
                                                 + (size_t) i03 * nb03);
        float * row_dst =
            (float *) ((char *) dst + (size_t) i01 * dst_nb1
                                     + (size_t) i02 * dst_nb2
                                     + (size_t) i03 * dst_nb3);

        const float * row_mask = NULL;
        float slope = 1.0f;
        if (mask != NULL) {
            const int m1_idx = i01 % mask_ne1;
            const int m2_idx = i02 % mask_ne2;
            const int m3_idx = i03 % mask_ne3;
            row_mask = (const float *) ((const char *) mask
                            + (size_t) m1_idx * mask_nb1
                            + (size_t) m2_idx * mask_nb2
                            + (size_t) m3_idx * mask_nb3);
            if (max_bias > 0.0f) {
                const int h = i02;
                slope = (h < n_head_log2)
                    ? powf(m0, (float) (h + 1))
                    : powf(m1, (float) (2 * (h - n_head_log2) + 1));
            }
        }

        float vmax = -FLT_MAX;
        for (int c = tid; c < ne00; c += GGML_CUDA8_SOFTMAX_EXT_BLOCK_SIZE) {
            const float v = ggml_cuda8_ext_row_value(row_src, row_mask, c, scale, slope);
            vmax = vmax > v ? vmax : v;
        }
        partial[tid] = vmax;
        __syncthreads();
        ggml_cuda8_ext_reduce_max_128(partial, tid);
        const float row_max = partial[0];
        __syncthreads();

        float sum = 0.0f;
        for (int c = tid; c < ne00; c += GGML_CUDA8_SOFTMAX_EXT_BLOCK_SIZE) {
            const float v = ggml_cuda8_ext_row_value(row_src, row_mask, c, scale, slope);
            sum += expf(v - row_max);
        }
        partial[tid] = sum;
        __syncthreads();
        ggml_cuda8_ext_reduce_sum_128(partial, tid);
        const float row_sum = partial[0];
        const float inv_sum = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;
        __syncthreads();

        for (int c = tid; c < ne00; c += GGML_CUDA8_SOFTMAX_EXT_BLOCK_SIZE) {
            const float v = ggml_cuda8_ext_row_value(row_src, row_mask, c, scale, slope);
            row_dst[c] = expf(v - row_max) * inv_sum;
        }
        // Guard the next iteration's partial[tid] write (same reasoning as
        // the plain kernel's G38 fix).
        __syncthreads();
    }
}

extern "C" int ggml_cuda8_softmax_ext_f32_launch(
        const float * src,
        const float * mask,    // may be NULL
        float * dst,
        int ne00, int ne01, int ne02, int ne03,
        size_t nb01, size_t nb02, size_t nb03,
        size_t dst_nb1, size_t dst_nb2, size_t dst_nb3,
        int mask_ne1, int mask_ne2, int mask_ne3,
        size_t mask_nb1, size_t mask_nb2, size_t mask_nb3,
        float scale, float max_bias
) {
    if (src == NULL || dst == NULL ||
        ne00 <= 0 || ne01 <= 0 || ne02 <= 0 || ne03 <= 0) {
        std::fprintf(stderr,
            "ggml-cuda8/softmax-ext: invalid args ne00=%d ne01=%d ne02=%d ne03=%d\n",
            ne00, ne01, ne02, ne03);
        return -1;
    }
    // ALiBi slope precomputation - mirrors ggml_compute_forward_soft_max_f32
    // (ggml-cpu/ops.cpp): n_head_log2 is the largest power of two <= n_head,
    // m0/m1 depend only on n_head and max_bias, not on the row, so they are
    // computed once here rather than per-thread with powf() inside the
    // kernel.
    const int n_head = ne02;
    int n_head_log2 = 1;
    while (n_head_log2 * 2 <= n_head) {
        n_head_log2 *= 2;
    }
    const float m0 = powf(2.0f, -(max_bias) / (float) n_head_log2);
    const float m1 = powf(2.0f, -(max_bias * 0.5f) / (float) n_head_log2);

    const long long total_rows_ll = (long long) ne01 * (long long) ne02 * (long long) ne03;
    if (total_rows_ll <= 0 || total_rows_ll > 0x7fffffffLL) {
        std::fprintf(stderr,
            "ggml-cuda8/softmax-ext: row count out of range (%lld)\n", total_rows_ll);
        return -1;
    }
    const int total_rows = (int) total_rows_ll;

    ggml_cuda8_softmax_ext_f32_kernel<<<ggml_cuda8_grid_rows(total_rows), GGML_CUDA8_SOFTMAX_EXT_BLOCK_SIZE>>>(
        src, mask, dst,
        ne00, ne01, ne02, ne03,
        nb01, nb02, nb03,
        dst_nb1, dst_nb2, dst_nb3,
        mask_ne1, mask_ne2, mask_ne3,
        mask_nb1, mask_nb2, mask_nb3,
        scale, max_bias, n_head_log2, m0, m1,
        total_rows
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr,
            "ggml-cuda8/softmax-ext: launch failed: %s (%d)\n",
            cudaGetErrorString(err), (int) err);
        return -1;
    }
    // G56 sync-batching test: per-op cudaDeviceSynchronize() removed; rely on
    // the segment-end cuda8_backend_synchronize() instead.
    return 0;
}