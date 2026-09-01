// ggml-cuda8-rope.cu  -  G28A: Rotary Positional Embedding kernel
// G45: added NEOX (mode=2). Still no YaRN, no mrope/vision. Fermi-safe.
#include <cuda_runtime.h>
#include <cstdio>
#include <math.h>

#include "ggml-cuda8-grid.cuh"

// One thread per pair of elements across all heads/positions.
// Layout: x[i0 + i1*ne0 + i2*ne0*ne1 + i3*ne0*ne1*ne2]
//   ne0 = head_dim, ne1 = n_heads, ne2 = seq_len, ne3 = batch
//   pos[i2] = position id for sequence index i2
//
// Both rope layouts rotate n_dims/2 pairs and pass through [n_dims, ne0), but
// they differ in which two elements form a pair. Mirrors rotate_pairs() in
// ggml/src/ggml-cpu/ops.cpp:
//
//   NORMAL (mode 0)  rotate_pairs(n_dims, n_offset=1,         scale=1)
//                    pair p -> elements (2p, 2p+1)          [cscscscs]
//   NEOX   (mode 2)  rotate_pairs(n_dims, n_offset=n_dims/2, scale=2)
//                    pair p -> elements (p, p + n_dims/2)   [ccccssss]
//
// theta is indexed by the pair number p in both cases, so only the element
// indices change.
static __global__ void kernel_rope_f32(
        const float * __restrict__ x,
        float       * __restrict__ dst,
        const int   * __restrict__ pos,
        const int ne0,
        const int ne1,
        const int ne2,
        const int ne3,
        const int n_dims,
        const int is_neox,
        const float theta_scale,
        const float freq_scale) {

    const int pairs_per_row = ne0 / 2;
    const int rows = ne1 * ne2 * ne3;
    const int total_pairs = pairs_per_row * rows;
    const int n_pairs_rot = n_dims / 2;

    // G38: grid-stride - the grid is clamped to 65535 blocks on Fermi.
    const int stride = blockDim.x * gridDim.x;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_pairs;
         idx += stride) {

        const int pair = idx % pairs_per_row;
        const int row  = idx / pairs_per_row;

        const int i1 = row % ne1;
        const int i2 = (row / ne1) % ne2;
        const int i3 = row / (ne1 * ne2);

        const int row_base = i1 * ne0
                           + i2 * ne0 * ne1
                           + i3 * ne0 * ne1 * ne2;

        if (pair >= n_pairs_rot) {
            // Beyond rotary dims - pass through unchanged. This region is
            // adjacent-paired in both modes.
            const int i0 = pair * 2;
            dst[row_base + i0]     = x[row_base + i0];
            dst[row_base + i0 + 1] = x[row_base + i0 + 1];
            continue;
        }

        // Element indices for this pair.
        const int ia = is_neox ? pair               : pair * 2;
        const int ib = is_neox ? pair + n_pairs_rot : pair * 2 + 1;

        const int p = pos[i2];
        float theta = (float)p * powf(theta_scale, (float)pair) * freq_scale;
        float cos_t = cosf(theta);
        float sin_t = sinf(theta);

        float x0 = x[row_base + ia];
        float x1 = x[row_base + ib];

        dst[row_base + ia] = x0 * cos_t - x1 * sin_t;
        dst[row_base + ib] = x0 * sin_t + x1 * cos_t;
    }
}

// mode: GGML_ROPE_TYPE_NORMAL (0) or GGML_ROPE_TYPE_NEOX (2).
// Anything else is rejected here as well as in supports_op - the pair layouts
// for MROPE/VISION differ in ways this kernel does not implement.
extern "C" int ggml_cuda8_op_rope_f32(
        const float * x,
        float * dst,
        const int * pos,
        int ne0, int ne1, int ne2, int ne3,
        int n_dims,
        int mode,
        float freq_base,
        float freq_scale) {

    if (x == NULL || dst == NULL || pos == NULL || ne0 < 2) {
        std::fprintf(stderr, "ggml-cuda8/rope: invalid args\n");
        return -1;
    }

    if (mode != 0 && mode != 2) {
        std::fprintf(stderr, "ggml-cuda8/rope: unsupported mode=%d\n", mode);
        return -1;
    }

    if (n_dims < 2 || n_dims > ne0 || (n_dims % 2) != 0) {
        std::fprintf(stderr, "ggml-cuda8/rope: bad n_dims=%d for ne0=%d\n", n_dims, ne0);
        return -1;
    }

    const int is_neox = (mode == 2) ? 1 : 0;

    const float theta_scale = powf(freq_base, -2.0f / (float)n_dims);
    const int total_pairs = (ne0 / 2) * ne1 * ne2 * ne3;
    const int block = 256;
    const int grid  = ggml_cuda8_grid_1d(total_pairs, block);

    kernel_rope_f32<<<grid, block>>>(
        x, dst, pos, ne0, ne1, ne2, ne3,
        n_dims, is_neox, theta_scale, freq_scale);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "ggml-cuda8/rope: launch failed: %s\n",
            cudaGetErrorString(err));
        return -1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "ggml-cuda8/rope: sync failed: %s\n",
            cudaGetErrorString(err));
        return -1;
    }
    return 0;
}
