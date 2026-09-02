// ggml-cuda8-swiglu.cu  -  G40: SwiGLU gated activation. Fermi-safe.
//
//     dst[i] = silu(gate[i]) * up[i],   silu(x) = x / (1 + exp(-x))
//
// Mirrors ggml_compute_forward_swiglu_f32 + ggml_vec_swiglu_f32
// (ggml/src/ggml-cpu/ops.cpp, vec.cpp).
//
// GGML_OP_GLU comes in two shapes, and both are handled here:
//
//   split  (src1 != NULL)  ggml_swiglu_split(a, b) - gate = src0, up = src1,
//                          nc = src0->ne[0]. This is what build_ffn emits for
//                          LLM_FFN_PAR, i.e. what Qwen3 and most LLaMA-family
//                          models actually use.
//   halves (src1 == NULL)  ggml_swiglu(a) - gate and up are the two halves of
//                          each src0 row, nc = src0->ne[0]/2. `swapped`
//                          selects which half is which.
//
// Row strides are passed explicitly (in floats) rather than assuming full
// contiguity: ggml only guarantees ggml_is_contiguous_1 here, meaning each row
// is contiguous but rows may be padded.

#include <cuda_runtime.h>
#include <cstdio>
#include <math.h>

#include "ggml-cuda8-grid.cuh"

static __global__ void kernel_swiglu_f32(
        const float * __restrict__ src0,
        const float * __restrict__ src1,   // may be NULL (halves form)
        float       * __restrict__ dst,
        const int nc,
        const int nrows,
        const int src0_stride,
        const int src1_stride,
        const int dst_stride,
        const int swapped) {

    const int total = nc * nrows;

    // G38: grid-stride - the grid is clamped to 65535 blocks on Fermi.
    const int stride = blockDim.x * gridDim.x;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        const int row = idx / nc;
        const int col = idx - row * nc;

        const float * gate;
        const float * up;

        if (src1 != NULL) {
            gate = src0 + (size_t) row * src0_stride;
            up   = src1 + (size_t) row * src1_stride;
        } else {
            const float * r = src0 + (size_t) row * src0_stride;
            gate = r + (swapped ? nc : 0);
            up   = r + (swapped ? 0  : nc);
        }

        const float x = gate[col];
        const float s = x / (1.0f + expf(-x));

        dst[(size_t) row * dst_stride + col] = s * up[col];
    }
}

extern "C" int ggml_cuda8_op_swiglu_f32(
        const float * src0,
        const float * src1,        // NULL for the halves form
        float * dst,
        int nc,
        int nrows,
        int src0_stride,
        int src1_stride,
        int dst_stride,
        int swapped) {

    if (src0 == NULL || dst == NULL || nc <= 0 || nrows < 0) {
        std::fprintf(stderr, "ggml-cuda8/swiglu: invalid args nc=%d nrows=%d\n", nc, nrows);
        return -1;
    }
    if (nrows == 0) {
        return 0;   // nothing to do - zero-sized sub-batch, not an error
    }

    const int total = nc * nrows;
    const int block = 256;
    const int grid  = ggml_cuda8_grid_1d(total, block);

    kernel_swiglu_f32<<<grid, block>>>(
        src0, src1, dst, nc, nrows,
        src0_stride, src1_stride, dst_stride, swapped);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "ggml-cuda8/swiglu: launch failed: %s\n",
            cudaGetErrorString(err));
        return -1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "ggml-cuda8/swiglu: sync failed: %s\n",
            cudaGetErrorString(err));
        return -1;
    }
    return 0;
}
