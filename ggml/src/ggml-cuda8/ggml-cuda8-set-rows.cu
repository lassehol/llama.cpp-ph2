// ggml-cuda8-set-rows.cu  -  G43: SET_ROWS scatter. Fermi-safe.
//
//     dst[idx[i]] = src0[i]
//
// Mirrors ggml_compute_forward_set_rows_impl (ggml/src/ggml-cpu/ops.cpp) for
// the F32 -> F32 case.
//
// This is what writes the KV cache. Without it, running with the cache on the
// GPU aborts in the scheduler:
//
//     pre-allocated tensor (cache_k_l0) in a buffer (CUDA8_0) that cannot run
//     the operation (SET_ROWS)
//
// because the destination is pre-allocated in our buffer and no other backend
// accepts that buffer type. `-nkvo` was the workaround; this removes the need
// for it, which in turn stops llama.cpp pinning the whole attention block to
// the CPU (llama-graph.cpp: `if (!cparams.offload_kqv)`).
//
// F16 destinations are deliberately not handled - that needs the F16 store
// path from G49. Until then the cache must be F32 (--cache-type-k/v f32),
// otherwise supports_op refuses the op and the abort above returns.
//
// Indices are int64_t (llama-kv-cache.cpp creates GGML_TYPE_I64), and the
// index tensor broadcasts over dims 2 and 3: i11 = i02 % ne11, i12 = i03 % ne12.

#include <cuda_runtime.h>
#include <cstdio>
#include <stdint.h>

#include "ggml-cuda8-grid.cuh"

static __global__ void kernel_set_rows_f32(
        const char * __restrict__ src0,
        const char * __restrict__ idx,
        char       * __restrict__ dst,
        const int nc,      // row width (ne00)
        const int nr,      // rows to scatter (ne01)
        const int ne02,
        const int ne03,
        const int ne11,
        const int ne12,
        const int ne1,     // dst row count, for the bounds check
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1,  const size_t nb2,  const size_t nb3) {

    const int total = nc * nr * ne02 * ne03;

    // G38: grid-stride - the grid is clamped to 65535 blocks on Fermi.
    const int stride = blockDim.x * gridDim.x;

    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < total; t += stride) {
        const int col  = t % nc;
        int rest       = t / nc;
        const int i    = rest % nr;
        rest           = rest / nr;
        const int i02  = rest % ne02;
        const int i03  = rest / ne02;

        const int i12 = i03 % ne12;
        const int i11 = i02 % ne11;
        const int i10 = i;

        const int64_t i1 = *(const int64_t *) (idx + (size_t) i10 * nb10
                                                   + (size_t) i11 * nb11
                                                   + (size_t) i12 * nb12);

        // The CPU path asserts here. A kernel cannot, and writing anyway would
        // corrupt memory outside the destination, so skip instead.
        if (i1 < 0 || i1 >= (int64_t) ne1) {
            continue;
        }

        const float * src_row = (const float *) (src0 + (size_t) i   * nb01
                                                      + (size_t) i02 * nb02
                                                      + (size_t) i03 * nb03);
        float * dst_row       = (float *)       (dst  + (size_t) i1  * nb1
                                                      + (size_t) i02 * nb2
                                                      + (size_t) i03 * nb3);

        dst_row[col] = src_row[col];
    }
}

extern "C" int ggml_cuda8_op_set_rows_f32(
        const void * src0,
        const void * idx,
        void * dst,
        int nc, int nr, int ne02, int ne03,
        int ne11, int ne12, int ne1,
        size_t nb01, size_t nb02, size_t nb03,
        size_t nb10, size_t nb11, size_t nb12,
        size_t nb1,  size_t nb2,  size_t nb3) {

    if (src0 == NULL || idx == NULL || dst == NULL ||
        nc <= 0 || nr <= 0 || ne02 <= 0 || ne03 <= 0) {
        std::fprintf(stderr, "ggml-cuda8/set_rows: invalid args nc=%d nr=%d\n", nc, nr);
        return -1;
    }

    if (ne11 <= 0 || ne12 <= 0) {
        std::fprintf(stderr, "ggml-cuda8/set_rows: bad idx dims ne11=%d ne12=%d\n", ne11, ne12);
        return -1;
    }

    const int total = nc * nr * ne02 * ne03;
    const int block = 256;
    const int grid  = ggml_cuda8_grid_1d(total, block);

    kernel_set_rows_f32<<<grid, block>>>(
        (const char *) src0, (const char *) idx, (char *) dst,
        nc, nr, ne02, ne03, ne11, ne12, ne1,
        nb01, nb02, nb03,
        nb10, nb11, nb12,
        nb1,  nb2,  nb3);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "ggml-cuda8/set_rows: launch failed: %s\n",
            cudaGetErrorString(err));
        return -1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "ggml-cuda8/set_rows: sync failed: %s\n",
            cudaGetErrorString(err));
        return -1;
    }
    return 0;
}
