// ggml/src/ggml-cuda8/ggml-cuda8-cont.cu
// G55: strided-gather CONT. Replaces the flat memcpy that ignored src0
// strides and corrupted every post-permute (non-contiguous) CONT - i.e.
// every attention CONT. dst is contiguous (ggml guarantees this for CONT
// output); src0 is read through its real byte strides nb[0..3].
#include <cuda_runtime.h>
#include <cstdio>

__global__ void kernel_cont_f32(
        const char * __restrict__ src0,   // byte base, read via strides
        float      * __restrict__ dst,    // packed contiguous output
        int ne0, int ne1, int ne2, int ne3,
        size_t nb0, size_t nb1, size_t nb2, size_t nb3,
        long long total) {
    // Grid-stride loop; grid.x clamped to 65535 on Fermi, stride covers rest.
    for (long long idx = (long long) blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (long long) gridDim.x * blockDim.x) {
        long long t = idx;
        const int i0 = (int)(t % ne0); t /= ne0;
        const int i1 = (int)(t % ne1); t /= ne1;
        const int i2 = (int)(t % ne2); t /= ne2;
        const int i3 = (int) t;
        const char * sp = src0
            + (size_t) i0 * nb0
            + (size_t) i1 * nb1
            + (size_t) i2 * nb2
            + (size_t) i3 * nb3;
        dst[idx] = *(const float *) sp;   // dst packed, src0 strided
    }
}

extern "C" int ggml_cuda8_cont_f32_launch(
        const void * src0, float * dst,
        int ne0, int ne1, int ne2, int ne3,
        size_t nb0, size_t nb1, size_t nb2, size_t nb3) {
    const long long total = (long long) ne0 * ne1 * ne2 * ne3;
    if (total <= 0) return 0;
    const int block = 256;
    long long blocks_ll = (total + block - 1) / block;
    int blocks = blocks_ll > 65535 ? 65535 : (int) blocks_ll;
    kernel_cont_f32<<<blocks, block>>>(
        (const char *) src0, dst, ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3, total);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "ggml-cuda8/cont: launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "ggml-cuda8/cont: sync failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}
