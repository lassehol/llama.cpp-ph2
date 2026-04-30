// ggml/src/ggml-cuda8/ggml-cuda8-scalar.cu
//
// G9B-3: F32 scalar ADD/MUL kernels.
// CUDA8/Fermi-safe.

#include <cuda_runtime.h>
#include <cstdio>

static __global__ void ggml_cuda8_add_scalar_f32_kernel(
    const float * src,
    float scalar,
    float * dst,
    int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        dst[i] = src[i] + scalar;
    }
}

static __global__ void ggml_cuda8_mul_scalar_f32_kernel(
    const float * src,
    float scalar,
    float * dst,
    int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        dst[i] = src[i] * scalar;
    }
}

extern "C" int ggml_cuda8_add_scalar_f32_launch(
    const float * src,
    float scalar,
    float * dst,
    int n
) {
    if (src == NULL || dst == NULL || n <= 0) {
        std::fprintf(stderr, "ggml-cuda8/scalar-add: invalid args\n");
        return -1;
    }

    const int block = 256;
    const int grid  = (n + block - 1) / block;

    ggml_cuda8_add_scalar_f32_kernel<<<grid, block>>>(src, scalar, dst, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr,
            "ggml-cuda8/scalar-add: kernel launch failed: %s (%d)\n",
            cudaGetErrorString(err), (int) err);
        return -1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr,
            "ggml-cuda8/scalar-add: sync failed: %s (%d)\n",
            cudaGetErrorString(err), (int) err);
        return -1;
    }

    return 0;
}

extern "C" int ggml_cuda8_mul_scalar_f32_launch(
    const float * src,
    float scalar,
    float * dst,
    int n
) {
    if (src == NULL || dst == NULL || n <= 0) {
        std::fprintf(stderr, "ggml-cuda8/scalar-mul: invalid args\n");
        return -1;
    }

    const int block = 256;
    const int grid  = (n + block - 1) / block;

    ggml_cuda8_mul_scalar_f32_kernel<<<grid, block>>>(src, scalar, dst, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr,
            "ggml-cuda8/scalar-mul: kernel launch failed: %s (%d)\n",
            cudaGetErrorString(err), (int) err);
        return -1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr,
            "ggml-cuda8/scalar-mul: sync failed: %s (%d)\n",
            cudaGetErrorString(err), (int) err);
        return -1;
    }

    return 0;
}
