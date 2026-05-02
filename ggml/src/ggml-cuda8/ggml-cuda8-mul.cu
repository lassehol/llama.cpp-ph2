// ggml-cuda8-mul.cu  -  G26A: element-wise F32 MUL kernel (Fermi-safe)
#include <cuda_runtime.h>
#include <cstdio>

static __global__ void kernel_mul_f32(
        const float * __restrict__ a,
        const float * __restrict__ b,
        float * __restrict__ c,
        const int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

extern "C" int ggml_cuda8_mul_f32_launch(
        const float * a,
        const float * b,
        float * c,
        int n) {
    if (a == NULL || b == NULL || c == NULL || n <= 0) {
        std::fprintf(stderr, "ggml-cuda8/mul: invalid args\n");
        return -1;
    }

    const int block = 256;
    const int grid  = (n + block - 1) / block;

    kernel_mul_f32<<<grid, block>>>(a, b, c, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "ggml-cuda8/mul: launch failed: %s\n",
            cudaGetErrorString(err));
        return -1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "ggml-cuda8/mul: sync failed: %s\n",
            cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

// G37: broadcast MUL kernel  (row-wise broadcast: src1 repeats along rows)
static __global__ void kernel_mul_broadcast_f32(
        const float * __restrict__ a,
        const float * __restrict__ b,
        float * __restrict__ c,
        const int n_total,
        const int n_repeat) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_total) {
        c[i] = a[i] * b[i % n_repeat];
    }
}

extern "C" int ggml_cuda8_mul_broadcast_f32_launch(
        const float * a,
        const float * b,
        float * c,
        int n_total,
        int n_repeat) {
    if (a == NULL || b == NULL || c == NULL || n_total <= 0 || n_repeat <= 0) {
        return -1;
    }

    const int block = 256;
    const int grid  = (n_total + block - 1) / block;

    kernel_mul_broadcast_f32<<<grid, block>>>(a, b, c, n_total, n_repeat);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -1;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return -1;
    return 0;
}
