// ggml-cuda8-mul.cu  -  G26A: element-wise F32 MUL kernel (Fermi-safe)
#include <cuda_runtime.h>
#include <cstdio>

#include "ggml-cuda8-grid.cuh"

static __global__ void kernel_mul_f32(
        const float * __restrict__ a,
        const float * __restrict__ b,
        float * __restrict__ c,
        const int n) {
    // G38: grid-stride - the grid is clamped to 65535 blocks on Fermi.
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        c[i] = a[i] * b[i];
    }
}

extern "C" int ggml_cuda8_mul_f32_launch(
        const float * a,
        const float * b,
        float * c,
        int n) {
    if (a == NULL || b == NULL || c == NULL || n < 0) {
        std::fprintf(stderr, "ggml-cuda8/mul: invalid args\n");
        return -1;
    }
    if (n == 0) {
        return 0;   // nothing to do - zero-sized sub-batch, not an error
    }

    const int block = 256;
    const int grid  = ggml_cuda8_grid_1d(n, block);

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
    // G38: grid-stride - the grid is clamped to 65535 blocks on Fermi.
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_total; i += stride) {
        c[i] = a[i] * b[i % n_repeat];
    }
}

extern "C" int ggml_cuda8_mul_broadcast_f32_launch(
        const float * a,
        const float * b,
        float * c,
        int n_total,
        int n_repeat) {
    if (a == NULL || b == NULL || c == NULL || n_total < 0 || n_repeat <= 0) {
        std::fprintf(stderr, "ggml-cuda8/mul_broadcast: invalid args "
            "(a=%p b=%p c=%p n_total=%d n_repeat=%d)\n",
            (void*)a, (void*)b, (void*)c, n_total, n_repeat);
        return -1;
    }
    // n_total == 0: nothing to multiply (e.g. a zero-sized sub-batch tensor
    // in llama.cpp's batched decode graph construction) - not an error.
    // Skip the launch entirely.
    if (n_total == 0) {
        return 0;
    }
    const int block = 256;
    const int grid  = ggml_cuda8_grid_1d(n_total, block);
    kernel_mul_broadcast_f32<<<grid, block>>>(a, b, c, n_total, n_repeat);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "ggml-cuda8/mul_broadcast: launch failed: %s\n",
            cudaGetErrorString(err));
        return -1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "ggml-cuda8/mul_broadcast: sync failed: %s\n",
            cudaGetErrorString(err));
        return -1;
    }
    return 0;
}
