// ggml/src/ggml-cuda8/ggml-cuda8-add.cu
//
// G9B-2: F32 vector ADD kernel (CUDA8-safe)

#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__ void ggml_cuda8_add_f32_kernel(
    const float * a,
    const float * b,
    float * c,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

extern "C" int ggml_cuda8_add_f32_launch(
    const float * a,
    const float * b,
    float * c,
    int n
) {
    if (!a || !b || !c || n <= 0) {
        std::fprintf(stderr, "ggml-cuda8/add: invalid args\n");
        return -1;
    }

    const int block = 256;
    const int grid  = (n + block - 1) / block;

    ggml_cuda8_add_f32_kernel<<<grid, block>>>(a, b, c, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr,
            "ggml-cuda8/add: kernel launch failed: %s\n",
            cudaGetErrorString(err));
        return -1;
    }

    return cudaDeviceSynchronize() == cudaSuccess ? 0 : -1;
}
