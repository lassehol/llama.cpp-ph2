// ggml-cuda8-rms-norm.cu  -  G24A: RMS_NORM kernel (Fermi-safe)
#include <cuda_runtime.h>

__global__ void kernel_rms_norm_f32(
        const float * __restrict__ x,
        float       * __restrict__ y,
        const int ncols,
        const float eps) {

    const int row = blockIdx.x;
    const float * x_row = x + (size_t)row * ncols;
    float       * y_row = y + (size_t)row * ncols;

    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;

    // partial sum-of-squares
    float sum_sq = 0.0f;
    for (int col = tid; col < ncols; col += bs) {
        float v = x_row[col];
        sum_sq += v * v;
    }
    sdata[tid] = sum_sq;
    __syncthreads();

    // tree reduction in shared memory (no warp shuffle on Fermi)
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // normalize:  y_i = x_i * rsqrt( mean(x^2) + eps )
    const float scale = rsqrtf(sdata[0] / (float)ncols + eps);
    for (int col = tid; col < ncols; col += bs) {
        y_row[col] = x_row[col] * scale;
    }
}

void ggml_cuda8_rms_norm_f32(
        const float * x, float * y,
        int nrows, int ncols, float eps,
        cudaStream_t stream) {
    const int block_size = 256;
    const size_t smem = block_size * sizeof(float);
    kernel_rms_norm_f32<<<nrows, block_size, smem, stream>>>(x, y, ncols, eps);
}

// -- G24A: extern "C" dispatch wrapper (called from .cpp) ---------------------
extern "C" int ggml_cuda8_op_rms_norm_f32(
        const float * x, float * y,
        int nrows, int ncols, float eps) {
    ggml_cuda8_rms_norm_f32(x, y, nrows, ncols, eps, 0);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : -1;
}
