// ggml-cuda8-rms-norm-smoke.cu  -  G24A: RMS_NORM standalone smoke test
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

// Forward-declare the kernel wrapper directly (no header dependency)
void ggml_cuda8_rms_norm_f32(
        const float * x, float * y,
        int nrows, int ncols, float eps,
        cudaStream_t stream);

#define CHECK_CUDA(x) do { \
    cudaError_t err_ = (x); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d  %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err_)); exit(1); } \
} while (0)

// -- CPU reference ------------------------------------------------------------
static void rms_norm_ref(const float *x, float *y,
                          int nrows, int ncols, float eps) {
    for (int r = 0; r < nrows; r++) {
        const float *xr = x + r * ncols;
        float       *yr = y + r * ncols;
        float ss = 0.0f;
        for (int c = 0; c < ncols; c++) ss += xr[c] * xr[c];
        float scale = 1.0f / sqrtf(ss / (float)ncols + eps);
        for (int c = 0; c < ncols; c++) yr[c] = xr[c] * scale;
    }
}

int main() {
    printf("ggml-cuda8-rms-norm-smoke: starting\n");

    const int nrows  = 4;
    const int ncols  = 128;
    const float eps  = 1e-5f;
    const size_t nb  = (size_t)nrows * ncols * sizeof(float);

    float *h_x   = (float *)malloc(nb);
    float *h_y   = (float *)malloc(nb);
    float *h_ref = (float *)malloc(nb);

    // deterministic pseudo-random input
    srand(42);
    for (int i = 0; i < nrows * ncols; i++)
        h_x[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;

    rms_norm_ref(h_x, h_ref, nrows, ncols, eps);

    float *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_x, nb));
    CHECK_CUDA(cudaMalloc(&d_y, nb));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, nb, cudaMemcpyHostToDevice));

    ggml_cuda8_rms_norm_f32(d_x, d_y, nrows, ncols, eps, 0);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_y, d_y, nb, cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < nrows * ncols; i++) {
        float e = fabsf(h_y[i] - h_ref[i]);
        if (e > max_err) max_err = e;
    }

    printf("  rows=%d  cols=%d  eps=%.1e  max_err=%.6e\n",
           nrows, ncols, eps, max_err);

    int pass = (max_err < 1e-4f);
    printf("ggml-cuda8-rms-norm-smoke: %s\n", pass ? "PASS" : "FAIL");

    cudaFree(d_x); cudaFree(d_y);
    free(h_x); free(h_y); free(h_ref);
    return pass ? 0 : 1;
}
