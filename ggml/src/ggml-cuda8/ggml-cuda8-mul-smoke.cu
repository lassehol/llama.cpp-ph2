// ggml-cuda8-mul-smoke.cu  -  G26A: element-wise MUL standalone smoke test
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

extern "C" int ggml_cuda8_mul_f32_launch(
        const float * a, const float * b, float * c, int n);

#define CHECK_CUDA(x) do { \
    cudaError_t err_ = (x); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d  %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err_)); exit(1); } \
} while (0)

int main() {
    printf("ggml-cuda8-mul-smoke: starting\n");

    const int n = 512;
    const size_t nb = (size_t)n * sizeof(float);

    float *h_a   = (float *)malloc(nb);
    float *h_b   = (float *)malloc(nb);
    float *h_c   = (float *)malloc(nb);
    float *h_ref = (float *)malloc(nb);

    srand(42);
    for (int i = 0; i < n; i++) {
        h_a[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        h_b[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        h_ref[i] = h_a[i] * h_b[i];
    }

    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, nb));
    CHECK_CUDA(cudaMalloc(&d_b, nb));
    CHECK_CUDA(cudaMalloc(&d_c, nb));
    CHECK_CUDA(cudaMemcpy(d_a, h_a, nb, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, nb, cudaMemcpyHostToDevice));

    int rc = ggml_cuda8_mul_f32_launch(d_a, d_b, d_c, n);
    if (rc != 0) {
        printf("ggml-cuda8-mul-smoke: launch FAILED rc=%d\n", rc);
        return 1;
    }

    CHECK_CUDA(cudaMemcpy(h_c, d_c, nb, cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = fabsf(h_c[i] - h_ref[i]);
        if (e > max_err) max_err = e;
    }

    printf("  n=%d  max_err=%.6e\n", n, max_err);

    int pass = (max_err < 1e-6f);
    printf("ggml-cuda8-mul-smoke: %s\n", pass ? "PASS" : "FAIL");

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c); free(h_ref);
    return pass ? 0 : 1;
}
