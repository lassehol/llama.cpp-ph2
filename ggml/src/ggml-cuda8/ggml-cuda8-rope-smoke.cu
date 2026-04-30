// ggml-cuda8-rope-smoke.cu  -  G28A: ROPE standalone smoke test
// Tests basic ROPE (mode=0) with head_dim=64, n_heads=4, seq_len=8
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

extern "C" int ggml_cuda8_op_rope_f32(
        const float * x, float * dst, const int * pos,
        int ne0, int ne1, int ne2, int ne3,
        int n_dims, float freq_base, float freq_scale);

#define CHECK_CUDA(x) do { \
    cudaError_t err_ = (x); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d  %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err_)); exit(1); } \
} while (0)

// CPU reference: basic ROPE (mode=0, no YaRN)
static void rope_ref(const float *x, float *dst, const int *pos,
                      int ne0, int ne1, int ne2, int ne3,
                      int n_dims, float freq_base, float freq_scale) {
    float theta_scale = powf(freq_base, -2.0f / (float)n_dims);
    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            for (int i1 = 0; i1 < ne1; i1++) {
                for (int pair = 0; pair < ne0/2; pair++) {
                    int i0 = pair * 2;
                    int idx = i0 + i1*ne0 + i2*ne0*ne1 + i3*ne0*ne1*ne2;
                    if (i0 >= n_dims) {
                        dst[idx]   = x[idx];
                        dst[idx+1] = x[idx+1];
                    } else {
                        float theta = (float)pos[i2] * powf(theta_scale, (float)pair) * freq_scale;
                        float c = cosf(theta);
                        float s = sinf(theta);
                        dst[idx]   = x[idx]*c - x[idx+1]*s;
                        dst[idx+1] = x[idx]*s + x[idx+1]*c;
                    }
                }
            }
        }
    }
}

int main() {
    printf("ggml-cuda8-rope-smoke: starting\n");

    const int ne0 = 64;    // head_dim
    const int ne1 = 4;     // n_heads
    const int ne2 = 8;     // seq_len
    const int ne3 = 1;     // batch
    const int n_dims = 64; // full rotary
    const float freq_base  = 10000.0f;
    const float freq_scale = 1.0f;

    const int n_floats = ne0 * ne1 * ne2 * ne3;
    const size_t data_bytes = (size_t)n_floats * sizeof(float);
    const size_t pos_bytes  = (size_t)ne2 * sizeof(int);

    float *h_x   = (float *)malloc(data_bytes);
    float *h_y   = (float *)malloc(data_bytes);
    float *h_ref = (float *)malloc(data_bytes);
    int   *h_pos = (int *)malloc(pos_bytes);

    srand(42);
    for (int i = 0; i < n_floats; i++)
        h_x[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < ne2; i++)
        h_pos[i] = i;  // positions 0,1,2,...,7

    rope_ref(h_x, h_ref, h_pos, ne0, ne1, ne2, ne3,
             n_dims, freq_base, freq_scale);

    float *d_x, *d_y;
    int   *d_pos;
    CHECK_CUDA(cudaMalloc(&d_x,   data_bytes));
    CHECK_CUDA(cudaMalloc(&d_y,   data_bytes));
    CHECK_CUDA(cudaMalloc(&d_pos, pos_bytes));
    CHECK_CUDA(cudaMemcpy(d_x,   h_x,   data_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_pos, h_pos, pos_bytes,  cudaMemcpyHostToDevice));

    int rc = ggml_cuda8_op_rope_f32(d_x, d_y, d_pos,
                ne0, ne1, ne2, ne3, n_dims, freq_base, freq_scale);
    if (rc != 0) {
        printf("ggml-cuda8-rope-smoke: launch FAILED rc=%d\n", rc);
        return 1;
    }

    CHECK_CUDA(cudaMemcpy(h_y, d_y, data_bytes, cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < n_floats; i++) {
        float e = fabsf(h_y[i] - h_ref[i]);
        if (e > max_err) max_err = e;
    }

    printf("  head_dim=%d  n_heads=%d  seq_len=%d  n_dims=%d\n",
           ne0, ne1, ne2, n_dims);
    printf("  freq_base=%.1f  freq_scale=%.1f\n", freq_base, freq_scale);
    printf("  max_err=%.6e\n", max_err);

    int pass = (max_err < 1e-4f);
    printf("ggml-cuda8-rope-smoke: %s\n", pass ? "PASS" : "FAIL");

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_pos);
    free(h_x); free(h_y); free(h_ref); free(h_pos);
    return pass ? 0 : 1;
}
