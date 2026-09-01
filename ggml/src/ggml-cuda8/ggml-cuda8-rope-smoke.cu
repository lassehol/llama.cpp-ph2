// ggml-cuda8-rope-smoke.cu  -  G28A: ROPE standalone smoke test
// G45: now covers NORMAL (mode 0) and NEOX (mode 2), head_dim=64, n_heads=4, seq_len=8.
//
// The two modes must not agree: NEOX pairs element p with p + n_dims/2 rather
// than with p+1, so a NEOX result identical to NORMAL would mean the mode was
// ignored. The test asserts they differ as well as that each matches its own
// CPU reference.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

extern "C" int ggml_cuda8_op_rope_f32(
        const float * x, float * dst, const int * pos,
        int ne0, int ne1, int ne2, int ne3,
        int n_dims, int mode, float freq_base, float freq_scale);

#define CHECK_CUDA(x) do { \
    cudaError_t err_ = (x); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d  %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err_)); exit(1); } \
} while (0)

// CPU reference for both pair layouts, no YaRN.
//
// Mirrors rotate_pairs() in ggml/src/ggml-cpu/ops.cpp:
//   mode 0 (NORMAL) pair p -> elements (2p, 2p+1)
//   mode 2 (NEOX)   pair p -> elements (p,  p + n_dims/2)
// Elements at or above n_dims pass through, adjacent-paired in both modes.
static void rope_ref(const float *x, float *dst, const int *pos,
                      int ne0, int ne1, int ne2, int ne3,
                      int n_dims, int mode, float freq_base, float freq_scale) {
    float theta_scale = powf(freq_base, -2.0f / (float)n_dims);
    const int is_neox = (mode == 2);
    const int n_pairs_rot = n_dims / 2;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            for (int i1 = 0; i1 < ne1; i1++) {
                const int row_base = i1*ne0 + i2*ne0*ne1 + i3*ne0*ne1*ne2;

                for (int pair = 0; pair < ne0/2; pair++) {
                    if (pair >= n_pairs_rot) {
                        const int i0 = pair * 2;
                        dst[row_base + i0]     = x[row_base + i0];
                        dst[row_base + i0 + 1] = x[row_base + i0 + 1];
                        continue;
                    }

                    const int ia = is_neox ? pair                : pair * 2;
                    const int ib = is_neox ? pair + n_pairs_rot  : pair * 2 + 1;

                    float theta = (float)pos[i2] * powf(theta_scale, (float)pair) * freq_scale;
                    float c = cosf(theta);
                    float s = sinf(theta);
                    float x0 = x[row_base + ia];
                    float x1 = x[row_base + ib];

                    dst[row_base + ia] = x0*c - x1*s;
                    dst[row_base + ib] = x0*s + x1*c;
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

    float *h_normal = (float *)malloc(data_bytes);   // kept to compare layouts

    float *d_x, *d_y;
    int   *d_pos;
    CHECK_CUDA(cudaMalloc(&d_x,   data_bytes));
    CHECK_CUDA(cudaMalloc(&d_y,   data_bytes));
    CHECK_CUDA(cudaMalloc(&d_pos, pos_bytes));
    CHECK_CUDA(cudaMemcpy(d_x,   h_x,   data_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_pos, h_pos, pos_bytes,  cudaMemcpyHostToDevice));

    printf("  head_dim=%d  n_heads=%d  seq_len=%d  n_dims=%d\n",
           ne0, ne1, ne2, n_dims);
    printf("  freq_base=%.1f  freq_scale=%.1f\n", freq_base, freq_scale);

    int pass = 1;

    const int modes[2]        = { 0, 2 };
    const char * mode_name[2] = { "NORMAL", "NEOX" };

    for (int m = 0; m < 2; m++) {
        const int mode = modes[m];

        rope_ref(h_x, h_ref, h_pos, ne0, ne1, ne2, ne3,
                 n_dims, mode, freq_base, freq_scale);

        CHECK_CUDA(cudaMemset(d_y, 0xFF, data_bytes));

        int rc = ggml_cuda8_op_rope_f32(d_x, d_y, d_pos,
                    ne0, ne1, ne2, ne3, n_dims, mode, freq_base, freq_scale);
        if (rc != 0) {
            printf("  %-6s (mode %d)  launch FAILED rc=%d\n", mode_name[m], mode, rc);
            pass = 0;
            continue;
        }

        CHECK_CUDA(cudaMemcpy(h_y, d_y, data_bytes, cudaMemcpyDeviceToHost));

        float max_err = 0.0f;
        for (int i = 0; i < n_floats; i++) {
            float e = fabsf(h_y[i] - h_ref[i]);
            if (e > max_err) max_err = e;
        }

        const int ok = (max_err < 1e-4f);
        printf("  %-6s (mode %d)  max_err=%.6e  %s\n",
               mode_name[m], mode, max_err, ok ? "PASS" : "FAIL");
        if (!ok) pass = 0;

        if (m == 0) {
            memcpy(h_normal, h_y, data_bytes);
        } else {
            // The layouts must actually differ. If the kernel ignored `mode`,
            // both results would be identical and both would still match a
            // reference that also ignored it - so compare them directly.
            int differs = 0;
            for (int i = 0; i < n_floats; i++) {
                if (fabsf(h_y[i] - h_normal[i]) > 1e-6f) { differs = 1; break; }
            }
            printf("  NEOX differs from NORMAL: %s\n", differs ? "yes (PASS)" : "NO (FAIL)");
            if (!differs) pass = 0;
        }
    }

    // Unsupported modes must be refused, not silently computed as NORMAL.
    {
        int rc = ggml_cuda8_op_rope_f32(d_x, d_y, d_pos,
                    ne0, ne1, ne2, ne3, n_dims, 8 /* MROPE */, freq_base, freq_scale);
        const int ok = (rc != 0);
        printf("  MROPE (mode 8) rejected: %s\n", ok ? "yes (PASS)" : "NO (FAIL)");
        if (!ok) pass = 0;
    }

    printf("ggml-cuda8-rope-smoke: %s\n", pass ? "PASS" : "FAIL");

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_pos);
    free(h_x); free(h_y); free(h_ref); free(h_normal); free(h_pos);
    return pass ? 0 : 1;
}
