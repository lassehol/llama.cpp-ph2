// ggml-cuda8-diagmask-smoke.cu  -  G31A: DIAG_MASK_INF standalone smoke
// Tests causal mask: 8x8 matrix, n_past=0
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <float.h>
#include <cuda_runtime.h>

extern "C" int ggml_cuda8_op_diag_mask_inf_f32(
        const float * x, float * dst,
        int ncols, int nrows, int rows_per_channel, int n_past);

#define CHECK_CUDA(x) do { \
    cudaError_t err_ = (x); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d  %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err_)); exit(1); } \
} while (0)

// CPU reference: causal mask
// dst[row][col] = x[row][col] - (col > n_past + row % rows_per_channel) * FLT_MAX
static void diag_mask_ref(const float *x, float *dst,
                           int ncols, int nrows, int rows_per_channel, int n_past) {
    for (int row = 0; row < nrows; row++) {
        for (int col = 0; col < ncols; col++) {
            int i = row * ncols + col;
            if (col > n_past + row % rows_per_channel)
                dst[i] = x[i] - FLT_MAX;
            else
                dst[i] = x[i];
        }
    }
}

int main() {
    printf("ggml-cuda8-diagmask-smoke: starting\n");

    const int ncols = 8;
    const int nrows = 8;
    const int rows_per_channel = 8;
    const int n_past = 0;
    const int n = ncols * nrows;
    const size_t nb = (size_t)n * sizeof(float);

    float *h_x   = (float *)malloc(nb);
    float *h_y   = (float *)malloc(nb);
    float *h_ref = (float *)malloc(nb);

    srand(42);
    for (int i = 0; i < n; i++)
        h_x[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;

    diag_mask_ref(h_x, h_ref, ncols, nrows, rows_per_channel, n_past);

    float *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_x, nb));
    CHECK_CUDA(cudaMalloc(&d_y, nb));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, nb, cudaMemcpyHostToDevice));

    int rc = ggml_cuda8_op_diag_mask_inf_f32(d_x, d_y,
                ncols, nrows, rows_per_channel, n_past);
    if (rc != 0) {
        printf("ggml-cuda8-diagmask-smoke: launch FAILED rc=%d\n", rc);
        return 1;
    }

    CHECK_CUDA(cudaMemcpy(h_y, d_y, nb, cudaMemcpyDeviceToHost));

    // Verify: check that masked positions are -huge, unmasked match exactly
    float max_err = 0.0f;
    int masked_count = 0;
    for (int i = 0; i < n; i++) {
        float e = fabsf(h_y[i] - h_ref[i]);
        if (e > max_err) max_err = e;
        if (h_ref[i] < -1e30f) masked_count++;
    }

    printf("  ncols=%d  nrows=%d  n_past=%d  masked=%d/%d\n",
           ncols, nrows, n_past, masked_count, n);
    printf("  max_err=%.6e\n", max_err);

    // Print mask pattern for visual verification
    printf("  mask pattern (. = keep, X = masked):\n");
    for (int row = 0; row < nrows; row++) {
        printf("    row %d: ", row);
        for (int col = 0; col < ncols; col++) {
            int i = row * ncols + col;
            printf("%c ", h_y[i] < -1e30f ? 'X' : '.');
        }
        printf("\n");
    }

    int pass = (max_err < 1e-6f) && (masked_count > 0);
    printf("ggml-cuda8-diagmask-smoke: %s\n", pass ? "PASS" : "FAIL");

    cudaFree(d_x); cudaFree(d_y);
    free(h_x); free(h_y); free(h_ref);
    return pass ? 0 : 1;
}
