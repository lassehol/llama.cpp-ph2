// ggml-cuda8-getrows-smoke.cu  -  G33A: GET_ROWS standalone smoke test
// Embedding lookup: vocab=32, embd=64, tokens=[3, 7, 0, 15]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

extern "C" int ggml_cuda8_op_get_rows_f32(
        const float * src0, const int * src1, float * dst,
        int ne00, int n_tokens);

#define CHECK_CUDA(x) do { \
    cudaError_t err_ = (x); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d  %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err_)); exit(1); } \
} while (0)

int main() {
    printf("ggml-cuda8-getrows-smoke: starting\n");

    const int vocab     = 32;
    const int embd      = 64;
    const int n_tokens  = 4;
    const int tokens[]  = {3, 7, 0, 15};

    const size_t table_bytes  = (size_t)vocab * embd * sizeof(float);
    const size_t tokens_bytes = (size_t)n_tokens * sizeof(int);
    const size_t out_bytes    = (size_t)n_tokens * embd * sizeof(float);

    // Host data
    float *h_table = (float *)malloc(table_bytes);
    float *h_out   = (float *)malloc(out_bytes);
    float *h_ref   = (float *)malloc(out_bytes);

    // Fill embedding table with deterministic values
    srand(42);
    for (int i = 0; i < vocab * embd; i++)
        h_table[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;

    // CPU reference
    for (int t = 0; t < n_tokens; t++)
        for (int c = 0; c < embd; c++)
            h_ref[t * embd + c] = h_table[tokens[t] * embd + c];

    // Device
    float *d_table, *d_out;
    int   *d_tokens;
    CHECK_CUDA(cudaMalloc(&d_table,  table_bytes));
    CHECK_CUDA(cudaMalloc(&d_tokens, tokens_bytes));
    CHECK_CUDA(cudaMalloc(&d_out,    out_bytes));
    CHECK_CUDA(cudaMemcpy(d_table,  h_table, table_bytes,  cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_tokens, tokens,  tokens_bytes, cudaMemcpyHostToDevice));

    int rc = ggml_cuda8_op_get_rows_f32(d_table, d_tokens, d_out, embd, n_tokens);
    if (rc != 0) {
        printf("ggml-cuda8-getrows-smoke: launch FAILED rc=%d\n", rc);
        return 1;
    }

    CHECK_CUDA(cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < n_tokens * embd; i++) {
        float e = fabsf(h_out[i] - h_ref[i]);
        if (e > max_err) max_err = e;
    }

    printf("  vocab=%d  embd=%d  n_tokens=%d  tokens=[%d,%d,%d,%d]\n",
           vocab, embd, n_tokens, tokens[0], tokens[1], tokens[2], tokens[3]);
    printf("  max_err=%.6e\n", max_err);

    int pass = (max_err == 0.0f);
    printf("ggml-cuda8-getrows-smoke: %s\n", pass ? "PASS" : "FAIL");

    cudaFree(d_table); cudaFree(d_tokens); cudaFree(d_out);
    free(h_table); free(h_out); free(h_ref);
    return pass ? 0 : 1;
}
