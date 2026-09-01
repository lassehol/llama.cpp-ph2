// ggml-cuda8-swiglu-smoke.cu  -  G40: SwiGLU standalone smoke test
//
// Covers both shapes GGML_OP_GLU can take, and padded row strides:
//
//   1. split form   (src1 != NULL)  - what build_ffn/LLM_FFN_PAR emits, so the
//                                     form Qwen3 actually uses
//   2. halves form, swapped=0       - gate = first half, up = second half
//   3. halves form, swapped=1       - the reverse
//   4. split form with padded rows  - row stride > nc, exercising the nb[1]
//                                     handling. ggml only guarantees
//                                     ggml_is_contiguous_1 here, so assuming
//                                     full contiguity would be a latent bug.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

extern "C" int ggml_cuda8_op_swiglu_f32(
        const float * src0, const float * src1, float * dst,
        int nc, int nrows,
        int src0_stride, int src1_stride, int dst_stride,
        int swapped);

#define CHECK_CUDA(x) do { \
    cudaError_t err_ = (x); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA %s:%d  %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err_)); exit(1); } \
} while (0)

// Reference: ggml_vec_swiglu_f32 scalar tail, y[i] = silu(x[i]) * g[i]
static float silu_ref(float x) { return x / (1.0f + expf(-x)); }

static void swiglu_ref(const float * src0, const float * src1, float * dst,
                       int nc, int nrows,
                       int src0_stride, int src1_stride, int dst_stride,
                       int swapped) {
    for (int r = 0; r < nrows; r++) {
        const float * gate;
        const float * up;
        if (src1) {
            gate = src0 + (size_t) r * src0_stride;
            up   = src1 + (size_t) r * src1_stride;
        } else {
            const float * row = src0 + (size_t) r * src0_stride;
            gate = row + (swapped ? nc : 0);
            up   = row + (swapped ? 0  : nc);
        }
        for (int c = 0; c < nc; c++) {
            dst[(size_t) r * dst_stride + c] = silu_ref(gate[c]) * up[c];
        }
    }
}

static int run_case(const char * label,
                    int nc, int nrows, int swapped, int use_split,
                    int src0_stride, int src1_stride, int dst_stride) {
    const size_t src0_floats = (size_t) nrows * src0_stride;
    const size_t src1_floats = use_split ? (size_t) nrows * src1_stride : 0;
    const size_t dst_floats  = (size_t) nrows * dst_stride;

    float * h_a   = (float *) malloc(src0_floats * sizeof(float));
    float * h_b   = use_split ? (float *) malloc(src1_floats * sizeof(float)) : NULL;
    float * h_out = (float *) malloc(dst_floats * sizeof(float));
    float * h_ref = (float *) malloc(dst_floats * sizeof(float));

    // Spread values so silu's saturating tails are exercised, not just x~0.
    for (size_t i = 0; i < src0_floats; i++)
        h_a[i] = ((float) rand() / (float) RAND_MAX) * 16.0f - 8.0f;
    if (h_b)
        for (size_t i = 0; i < src1_floats; i++)
            h_b[i] = ((float) rand() / (float) RAND_MAX) * 4.0f - 2.0f;

    memset(h_ref, 0, dst_floats * sizeof(float));
    swiglu_ref(h_a, h_b, h_ref, nc, nrows,
               src0_stride, src1_stride, dst_stride, swapped);

    float *d_a = NULL, *d_b = NULL, *d_out = NULL;
    CHECK_CUDA(cudaMalloc(&d_a, src0_floats * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_a, h_a, src0_floats * sizeof(float), cudaMemcpyHostToDevice));
    if (use_split) {
        CHECK_CUDA(cudaMalloc(&d_b, src1_floats * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_b, h_b, src1_floats * sizeof(float), cudaMemcpyHostToDevice));
    }
    CHECK_CUDA(cudaMalloc(&d_out, dst_floats * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_out, 0, dst_floats * sizeof(float)));

    int rc = ggml_cuda8_op_swiglu_f32(d_a, d_b, d_out, nc, nrows,
                                      src0_stride, src1_stride, dst_stride, swapped);
    int pass = 0;
    if (rc != 0) {
        printf("  %-34s launch FAILED rc=%d\n", label, rc);
    } else {
        CHECK_CUDA(cudaMemcpy(h_out, d_out, dst_floats * sizeof(float), cudaMemcpyDeviceToHost));

        float max_err = 0.0f;
        for (int r = 0; r < nrows; r++) {
            for (int c = 0; c < nc; c++) {
                const size_t i = (size_t) r * dst_stride + c;
                const float e = fabsf(h_out[i] - h_ref[i]);
                if (e > max_err) max_err = e;
            }
        }
        pass = (max_err < 1e-5f);
        printf("  %-34s max_err=%.6e  %s\n", label, max_err, pass ? "PASS" : "FAIL");
    }

    cudaFree(d_a); if (d_b) cudaFree(d_b); cudaFree(d_out);
    free(h_a); if (h_b) free(h_b); free(h_out); free(h_ref);
    return pass;
}

int main() {
    printf("ggml-cuda8-swiglu-smoke: starting\n");
    srand(1234);

    const int nc    = 256;
    const int nrows = 33;   // deliberately not a multiple of the block size

    int ok = 1;

    // 1. split form - gate and up as separate tensors (the Qwen3 case)
    ok &= run_case("split (src1 != NULL)", nc, nrows, 0, 1, nc, nc, nc);

    // 2/3. halves form - gate and up inside one row, both orderings
    ok &= run_case("halves swapped=0", nc, nrows, 0, 0, 2 * nc, 0, nc);
    ok &= run_case("halves swapped=1", nc, nrows, 1, 0, 2 * nc, 0, nc);

    // 4. padded rows - ggml only promises contiguous rows, not a packed tensor
    ok &= run_case("split, padded row strides", nc, nrows, 0, 1,
                   nc + 7, nc + 3, nc + 5);

    printf("ggml-cuda8-swiglu-smoke: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
