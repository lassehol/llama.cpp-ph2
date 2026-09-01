// ggml-cuda8-oversized-smoke.cu
// G38: deliberately exceed the Fermi 65535-block grid limit.
//
// Every other smoke in this directory uses small shapes, which is why the
// unclamped launches survived from G9 to G38 unnoticed. This one is the
// opposite: each case is sized so that a naive
//     grid = (n + block - 1) / block
// would ask for more than 65535 blocks and the launch would be rejected with
// cudaErrorInvalidConfiguration.
//
// Two distinct failure modes are covered:
//
//   1. no clamp        -> launch rejected, launcher returns non-zero
//   2. clamp, no loop  -> launch succeeds but silently computes only the
//                         first 65535 blocks' worth of output. This is why
//                         the checks below deliberately look PAST that
//                         boundary rather than just at element 0.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// Kernel entry points under test (all extern "C" in the kernels library).
extern "C" int ggml_cuda8_add_f32_launch(const float * a, const float * b, float * c, int n);
extern "C" int ggml_cuda8_mul_scalar_f32_launch(const float * src, float scalar, float * dst, int n);
extern "C" int ggml_cuda8_softmax_rows_f32_launch(const float * src, float * dst, int rows, int cols);
extern "C" int ggml_cuda8_reduce_sum_rows_f32_launch(const float * src, float * dst, int rows, int cols);
extern "C" int ggml_cuda8_op_rms_norm_f32(const float * x, float * y, int nrows, int ncols, float eps);
extern "C" int ggml_cuda8_op_get_rows_f32(const float * src0, const int * src1, float * dst,
                                          int ne00, int n_tokens);

// 65535 blocks * 256 threads. Element-wise kernels break above this.
static const int GRID_LIMIT_256 = 65535 * 256;   // 16,776,960

// Rows above the grid limit. One-block-per-row kernels break above this.
static const int OVERSIZED_ROWS = 70000;

#define CUDA_OK(expr)                                                          \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "  CUDA error %s at line %d: %s\n",           \
                         #expr, __LINE__, cudaGetErrorString(_e));             \
            return false;                                                      \
        }                                                                      \
    } while (0)

static bool approx(float a, float b, float tol) {
    return std::fabs(a - b) <= tol * (1.0f + std::fabs(b));
}

// ---------------------------------------------------------------------------
// 1. Element-wise: ADD and MUL_SCALAR over more than 65535*256 elements.
// ---------------------------------------------------------------------------
static bool test_elementwise_oversized() {
    const int n = GRID_LIMIT_256 + 4096;   // just past the ceiling
    const size_t bytes = (size_t) n * sizeof(float);

    std::printf("  elementwise: n=%d (%.1f MiB/buffer, needs %d blocks of 256)\n",
                n, (double) bytes / (1024.0 * 1024.0), (n + 255) / 256);

    std::vector<float> h_a(n), h_b(n), h_out(n);
    for (int i = 0; i < n; ++i) {
        h_a[i] = (float) (i % 1000);
        h_b[i] = 2.0f;
    }

    float *d_a = NULL, *d_b = NULL, *d_c = NULL;
    CUDA_OK(cudaMalloc(&d_a, bytes));
    CUDA_OK(cudaMalloc(&d_b, bytes));
    CUDA_OK(cudaMalloc(&d_c, bytes));
    CUDA_OK(cudaMemcpy(d_a, &h_a[0], bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_b, &h_b[0], bytes, cudaMemcpyHostToDevice));

    // Poison the output so "kernel never ran" is distinguishable from "ran".
    CUDA_OK(cudaMemset(d_c, 0xFF, bytes));

    bool ok = true;

    if (ggml_cuda8_add_f32_launch(d_a, d_b, d_c, n) != 0) {
        std::printf("    ADD_F32            FAIL (launcher returned non-zero)\n");
        ok = false;
    } else {
        CUDA_OK(cudaMemcpy(&h_out[0], d_c, bytes, cudaMemcpyDeviceToHost));
        int bad = -1;
        for (int i = 0; i < n; ++i) {
            if (!approx(h_out[i], h_a[i] + 2.0f, 1e-5f)) { bad = i; break; }
        }
        if (bad >= 0) {
            std::printf("    ADD_F32            FAIL at i=%d (%s the %d-block boundary): got %g want %g\n",
                        bad, bad >= GRID_LIMIT_256 ? "PAST" : "before",
                        65535, (double) h_out[bad], (double) (h_a[bad] + 2.0f));
            ok = false;
        } else {
            std::printf("    ADD_F32            PASS (all %d elements, incl. %d past the boundary)\n",
                        n, n - GRID_LIMIT_256);
        }
    }

    CUDA_OK(cudaMemset(d_c, 0xFF, bytes));

    if (ggml_cuda8_mul_scalar_f32_launch(d_a, 3.0f, d_c, n) != 0) {
        std::printf("    MUL_SCALAR_F32     FAIL (launcher returned non-zero)\n");
        ok = false;
    } else {
        CUDA_OK(cudaMemcpy(&h_out[0], d_c, bytes, cudaMemcpyDeviceToHost));
        int bad = -1;
        for (int i = 0; i < n; ++i) {
            if (!approx(h_out[i], h_a[i] * 3.0f, 1e-5f)) { bad = i; break; }
        }
        if (bad >= 0) {
            std::printf("    MUL_SCALAR_F32     FAIL at i=%d (%s the boundary): got %g want %g\n",
                        bad, bad >= GRID_LIMIT_256 ? "PAST" : "before",
                        (double) h_out[bad], (double) (h_a[bad] * 3.0f));
            ok = false;
        } else {
            std::printf("    MUL_SCALAR_F32     PASS\n");
        }
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return ok;
}

// ---------------------------------------------------------------------------
// 2. One-block-per-row kernels with more than 65535 rows.
// ---------------------------------------------------------------------------
static bool test_row_kernels_oversized() {
    const int rows = OVERSIZED_ROWS;
    const int cols = 16;
    const size_t n = (size_t) rows * cols;
    const size_t bytes = n * sizeof(float);

    std::printf("  row kernels: rows=%d (%d over the 65535 limit), cols=%d\n",
                rows, rows - 65535, cols);

    std::vector<float> h_in(n), h_out(n);
    std::vector<float> h_rowout(rows);
    for (size_t i = 0; i < n; ++i) {
        h_in[i] = 2.0f;   // rms_norm: mean(x^2)=4 -> scale=0.5 -> y=1.0
    }

    float *d_in = NULL, *d_out = NULL, *d_rowout = NULL;
    CUDA_OK(cudaMalloc(&d_in, bytes));
    CUDA_OK(cudaMalloc(&d_out, bytes));
    CUDA_OK(cudaMalloc(&d_rowout, (size_t) rows * sizeof(float)));
    CUDA_OK(cudaMemcpy(d_in, &h_in[0], bytes, cudaMemcpyHostToDevice));

    bool ok = true;

    // SUM_ROWS: every row is 16 * 2.0 = 32.0
    CUDA_OK(cudaMemset(d_rowout, 0xFF, (size_t) rows * sizeof(float)));
    if (ggml_cuda8_reduce_sum_rows_f32_launch(d_in, d_rowout, rows, cols) != 0) {
        std::printf("    REDUCE_SUM_ROWS    FAIL (launcher returned non-zero)\n");
        ok = false;
    } else {
        CUDA_OK(cudaMemcpy(&h_rowout[0], d_rowout, (size_t) rows * sizeof(float), cudaMemcpyDeviceToHost));
        int bad = -1;
        for (int r = 0; r < rows; ++r) {
            if (!approx(h_rowout[r], 32.0f, 1e-5f)) { bad = r; break; }
        }
        if (bad >= 0) {
            std::printf("    REDUCE_SUM_ROWS    FAIL at row=%d (%s the boundary): got %g want 32\n",
                        bad, bad >= 65535 ? "PAST" : "before", (double) h_rowout[bad]);
            ok = false;
        } else {
            std::printf("    REDUCE_SUM_ROWS    PASS (all %d rows)\n", rows);
        }
    }

    // SOFTMAX: uniform row -> every element 1/cols
    CUDA_OK(cudaMemset(d_out, 0xFF, bytes));
    if (ggml_cuda8_softmax_rows_f32_launch(d_in, d_out, rows, cols) != 0) {
        std::printf("    SOFTMAX_ROWS       FAIL (launcher returned non-zero)\n");
        ok = false;
    } else {
        CUDA_OK(cudaMemcpy(&h_out[0], d_out, bytes, cudaMemcpyDeviceToHost));
        long bad = -1;
        for (size_t i = 0; i < n; ++i) {
            if (!approx(h_out[i], 1.0f / (float) cols, 1e-5f)) { bad = (long) i; break; }
        }
        if (bad >= 0) {
            std::printf("    SOFTMAX_ROWS       FAIL at row=%ld (%s the boundary): got %g want %g\n",
                        bad / cols, (bad / cols) >= 65535 ? "PAST" : "before",
                        (double) h_out[bad], (double) (1.0f / (float) cols));
            ok = false;
        } else {
            std::printf("    SOFTMAX_ROWS       PASS (all %d rows)\n", rows);
        }
    }

    // RMS_NORM: constant 2.0 row -> all ones
    CUDA_OK(cudaMemset(d_out, 0xFF, bytes));
    if (ggml_cuda8_op_rms_norm_f32(d_in, d_out, rows, cols, 1e-5f) != 0) {
        std::printf("    RMS_NORM           FAIL (launcher returned non-zero)\n");
        ok = false;
    } else {
        CUDA_OK(cudaMemcpy(&h_out[0], d_out, bytes, cudaMemcpyDeviceToHost));
        long bad = -1;
        for (size_t i = 0; i < n; ++i) {
            if (!approx(h_out[i], 1.0f, 1e-4f)) { bad = (long) i; break; }
        }
        if (bad >= 0) {
            std::printf("    RMS_NORM           FAIL at row=%ld (%s the boundary): got %g want 1\n",
                        bad / cols, (bad / cols) >= 65535 ? "PAST" : "before", (double) h_out[bad]);
            ok = false;
        } else {
            std::printf("    RMS_NORM           PASS (all %d rows)\n", rows);
        }
    }

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_rowout);
    return ok;
}

// ---------------------------------------------------------------------------
// 3. GET_ROWS with more than 65535 tokens.
// ---------------------------------------------------------------------------
static bool test_get_rows_oversized() {
    const int n_tokens = OVERSIZED_ROWS;
    const int vocab    = 1000;
    const int ne00     = 16;

    std::printf("  get_rows: n_tokens=%d (%d over the limit), ne00=%d\n",
                n_tokens, n_tokens - 65535, ne00);

    std::vector<float> h_table((size_t) vocab * ne00);
    for (int r = 0; r < vocab; ++r) {
        for (int c = 0; c < ne00; ++c) {
            h_table[(size_t) r * ne00 + c] = (float) (r * 100 + c);
        }
    }
    std::vector<int> h_idx(n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        h_idx[t] = t % vocab;
    }
    std::vector<float> h_out((size_t) n_tokens * ne00);

    float * d_table = NULL;
    int   * d_idx   = NULL;
    float * d_out   = NULL;
    CUDA_OK(cudaMalloc(&d_table, h_table.size() * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_idx,   (size_t) n_tokens * sizeof(int)));
    CUDA_OK(cudaMalloc(&d_out,   h_out.size() * sizeof(float)));
    CUDA_OK(cudaMemcpy(d_table, &h_table[0], h_table.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_idx,   &h_idx[0],   (size_t) n_tokens * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(d_out, 0xFF, h_out.size() * sizeof(float)));

    bool ok = true;

    if (ggml_cuda8_op_get_rows_f32(d_table, d_idx, d_out, ne00, n_tokens) != 0) {
        std::printf("    GET_ROWS_F32       FAIL (launcher returned non-zero)\n");
        ok = false;
    } else {
        CUDA_OK(cudaMemcpy(&h_out[0], d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost));
        long bad = -1;
        for (int t = 0; t < n_tokens && bad < 0; ++t) {
            for (int c = 0; c < ne00; ++c) {
                const float want = (float) ((t % vocab) * 100 + c);
                if (!approx(h_out[(size_t) t * ne00 + c], want, 1e-5f)) { bad = t; break; }
            }
        }
        if (bad >= 0) {
            std::printf("    GET_ROWS_F32       FAIL at token=%ld (%s the boundary)\n",
                        bad, bad >= 65535 ? "PAST" : "before");
            ok = false;
        } else {
            std::printf("    GET_ROWS_F32       PASS (all %d tokens)\n", n_tokens);
        }
    }

    cudaFree(d_table);
    cudaFree(d_idx);
    cudaFree(d_out);
    return ok;
}

int main() {
    std::printf("ggml-cuda8-oversized-smoke: starting\n\n");

    int device = 0;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cannot query device 0\n");
        return 1;
    }
    std::printf("device:            %s (cc %d.%d)\n", prop.name, prop.major, prop.minor);
    std::printf("maxGridSize[0]:    %d\n", prop.maxGridSize[0]);
    std::printf("totalGlobalMem:    %.1f MiB\n\n", (double) prop.totalGlobalMem / (1024.0 * 1024.0));

    // ~200 MiB for the element-wise case; skip rather than fail on a tiny card.
    const size_t needed = 3ull * (GRID_LIMIT_256 + 4096) * sizeof(float);
    size_t mem_free = 0, mem_total = 0;
    cudaMemGetInfo(&mem_free, &mem_total);

    bool ok = true;

    if (mem_free < needed + (32ull << 20)) {
        std::printf("  elementwise: SKIPPED (needs %.1f MiB, %.1f MiB free)\n",
                    (double) needed / (1024.0 * 1024.0),
                    (double) mem_free / (1024.0 * 1024.0));
    } else {
        ok &= test_elementwise_oversized();
    }

    std::printf("\n");
    ok &= test_row_kernels_oversized();
    std::printf("\n");
    ok &= test_get_rows_oversized();

    std::printf("\nggml-cuda8-oversized-smoke: %s\n", ok ? "SUCCESS" : "FAIL");
    return ok ? 0 : 1;
}
