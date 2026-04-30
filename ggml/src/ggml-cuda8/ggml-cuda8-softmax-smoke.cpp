// ggml/src/ggml-cuda8/ggml-cuda8-softmax-smoke.cpp
//
// G9B-5B row-wise softmax smoke test.

#include "ggml-cuda8-dispatch.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <float.h>

static void setup_f32_matrix_2d(
    ggml_tensor & t,
    int64_t cols,
    int64_t rows,
    void * data
) {
    std::memset(&t, 0, sizeof(t));

    t.type = GGML_TYPE_F32;

    t.ne[0] = cols;
    t.ne[1] = rows;
    t.ne[2] = 1;
    t.ne[3] = 1;

    t.nb[0] = sizeof(float);
    t.nb[1] = (size_t) cols * sizeof(float);
    t.nb[2] = t.nb[1] * (size_t) rows;
    t.nb[3] = t.nb[2];

    t.data = data;
}

static void cpu_softmax_rows(
    const std::vector<float> & src,
    std::vector<float> & dst,
    int rows,
    int cols
) {
    for (int r = 0; r < rows; ++r) {
        const float * row_src = &src[(size_t) r * cols];
        float * row_dst = &dst[(size_t) r * cols];

        float vmax = -FLT_MAX;

        for (int c = 0; c < cols; ++c) {
            vmax = vmax > row_src[c] ? vmax : row_src[c];
        }

        float sum = 0.0f;

        for (int c = 0; c < cols; ++c) {
            const float e = std::exp(row_src[c] - vmax);
            row_dst[c] = e;
            sum += e;
        }

        const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;

        for (int c = 0; c < cols; ++c) {
            row_dst[c] *= inv_sum;
        }
    }
}

static bool run_one_case(
    ggml_cuda8_context * ctx,
    int rows,
    int cols
) {
    std::printf("\nSOFTMAX_ROWS_F32 test rows=%d cols=%d\n", rows, cols);

    const size_t n = (size_t) rows * (size_t) cols;

    std::vector<float> src(n);
    std::vector<float> dst(n, 0.0f);
    std::vector<float> ref(n, 0.0f);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const int v = (r * 17 + c * 31 + 7) % 101;
            float x = ((float) v - 50.0f) * 0.03125f;

            if ((c % 29) == 0) {
                x += (float) (r % 7) * 0.02f;
            }

            src[(size_t) r * cols + c] = x;
        }
    }

    cpu_softmax_rows(src, ref, rows, cols);

    ggml_tensor t_src;
    ggml_tensor t_dst;
    ggml_tensor t_dummy;

    setup_f32_matrix_2d(t_src, cols, rows, &src[0]);
    setup_f32_matrix_2d(t_dst, cols, rows, &dst[0]);
    setup_f32_matrix_2d(t_dummy, 1, 1, &dst[0]);

    const int op = GGML_CUDA8_OP_SOFTMAX_ROWS_F32;

    if (!ggml_cuda8_dispatch_supported(ctx, op, &t_src, &t_dummy, &t_dst)) {
        std::fprintf(stderr, "softmax-smoke: supported() false\n");
        return false;
    }

    if (ggml_cuda8_dispatch_execute(ctx, op, &t_src, &t_dummy, &t_dst) != 0) {
        std::fprintf(stderr, "softmax-smoke: execute failed\n");
        return false;
    }

    double max_abs_err = 0.0;
    double row_sum_abs_err = 0.0;

    for (int r = 0; r < rows; ++r) {
        double row_sum = 0.0;

        for (int c = 0; c < cols; ++c) {
            const size_t i = (size_t) r * cols + c;

            const double diff = (double) dst[i] - (double) ref[i];
            const double abs_err = diff < 0.0 ? -diff : diff;

            if (abs_err > max_abs_err) max_abs_err = abs_err;
            row_sum += (double) dst[i];
        }

        const double sum_diff = row_sum - 1.0;
        const double sum_abs_err = sum_diff < 0.0 ? -sum_diff : sum_diff;

        if (sum_abs_err > row_sum_abs_err) {
            row_sum_abs_err = sum_abs_err;
        }
    }

    std::printf("max_abs_err vs CPU softmax: %.9g\n", max_abs_err);
    std::printf("max_row_sum_abs_err:       %.9g\n", row_sum_abs_err);

    const double abs_tol = 2e-5;
    const double sum_tol = 2e-5;

    if (max_abs_err > abs_tol || row_sum_abs_err > sum_tol) {
        std::fprintf(stderr,
            "softmax-smoke: FAIL rows=%d cols=%d abs=%g row_sum_abs=%g\n",
            rows, cols, max_abs_err, row_sum_abs_err);
        return false;
    }

    std::printf("PASS\n");
    return true;
}

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    std::printf("ggml-cuda8-softmax-smoke: starting\n");

    ggml_cuda8_context * ctx = NULL;

    if (ggml_cuda8_context_create(0, &ctx) != 0 || ctx == NULL) {
        std::fprintf(stderr, "ggml-cuda8-softmax-smoke: failed to create context\n");
        return 1;
    }

    ggml_cuda8_context_print(ctx);

    bool ok = true;

    ok = ok && run_one_case(ctx, 4,    32);
    ok = ok && run_one_case(ctx, 128,  64);
    ok = ok && run_one_case(ctx, 256, 128);
    ok = ok && run_one_case(ctx, 512, 512);

    ggml_cuda8_context_destroy(ctx);

    if (!ok) {
        std::fprintf(stderr, "ggml-cuda8-softmax-smoke: FAILED\n");
        return 1;
    }

    std::printf("ggml-cuda8-softmax-smoke: SUCCESS\n");
    return 0;
}
