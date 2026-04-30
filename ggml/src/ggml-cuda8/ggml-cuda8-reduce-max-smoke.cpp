// ggml/src/ggml-cuda8/ggml-cuda8-reduce-max-smoke.cpp
//
// G9B-5A row-wise reduce max smoke test.

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

static void setup_f32_vector(
    ggml_tensor & t,
    int64_t n,
    void * data
) {
    std::memset(&t, 0, sizeof(t));

    t.type = GGML_TYPE_F32;

    t.ne[0] = n;
    t.ne[1] = 1;
    t.ne[2] = 1;
    t.ne[3] = 1;

    t.nb[0] = sizeof(float);
    t.nb[1] = (size_t) n * sizeof(float);
    t.nb[2] = t.nb[1];
    t.nb[3] = t.nb[1];

    t.data = data;
}

static bool run_one_case(
    ggml_cuda8_context * ctx,
    int rows,
    int cols
) {
    std::printf("\nREDUCE_MAX_ROWS_F32 test rows=%d cols=%d\n", rows, cols);

    const size_t n_src = (size_t) rows * (size_t) cols;
    const size_t n_dst = (size_t) rows;

    std::vector<float> src(n_src);
    std::vector<float> dst(n_dst, 0.0f);
    std::vector<float> ref(n_dst, 0.0f);

    for (int r = 0; r < rows; ++r) {
        float vmax = -FLT_MAX;

        for (int c = 0; c < cols; ++c) {
            const int v = (r * 17 + c * 31 + 7) % 101;

            // Include both positive and negative values, plus a deterministic row-dependent bump.
            float x = ((float) v - 50.0f) * 0.015625f;
            if ((c % 37) == 0) {
                x += (float) (r % 13) * 0.01f;
            }

            src[(size_t) r * cols + c] = x;
            vmax = vmax > x ? vmax : x;
        }

        ref[r] = vmax;
    }

    ggml_tensor t_src;
    ggml_tensor t_dst;
    ggml_tensor t_dummy;

    setup_f32_matrix_2d(t_src, cols, rows, &src[0]);
    setup_f32_vector(t_dst, rows, &dst[0]);

    // Dispatcher signature requires src1, but this op ignores it.
    setup_f32_vector(t_dummy, 1, &dst[0]);

    const int op = GGML_CUDA8_OP_REDUCE_MAX_ROWS_F32;

    if (!ggml_cuda8_dispatch_supported(ctx, op, &t_src, &t_dummy, &t_dst)) {
        std::fprintf(stderr, "reduce-max-smoke: supported() false\n");
        return false;
    }

    if (ggml_cuda8_dispatch_execute(ctx, op, &t_src, &t_dummy, &t_dst) != 0) {
        std::fprintf(stderr, "reduce-max-smoke: execute failed\n");
        return false;
    }

    double max_abs_err = 0.0;

    for (size_t i = 0; i < n_dst; ++i) {
        const double diff = (double) dst[i] - (double) ref[i];
        const double abs_err = diff < 0.0 ? -diff : diff;

        if (abs_err > max_abs_err) max_abs_err = abs_err;
    }

    std::printf("max_abs_err: %.9g\n", max_abs_err);

    const double abs_tol = 1e-6;

    if (max_abs_err > abs_tol) {
        std::fprintf(stderr,
            "reduce-max-smoke: FAIL rows=%d cols=%d abs=%g\n",
            rows, cols, max_abs_err);
        return false;
    }

    std::printf("PASS\n");
    return true;
}

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    std::printf("ggml-cuda8-reduce-max-smoke: starting\n");

    ggml_cuda8_context * ctx = NULL;

    if (ggml_cuda8_context_create(0, &ctx) != 0 || ctx == NULL) {
        std::fprintf(stderr, "ggml-cuda8-reduce-max-smoke: failed to create context\n");
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
        std::fprintf(stderr, "ggml-cuda8-reduce-max-smoke: FAILED\n");
        return 1;
    }

    std::printf("ggml-cuda8-reduce-max-smoke: SUCCESS\n");
    return 0;
}
