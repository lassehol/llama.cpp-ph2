// ggml/src/ggml-cuda8/ggml-cuda8-scalar-smoke.cpp
//
// G9B-3 scalar ADD/MUL smoke test.

#include "ggml-cuda8-dispatch.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>

static void setup_f32_tensor_2d(
    ggml_tensor & t,
    int64_t n0,
    int64_t n1,
    void * data
) {
    std::memset(&t, 0, sizeof(t));

    t.type = GGML_TYPE_F32;

    t.ne[0] = n0;
    t.ne[1] = n1;
    t.ne[2] = 1;
    t.ne[3] = 1;

    t.nb[0] = sizeof(float);
    t.nb[1] = (size_t) n0 * sizeof(float);
    t.nb[2] = t.nb[1] * (size_t) n1;
    t.nb[3] = t.nb[2];

    t.data = data;
}

static void setup_f32_scalar(
    ggml_tensor & t,
    float * scalar
) {
    std::memset(&t, 0, sizeof(t));

    t.type = GGML_TYPE_F32;

    t.ne[0] = 1;
    t.ne[1] = 1;
    t.ne[2] = 1;
    t.ne[3] = 1;

    t.nb[0] = sizeof(float);
    t.nb[1] = sizeof(float);
    t.nb[2] = sizeof(float);
    t.nb[3] = sizeof(float);

    t.data = scalar;
}

static bool run_one_case(
    ggml_cuda8_context * ctx,
    int op,
    int64_t n0,
    int64_t n1,
    float scalar
) {
    const size_t n = (size_t) n0 * (size_t) n1;

    std::vector<float> src(n);
    std::vector<float> dst(n, 0.0f);
    std::vector<float> ref(n, 0.0f);

    for (size_t i = 0; i < n; ++i) {
        src[i] = (float) (i % 97) * 0.03125f - 1.5f;

        if (op == GGML_CUDA8_OP_ADD_SCALAR_F32) {
            ref[i] = src[i] + scalar;
        } else {
            ref[i] = src[i] * scalar;
        }
    }

    ggml_tensor t_src;
    ggml_tensor t_scalar;
    ggml_tensor t_dst;

    setup_f32_tensor_2d(t_src, n0, n1, &src[0]);
    setup_f32_scalar(t_scalar, &scalar);
    setup_f32_tensor_2d(t_dst, n0, n1, &dst[0]);

    if (!ggml_cuda8_dispatch_supported(ctx, op, &t_src, &t_scalar, &t_dst)) {
        std::fprintf(stderr,
            "scalar-smoke: supported() false for op=%s\n",
            ggml_cuda8_op_name(op));
        return false;
    }

    if (ggml_cuda8_dispatch_execute(ctx, op, &t_src, &t_scalar, &t_dst) != 0) {
        std::fprintf(stderr,
            "scalar-smoke: execute failed for op=%s\n",
            ggml_cuda8_op_name(op));
        return false;
    }

    for (size_t i = 0; i < n; ++i) {
        const float diff = dst[i] - ref[i];
        const float abs_diff = diff < 0.0f ? -diff : diff;

        if (abs_diff > 1e-6f) {
            std::fprintf(stderr,
                "scalar-smoke: mismatch op=%s i=%zu got=%f expected=%f\n",
                ggml_cuda8_op_name(op),
                i,
                dst[i],
                ref[i]);
            return false;
        }
    }

    std::printf("%s PASS for [%lld x %lld], scalar=%f\n",
        ggml_cuda8_op_name(op),
        (long long) n0,
        (long long) n1,
        scalar);

    return true;
}

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    std::printf("ggml-cuda8-scalar-smoke: starting\n");

    ggml_cuda8_context * ctx = NULL;

    if (ggml_cuda8_context_create(0, &ctx) != 0 || ctx == NULL) {
        std::fprintf(stderr, "ggml-cuda8-scalar-smoke: failed to create context\n");
        return 1;
    }

    ggml_cuda8_context_print(ctx);

    bool ok = true;

    ok = ok && run_one_case(ctx, GGML_CUDA8_OP_ADD_SCALAR_F32, 64, 1,  2.0f);
    ok = ok && run_one_case(ctx, GGML_CUDA8_OP_ADD_SCALAR_F32, 64, 64, -0.5f);

    ok = ok && run_one_case(ctx, GGML_CUDA8_OP_MUL_SCALAR_F32, 128, 1,  3.0f);
    ok = ok && run_one_case(ctx, GGML_CUDA8_OP_MUL_SCALAR_F32, 128, 64, -2.0f);

    ggml_cuda8_context_destroy(ctx);

    if (!ok) {
        std::fprintf(stderr, "ggml-cuda8-scalar-smoke: FAILED\n");
        return 1;
    }

    std::printf("ggml-cuda8-scalar-smoke: SUCCESS\n");
    return 0;
}
