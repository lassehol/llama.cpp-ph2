// ggml/src/ggml-cuda8/ggml-cuda8-cpy-smoke.cpp
//
// G9B-1 CPY smoke test (dispatcher path).
//
// Creates manual ggml_tensor metadata for contiguous F32,
// dispatches GGML_CUDA8_OP_CPY_F32, and verifies exact equality.

#include "ggml-cuda8-dispatch.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void setup_tensor_f32_contig(
    ggml_tensor & t,
    int64_t n0,
    int64_t n1,
    int64_t n2,
    int64_t n3,
    void * data
) {
    std::memset(&t, 0, sizeof(t));
    t.type = GGML_TYPE_F32;

    t.ne[0] = n0; t.ne[1] = n1; t.ne[2] = n2; t.ne[3] = n3;

    t.nb[0] = sizeof(float);
    t.nb[1] = (size_t) n0 * sizeof(float);
    t.nb[2] = t.nb[1] * (size_t) n1;
    t.nb[3] = t.nb[2] * (size_t) n2;

    t.data = data;
}

static size_t nbytes_f32(int64_t n0, int64_t n1, int64_t n2, int64_t n3) {
    return (size_t) n0 * (size_t) n1 * (size_t) n2 * (size_t) n3 * sizeof(float);
}

static bool run_one_case(ggml_cuda8_context * ctx, int64_t n0, int64_t n1) {
    // 2D test: [n0, n1]
    const size_t nbytes = nbytes_f32(n0, n1, 1, 1);
    const size_t n = (size_t) n0 * (size_t) n1;

    std::vector<float> src(n);
    std::vector<float> dst(n, 0.0f);

    for (size_t i = 0; i < n; ++i) {
        // deterministic pattern
        src[i] = (float) (i % 257) * 0.125f - 10.0f;
    }

    ggml_tensor t_src;
    ggml_tensor t_dst;

    setup_tensor_f32_contig(t_src, n0, n1, 1, 1, &src[0]);
    setup_tensor_f32_contig(t_dst, n0, n1, 1, 1, &dst[0]);

    const int op = GGML_CUDA8_OP_CPY_F32;

    if (!ggml_cuda8_dispatch_supported(ctx, op, &t_src, &t_src /*unused*/, &t_dst)) {
        std::fprintf(stderr, "cpy-smoke: supported() false for n0=%lld n1=%lld\n",
            (long long) n0, (long long) n1);
        return false;
    }

    if (ggml_cuda8_dispatch_execute(ctx, op, &t_src, &t_src /*unused*/, &t_dst) != 0) {
        std::fprintf(stderr, "cpy-smoke: execute() failed for n0=%lld n1=%lld\n",
            (long long) n0, (long long) n1);
        return false;
    }

    // exact compare
    if (std::memcmp(&src[0], &dst[0], nbytes) != 0) {
        // locate first mismatch
        for (size_t i = 0; i < n; ++i) {
            if (src[i] != dst[i]) {
                std::fprintf(stderr,
                    "cpy-smoke: mismatch at %zu got=%f expected=%f\n",
                    i, dst[i], src[i]);
                break;
            }
        }
        return false;
    }

    std::printf("CPY PASS for [%lld x %lld]\n", (long long) n0, (long long) n1);
    return true;
}

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    std::printf("ggml-cuda8-cpy-smoke: starting\n");

    ggml_cuda8_context * ctx = NULL;
    if (ggml_cuda8_context_create(0, &ctx) != 0 || ctx == NULL) {
        std::fprintf(stderr, "ggml-cuda8-cpy-smoke: failed to create context\n");
        return 1;
    }

    ggml_cuda8_context_print(ctx);

    bool ok = true;

    ok = ok && run_one_case(ctx, 64,  1);
    ok = ok && run_one_case(ctx, 256, 1);
    ok = ok && run_one_case(ctx, 64,  64);
    ok = ok && run_one_case(ctx, 128, 256);

    ggml_cuda8_context_destroy(ctx);

    if (!ok) {
        std::fprintf(stderr, "ggml-cuda8-cpy-smoke: FAILED\n");
        return 1;
    }

    std::printf("ggml-cuda8-cpy-smoke: SUCCESS\n");
    return 0;
}
