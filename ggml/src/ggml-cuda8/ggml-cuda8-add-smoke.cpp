// ggml/src/ggml-cuda8/ggml-cuda8-add-smoke.cpp
//
// G9B-2 ADD smoke test

#include "ggml-cuda8-dispatch.h"
#include "ggml.h"

#include <cstdio>
#include <vector>
#include <cstring>

static void setup_f32_vec(ggml_tensor & t, int n, void * data) {
    std::memset(&t, 0, sizeof(t));
    t.type = GGML_TYPE_F32;
    t.ne[0] = n; t.ne[1] = 1;
    t.nb[0] = sizeof(float);
    t.nb[1] = (size_t) n * sizeof(float);
    t.data = data;
}

int main() {
    printf("ggml-cuda8-add-smoke: starting\n");

    ggml_cuda8_context * ctx = NULL;
    if (ggml_cuda8_context_create(0, &ctx) != 0) return 1;
    ggml_cuda8_context_print(ctx);

    const int n = 1024;
    std::vector<float> a(n), b(n), c(n), ref(n);

    for (int i = 0; i < n; ++i) {
        a[i] = (float) i * 0.25f;
        b[i] = (float) i * -0.125f;
        ref[i] = a[i] + b[i];
    }

    ggml_tensor ta, tb, tc;
    setup_f32_vec(ta, n, &a[0]);
    setup_f32_vec(tb, n, &b[0]);
    setup_f32_vec(tc, n, &c[0]);

    const int op = GGML_CUDA8_OP_ADD_F32;

    if (!ggml_cuda8_dispatch_supported(ctx, op, &ta, &tb, &tc)) {
        fprintf(stderr, "ADD not supported\n");
        return 1;
    }

    if (ggml_cuda8_dispatch_execute(ctx, op, &ta, &tb, &tc) != 0) {
        fprintf(stderr, "ADD execute failed\n");
        return 1;
    }

    for (int i = 0; i < n; ++i) {
        if (c[i] != ref[i]) {
            fprintf(stderr, "ADD mismatch at %d got=%f exp=%f\n",
                i, c[i], ref[i]);
            return 1;
        }
    }

    printf("ggml-cuda8-add-smoke: SUCCESS\n");
    ggml_cuda8_context_destroy(ctx);
    return 0;
}
