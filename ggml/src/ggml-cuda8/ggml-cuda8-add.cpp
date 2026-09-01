// ggml/src/ggml-cuda8/ggml-cuda8-add.cpp
//
// G9B-2: F32 vector ADD dispatcher helper

#include "ggml-cuda8-add.h"
#include "ggml-cuda8-backend-buffer.h"
#include <cstdio>

extern "C" int ggml_cuda8_add_f32_launch(
    const float * a, const float * b, float * c, int n
);

static bool is_contig_f32_1d(const ggml_tensor * t) {
    return t && t->type == GGML_TYPE_F32 && t->ne[1] == 1 && t->nb[0] == sizeof(float);
}

int ggml_cuda8_supported_add_f32(
    const ggml_cuda8_context * ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    const ggml_tensor * dst
) {
    (void) ctx;
    if (!is_contig_f32_1d(src0)) return 0;
    if (!is_contig_f32_1d(src1)) return 0;
    if (!is_contig_f32_1d(dst))  return 0;
    if (src0->ne[0] != src1->ne[0]) return 0;
    if (src0->ne[0] != dst->ne[0])  return 0;
    return 1;
}

int ggml_cuda8_exec_add_f32(
    ggml_cuda8_context * ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst
) {
    if (!ggml_cuda8_supported_add_f32(ctx, src0, src1, dst)) {
        std::fprintf(stderr, "ggml-cuda8/add: unsupported layout\n");
        return -1;
    }

    const int n = (int) src0->ne[0];
    const size_t bytes = (size_t) n * sizeof(float);

    ggml_cuda8_backend_buffer * b0 = NULL;
    ggml_cuda8_backend_buffer * b1 = NULL;
    ggml_cuda8_backend_buffer * bd = NULL;

    // G-fix: free previously-allocated buffers on a later allocation
    // failure instead of leaking them. This mirrors the cleanup pattern
    // already used by exec_mul_mat_q8_0_f32_vec() and
    // exec_scalar_f32_host_staging() in ggml-cuda8-dispatch.cpp /
    // ggml-cuda8-scalar.cpp. ADD_F32 is one of the most frequently
    // dispatched ops (every residual/bias-add node in the graph-builder
    // pipelines goes through it), so a leak here is not just a style
    // issue -- on a ~870 MiB-free GTX 560 it can exhaust device memory
    // over a long-running session.
    if (ggml_cuda8_context_alloc_buffer(ctx, bytes, &b0) != 0) {
        return -1;
    }
    if (ggml_cuda8_context_alloc_buffer(ctx, bytes, &b1) != 0) {
        ggml_cuda8_backend_buffer_free(b0);
        return -1;
    }
    if (ggml_cuda8_context_alloc_buffer(ctx, bytes, &bd) != 0) {
        ggml_cuda8_backend_buffer_free(b0);
        ggml_cuda8_backend_buffer_free(b1);
        return -1;
    }

    int rc = 0;
    if (ggml_cuda8_backend_buffer_upload(b0, 0, src0->data, bytes) != 0) rc = -1;
    if (rc == 0 && ggml_cuda8_backend_buffer_upload(b1, 0, src1->data, bytes) != 0) rc = -1;

    if (rc == 0) {
        rc = ggml_cuda8_add_f32_launch(
            (const float *) ggml_cuda8_backend_buffer_get_base_const(b0),
            (const float *) ggml_cuda8_backend_buffer_get_base_const(b1),
            (float *)       ggml_cuda8_backend_buffer_get_base(bd),
            n
        );
    }

    if (rc == 0) {
        if (ggml_cuda8_backend_buffer_download(bd, 0, dst->data, bytes) != 0) rc = -1;
    }

    ggml_cuda8_backend_buffer_free(b0);
    ggml_cuda8_backend_buffer_free(b1);
    ggml_cuda8_backend_buffer_free(bd);
    return rc;
}
