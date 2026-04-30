// ggml/src/ggml-cuda8/ggml-cuda8-cpy.cpp
//
// G9B-1: F32 -> F32 CPY dispatcher helper

#include "ggml-cuda8-cpy.h"
#include "ggml-cuda8-backend-buffer.h"

#include <cstdio>

static bool tensor_is_contiguous_f32(const ggml_tensor * t) {
    if (t->type != GGML_TYPE_F32) return false;
    if (t->nb[0] != sizeof(float)) return false;

    size_t expected = sizeof(float);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (t->ne[i] <= 1) continue;
        if (t->nb[i] != expected) return false;
        expected *= (size_t) t->ne[i];
    }
    return true;
}

static size_t tensor_nbytes(const ggml_tensor * t) {
    size_t n = sizeof(float);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (t->ne[i] > 0) n *= (size_t) t->ne[i];
    }
    return n;
}

int ggml_cuda8_supported_cpy_f32(
    const ggml_cuda8_context * ctx,
    const ggml_tensor * src,
    const ggml_tensor * dst
) {
    (void) ctx;

    if (!src || !dst) return 0;
    if (!src->data || !dst->data) return 0;

    if (src->type != GGML_TYPE_F32) return 0;
    if (dst->type != GGML_TYPE_F32) return 0;

    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (src->ne[i] != dst->ne[i]) return 0;
    }

    if (!tensor_is_contiguous_f32(src)) return 0;
    if (!tensor_is_contiguous_f32(dst)) return 0;

    return 1;
}

int ggml_cuda8_exec_cpy_f32(
    struct ggml_cuda8_context * ctx,
    const ggml_tensor * src,
    ggml_tensor * dst
) {
    if (!ggml_cuda8_supported_cpy_f32(ctx, src, dst)) {
        std::fprintf(stderr, "ggml-cuda8/cpy: unsupported tensor layout\n");
        return -1;
    }

    const size_t bytes = tensor_nbytes(src);

    ggml_cuda8_backend_buffer * buf = NULL;

    if (ggml_cuda8_context_alloc_buffer(ctx, bytes, &buf) != 0) {
        return -1;
    }

    int rc = 0;

    if (ggml_cuda8_backend_buffer_upload(buf, 0, src->data, bytes) != 0) {
        rc = -1;
    }

    if (rc == 0 &&
        ggml_cuda8_backend_buffer_download(buf, 0, dst->data, bytes) != 0) {
        rc = -1;
    }

    ggml_cuda8_backend_buffer_free(buf);
    return rc;
}
