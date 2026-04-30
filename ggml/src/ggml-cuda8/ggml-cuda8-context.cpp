// ggml/src/ggml-cuda8/ggml-cuda8-context.cpp
//
// G7 minimal CUDA8 backend context skeleton.

#include "ggml-cuda8-context.h"

#include "ggml-cuda8-mulmat.cuh"
#include "q8_0-mmv.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

static int ggml_cuda8_context_check_buffer_device(
    const struct ggml_cuda8_context * ctx,
    const struct ggml_cuda8_backend_buffer * buf,
    const char * name
) {
    if (ctx == NULL || buf == NULL) {
        std::fprintf(stderr, "ggml-cuda8/context: null ctx or buffer for %s\n", name);
        return -1;
    }

    const int buf_device = ggml_cuda8_backend_buffer_get_device(buf);

    if (buf_device != ctx->device) {
        std::fprintf(stderr,
            "ggml-cuda8/context: buffer %s device mismatch: buf=%d ctx=%d\n",
            name, buf_device, ctx->device);
        return -1;
    }

    if (ggml_cuda8_backend_buffer_get_base_const(buf) == NULL) {
        std::fprintf(stderr,
            "ggml-cuda8/context: buffer %s has null base pointer\n",
            name);
        return -1;
    }

    return 0;
}

extern "C" int ggml_cuda8_context_create(
    int device,
    struct ggml_cuda8_context ** out_ctx
) {
    if (out_ctx == NULL) {
        std::fprintf(stderr, "ggml-cuda8/context: null out_ctx\n");
        return -1;
    }

    *out_ctx = NULL;

    const int count = ggml_cuda8_backend_get_device_count();

    if (count <= 0) {
        std::fprintf(stderr, "ggml-cuda8/context: no CUDA8 devices available\n");
        return -1;
    }

    if (device < 0 || device >= count) {
        std::fprintf(stderr,
            "ggml-cuda8/context: invalid device %d, count=%d\n",
            device, count);
        return -1;
    }

    cudaError_t err = cudaSetDevice(device);

    if (err != cudaSuccess) {
        std::fprintf(stderr,
            "ggml-cuda8/context: cudaSetDevice(%d) failed: %s (%d)\n",
            device, cudaGetErrorString(err), (int) err);
        return -1;
    }

    ggml_cuda8_context * ctx =
        (ggml_cuda8_context *) std::malloc(sizeof(ggml_cuda8_context));

    if (ctx == NULL) {
        std::fprintf(stderr, "ggml-cuda8/context: host malloc failed\n");
        return -1;
    }

    std::memset(ctx, 0, sizeof(*ctx));
    ctx->device = device;

    if (ggml_cuda8_backend_get_device_info(device, &ctx->device_info) != 0) {
        std::free(ctx);
        return -1;
    }

    *out_ctx = ctx;
    return 0;
}

extern "C" int ggml_cuda8_context_destroy(
    struct ggml_cuda8_context * ctx
) {
    if (ctx == NULL) {
        return 0;
    }

    std::memset(ctx, 0, sizeof(*ctx));
    std::free(ctx);

    return 0;
}

extern "C" const char * ggml_cuda8_context_backend_name(
    const struct ggml_cuda8_context * ctx
) {
    (void) ctx;
    return ggml_cuda8_backend_name();
}

extern "C" int ggml_cuda8_context_get_device(
    const struct ggml_cuda8_context * ctx
) {
    if (ctx == NULL) {
        return -1;
    }

    return ctx->device;
}

extern "C" const struct ggml_cuda8_device_info * ggml_cuda8_context_get_device_info(
    const struct ggml_cuda8_context * ctx
) {
    if (ctx == NULL) {
        return NULL;
    }

    return &ctx->device_info;
}

extern "C" int ggml_cuda8_context_print(
    const struct ggml_cuda8_context * ctx
) {
    if (ctx == NULL) {
        std::fprintf(stderr, "ggml-cuda8/context: null ctx in print\n");
        return -1;
    }

    const ggml_cuda8_device_info * info = &ctx->device_info;

    std::printf("ggml-cuda8/context: backend=%s device=%d\n",
        ggml_cuda8_context_backend_name(ctx),
        ctx->device
    );

    std::printf(
        "ggml-cuda8/context: device %d: %s | cc %d.%d | SMs %d | warp %d | global mem %.1f MiB\n",
        info->index,
        info->name,
        info->cc_major,
        info->cc_minor,
        info->multi_processor_count,
        info->warp_size,
        (double) info->total_global_mem / (1024.0 * 1024.0)
    );

    return 0;
}

extern "C" int ggml_cuda8_context_alloc_buffer(
    struct ggml_cuda8_context * ctx,
    size_t size,
    struct ggml_cuda8_backend_buffer ** out_buf
) {
    if (ctx == NULL) {
        std::fprintf(stderr, "ggml-cuda8/context: null ctx in alloc_buffer\n");
        return -1;
    }

    return ggml_cuda8_backend_buffer_alloc(ctx->device, size, out_buf);
}

extern "C" int ggml_cuda8_context_free_buffer(
    struct ggml_cuda8_context * ctx,
    struct ggml_cuda8_backend_buffer * buf
) {
    (void) ctx;
    return ggml_cuda8_backend_buffer_free(buf);
}

extern "C" int ggml_cuda8_context_mul_mat_q8_0_f32(
    struct ggml_cuda8_context * ctx,
    const struct ggml_cuda8_backend_buffer * src0_q8_0,
    const struct ggml_cuda8_backend_buffer * src1_f32,
    struct ggml_cuda8_backend_buffer * dst_f32,
    int rows,
    int cols
) {
    if (ctx == NULL) {
        std::fprintf(stderr, "ggml-cuda8/context: null ctx in mul_mat\n");
        return -1;
    }

    if (rows <= 0 || cols <= 0) {
        std::fprintf(stderr,
            "ggml-cuda8/context: invalid mul_mat shape rows=%d cols=%d\n",
            rows, cols);
        return -1;
    }

    if ((cols % GGML_CUDA8_QK8_0) != 0) {
        std::fprintf(stderr,
            "ggml-cuda8/context: cols must be multiple of %d, got %d\n",
            GGML_CUDA8_QK8_0, cols);
        return -1;
    }

    if (ggml_cuda8_context_check_buffer_device(ctx, src0_q8_0, "src0_q8_0") != 0) {
        return -1;
    }

    if (ggml_cuda8_context_check_buffer_device(ctx, src1_f32, "src1_f32") != 0) {
        return -1;
    }

    if (ggml_cuda8_context_check_buffer_device(ctx, dst_f32, "dst_f32") != 0) {
        return -1;
    }

    const int blocks_per_row =
        (cols + GGML_CUDA8_QK8_0 - 1) / GGML_CUDA8_QK8_0;

    const size_t need_src0 =
        (size_t) rows * (size_t) blocks_per_row * sizeof(ggml_cuda8_q8_0_block);

    const size_t need_src1 =
        (size_t) cols * sizeof(float);

    const size_t need_dst =
        (size_t) rows * sizeof(float);

    if (ggml_cuda8_backend_buffer_get_size(src0_q8_0) < need_src0) {
        std::fprintf(stderr,
            "ggml-cuda8/context: src0 too small have=%zu need=%zu\n",
            ggml_cuda8_backend_buffer_get_size(src0_q8_0), need_src0);
        return -1;
    }

    if (ggml_cuda8_backend_buffer_get_size(src1_f32) < need_src1) {
        std::fprintf(stderr,
            "ggml-cuda8/context: src1 too small have=%zu need=%zu\n",
            ggml_cuda8_backend_buffer_get_size(src1_f32), need_src1);
        return -1;
    }

    if (ggml_cuda8_backend_buffer_get_size(dst_f32) < need_dst) {
        std::fprintf(stderr,
            "ggml-cuda8/context: dst too small have=%zu need=%zu\n",
            ggml_cuda8_backend_buffer_get_size(dst_f32), need_dst);
        return -1;
    }

    cudaError_t err = cudaSetDevice(ctx->device);

    if (err != cudaSuccess) {
        std::fprintf(stderr,
            "ggml-cuda8/context: cudaSetDevice(%d) failed in mul_mat: %s (%d)\n",
            ctx->device, cudaGetErrorString(err), (int) err);
        return -1;
    }

    const ggml_cuda8_q8_0_block * d_src0 =
        (const ggml_cuda8_q8_0_block *) ggml_cuda8_backend_buffer_get_base_const(src0_q8_0);

    const float * d_src1 =
        (const float *) ggml_cuda8_backend_buffer_get_base_const(src1_f32);

    float * d_dst =
        (float *) ggml_cuda8_backend_buffer_get_base(dst_f32);

    return ggml_cuda8_mul_mat_q8_0_f32(
        d_src0,
        d_src1,
        d_dst,
        rows,
        cols
    );
}
