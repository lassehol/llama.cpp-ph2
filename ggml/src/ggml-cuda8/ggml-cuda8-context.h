// ggml/src/ggml-cuda8/ggml-cuda8-context.h
//
// G7 minimal CUDA8 backend context skeleton.
//
// This is still NOT real ggml_backend_t registration.
// It is a small object model that owns:
//   - selected CUDA device id
//   - device info snapshot
//   - buffer allocation/free wrappers
//   - Q8_0 MUL_MAT execution using backend buffers
//
// This context is the object we can later wrap with real GGML backend APIs.

#ifndef GGML_CUDA8_CONTEXT_H
#define GGML_CUDA8_CONTEXT_H

#include "ggml-cuda8-backend.h"
#include "ggml-cuda8-backend-buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_cuda8_context {
    int device;
    struct ggml_cuda8_device_info device_info;
};

int ggml_cuda8_context_create(
    int device,
    struct ggml_cuda8_context ** out_ctx
);

int ggml_cuda8_context_destroy(
    struct ggml_cuda8_context * ctx
);

const char * ggml_cuda8_context_backend_name(
    const struct ggml_cuda8_context * ctx
);

int ggml_cuda8_context_get_device(
    const struct ggml_cuda8_context * ctx
);

const struct ggml_cuda8_device_info * ggml_cuda8_context_get_device_info(
    const struct ggml_cuda8_context * ctx
);

int ggml_cuda8_context_print(
    const struct ggml_cuda8_context * ctx
);

int ggml_cuda8_context_alloc_buffer(
    struct ggml_cuda8_context * ctx,
    size_t size,
    struct ggml_cuda8_backend_buffer ** out_buf
);

int ggml_cuda8_context_free_buffer(
    struct ggml_cuda8_context * ctx,
    struct ggml_cuda8_backend_buffer * buf
);

int ggml_cuda8_context_mul_mat_q8_0_f32(
    struct ggml_cuda8_context * ctx,
    const struct ggml_cuda8_backend_buffer * src0_q8_0,
    const struct ggml_cuda8_backend_buffer * src1_f32,
    struct ggml_cuda8_backend_buffer * dst_f32,
    int rows,
    int cols
);

#ifdef __cplusplus
}
#endif

#endif // GGML_CUDA8_CONTEXT_H
