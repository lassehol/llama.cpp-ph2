// ggml/src/ggml-cuda8/ggml-cuda8-ggml-buffer.h
//
// G11A: minimal ggml_backend_buffer_t wrapper for CUDA8.
//
// G11A-4A adds a lightweight residency registry:
//   - register/unregister CUDA8 ggml_backend_buffer_t wrappers
//   - find whether a pointer belongs to a registered CUDA8 buffer
//   - check tensor->data residency with explicit byte size

#ifndef GGML_CUDA8_GGML_BUFFER_H
#define GGML_CUDA8_GGML_BUFFER_H

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

ggml_backend_buffer_type_t ggml_cuda8_ggml_buffer_type(void);
ggml_backend_buffer_t ggml_cuda8_ggml_buffer_alloc(size_t size);
int ggml_cuda8_ggml_buffer_is_cuda8(ggml_backend_buffer_t buffer);

// Registry hooks used by the CUDA8 GGML buffer wrapper.
// These are exposed for smoke tests and future dispatcher/device-residency logic.
int ggml_cuda8_ggml_register_buffer(ggml_backend_buffer_t buffer);
int ggml_cuda8_ggml_unregister_buffer(ggml_backend_buffer_t buffer);

int ggml_cuda8_ggml_buffer_contains_ptr(
    ggml_backend_buffer_t buffer,
    const void * ptr,
    size_t size
);

int ggml_cuda8_ggml_find_buffer_for_ptr(
    const void * ptr,
    size_t size,
    ggml_backend_buffer_t * out_buffer,
    size_t * out_offset
);

int ggml_cuda8_ggml_tensor_is_device_resident(
    const struct ggml_tensor * tensor,
    size_t nbytes,
    ggml_backend_buffer_t * out_buffer,
    size_t * out_offset
);

#ifdef __cplusplus
}
#endif

#endif // GGML_CUDA8_GGML_BUFFER_H
