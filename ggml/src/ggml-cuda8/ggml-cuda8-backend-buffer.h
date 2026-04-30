// ggml/src/ggml-cuda8/ggml-cuda8-backend-buffer.h
//
// G5 minimal CUDA8 backend buffer layout.
//
// This is NOT real ggml_backend_buffer_t registration yet.
// It is a small C ABI buffer object that models the layout we will later map
// to GGML backend buffers.
//
// Current scope:
//   - device id
//   - device pointer
//   - byte size
//   - allocation/free
//   - host->device upload
//   - device->host download
//   - memset
//   - simple alignment/type metadata

#ifndef GGML_CUDA8_BACKEND_BUFFER_H
#define GGML_CUDA8_BACKEND_BUFFER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_CUDA8_BACKEND_BUFFER_ALIGNMENT 256

struct ggml_cuda8_backend_buffer {
    int    device;
    size_t size;
    void * device_ptr;
};

const char * ggml_cuda8_backend_buffer_type_name(void);

size_t ggml_cuda8_backend_buffer_alignment(void);

int ggml_cuda8_backend_buffer_alloc(
    int device,
    size_t size,
    struct ggml_cuda8_backend_buffer ** out_buf
);

int ggml_cuda8_backend_buffer_free(
    struct ggml_cuda8_backend_buffer * buf
);

int ggml_cuda8_backend_buffer_clear(
    struct ggml_cuda8_backend_buffer * buf,
    int value
);

int ggml_cuda8_backend_buffer_upload(
    struct ggml_cuda8_backend_buffer * buf,
    size_t offset,
    const void * src,
    size_t size
);

int ggml_cuda8_backend_buffer_download(
    const struct ggml_cuda8_backend_buffer * buf,
    size_t offset,
    void * dst,
    size_t size
);

void * ggml_cuda8_backend_buffer_get_base(
    struct ggml_cuda8_backend_buffer * buf
);

const void * ggml_cuda8_backend_buffer_get_base_const(
    const struct ggml_cuda8_backend_buffer * buf
);

size_t ggml_cuda8_backend_buffer_get_size(
    const struct ggml_cuda8_backend_buffer * buf
);

int ggml_cuda8_backend_buffer_get_device(
    const struct ggml_cuda8_backend_buffer * buf
);

#ifdef __cplusplus
}
#endif

#endif // GGML_CUDA8_BACKEND_BUFFER_H
