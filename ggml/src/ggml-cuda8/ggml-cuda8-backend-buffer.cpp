// ggml/src/ggml-cuda8/ggml-cuda8-backend-buffer.cpp
//
// G5 minimal CUDA8 backend buffer layout implementation.

#include "ggml-cuda8-backend-buffer.h"

#include "buffer.cuh"

#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <cuda_runtime.h>

static int ggml_cuda8_check_range(
    size_t buffer_size,
    size_t offset,
    size_t size
) {
    if (size == 0) {
        std::fprintf(stderr, "ggml-cuda8/backend-buffer: zero-size transfer\n");
        return -1;
    }

    if (offset > buffer_size) {
        std::fprintf(stderr,
            "ggml-cuda8/backend-buffer: offset out of range offset=%zu size=%zu\n",
            offset, buffer_size);
        return -1;
    }

    if (size > buffer_size - offset) {
        std::fprintf(stderr,
            "ggml-cuda8/backend-buffer: transfer out of range offset=%zu transfer=%zu buffer=%zu\n",
            offset, size, buffer_size);
        return -1;
    }

    return 0;
}

const char * ggml_cuda8_backend_buffer_type_name(void) {
    return "CUDA8";
}

size_t ggml_cuda8_backend_buffer_alignment(void) {
    return GGML_CUDA8_BACKEND_BUFFER_ALIGNMENT;
}

int ggml_cuda8_backend_buffer_alloc(
    int device,
    size_t size,
    struct ggml_cuda8_backend_buffer ** out_buf
) {
    if (out_buf == NULL) {
        std::fprintf(stderr, "ggml-cuda8/backend-buffer: null out_buf\n");
        return -1;
    }

    *out_buf = NULL;

    if (size == 0) {
        std::fprintf(stderr, "ggml-cuda8/backend-buffer: zero-size alloc\n");
        return -1;
    }

    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        std::fprintf(stderr,
            "ggml-cuda8/backend-buffer: cudaSetDevice(%d) failed: %s (%d)\n",
            device, cudaGetErrorString(err), (int) err);
        return -1;
    }

    ggml_cuda8_backend_buffer * buf =
        (ggml_cuda8_backend_buffer *) std::malloc(sizeof(ggml_cuda8_backend_buffer));

    if (buf == NULL) {
        std::fprintf(stderr, "ggml-cuda8/backend-buffer: host malloc failed\n");
        return -1;
    }

    buf->device = device;
    buf->size = size;
    buf->device_ptr = NULL;

    if (ggml_cuda8_buffer_malloc(&buf->device_ptr, size) != 0) {
        std::free(buf);
        return -1;
    }

    *out_buf = buf;
    return 0;
}

int ggml_cuda8_backend_buffer_free(
    struct ggml_cuda8_backend_buffer * buf
) {
    if (buf == NULL) {
        return 0;
    }

    int rc = 0;

    cudaError_t err = cudaSetDevice(buf->device);
    if (err != cudaSuccess) {
        std::fprintf(stderr,
            "ggml-cuda8/backend-buffer: cudaSetDevice(%d) failed during free: %s (%d)\n",
            buf->device, cudaGetErrorString(err), (int) err);
        rc = -1;
    }

    if (buf->device_ptr != NULL) {
        if (ggml_cuda8_buffer_free(buf->device_ptr) != 0) {
            rc = -1;
        }
        buf->device_ptr = NULL;
    }

    buf->size = 0;
    std::free(buf);

    return rc;
}

int ggml_cuda8_backend_buffer_clear(
    struct ggml_cuda8_backend_buffer * buf,
    int value
) {
    if (buf == NULL || buf->device_ptr == NULL) {
        std::fprintf(stderr, "ggml-cuda8/backend-buffer: invalid clear buffer\n");
        return -1;
    }

    cudaError_t err = cudaSetDevice(buf->device);
    if (err != cudaSuccess) {
        std::fprintf(stderr,
            "ggml-cuda8/backend-buffer: cudaSetDevice(%d) failed during clear: %s (%d)\n",
            buf->device, cudaGetErrorString(err), (int) err);
        return -1;
    }

    return ggml_cuda8_buffer_memset(buf->device_ptr, value, buf->size);
}

int ggml_cuda8_backend_buffer_upload(
    struct ggml_cuda8_backend_buffer * buf,
    size_t offset,
    const void * src,
    size_t size
) {
    if (buf == NULL || buf->device_ptr == NULL || src == NULL) {
        std::fprintf(stderr, "ggml-cuda8/backend-buffer: invalid upload args\n");
        return -1;
    }

    if (ggml_cuda8_check_range(buf->size, offset, size) != 0) {
        return -1;
    }

    cudaError_t err = cudaSetDevice(buf->device);
    if (err != cudaSuccess) {
        std::fprintf(stderr,
            "ggml-cuda8/backend-buffer: cudaSetDevice(%d) failed during upload: %s (%d)\n",
            buf->device, cudaGetErrorString(err), (int) err);
        return -1;
    }

    uint8_t * dst =
        (uint8_t *) buf->device_ptr + offset;

    return ggml_cuda8_buffer_upload(dst, src, size);
}

int ggml_cuda8_backend_buffer_download(
    const struct ggml_cuda8_backend_buffer * buf,
    size_t offset,
    void * dst,
    size_t size
) {
    if (buf == NULL || buf->device_ptr == NULL || dst == NULL) {
        std::fprintf(stderr, "ggml-cuda8/backend-buffer: invalid download args\n");
        return -1;
    }

    if (ggml_cuda8_check_range(buf->size, offset, size) != 0) {
        return -1;
    }

    cudaError_t err = cudaSetDevice(buf->device);
    if (err != cudaSuccess) {
        std::fprintf(stderr,
            "ggml-cuda8/backend-buffer: cudaSetDevice(%d) failed during download: %s (%d)\n",
            buf->device, cudaGetErrorString(err), (int) err);
        return -1;
    }

    const uint8_t * src =
        (const uint8_t *) buf->device_ptr + offset;

    return ggml_cuda8_buffer_download(dst, src, size);
}

void * ggml_cuda8_backend_buffer_get_base(
    struct ggml_cuda8_backend_buffer * buf
) {
    if (buf == NULL) {
        return NULL;
    }

    return buf->device_ptr;
}

const void * ggml_cuda8_backend_buffer_get_base_const(
    const struct ggml_cuda8_backend_buffer * buf
) {
    if (buf == NULL) {
        return NULL;
    }

    return buf->device_ptr;
}

size_t ggml_cuda8_backend_buffer_get_size(
    const struct ggml_cuda8_backend_buffer * buf
) {
    if (buf == NULL) {
        return 0;
    }

    return buf->size;
}

int ggml_cuda8_backend_buffer_get_device(
    const struct ggml_cuda8_backend_buffer * buf
) {
    if (buf == NULL) {
        return -1;
    }

    return buf->device;
}
