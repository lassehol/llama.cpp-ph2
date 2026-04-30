// ggml/src/ggml-cuda8/buffer.cuh
#pragma once

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

int ggml_cuda8_buffer_malloc(void ** dev_ptr, size_t size);
int ggml_cuda8_buffer_free(void * dev_ptr);

int ggml_cuda8_buffer_upload(void * dev_ptr, const void * host_ptr, size_t size);
int ggml_cuda8_buffer_download(void * host_ptr, const void * dev_ptr, size_t size);
int ggml_cuda8_buffer_memset(void * dev_ptr, int value, size_t size);

int ggml_cuda8_buffer_roundtrip_test(size_t n_floats);

#ifdef __cplusplus
}
#endif
