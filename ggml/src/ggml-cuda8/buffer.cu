// ggml/src/ggml-cuda8/buffer.cu
//
// Minimal legacy CUDA8 / Fermi buffer utilities.
// Purpose:
//   - device allocation / free
//   - host -> device copy
//   - device -> host copy
//   - one tiny self-test helper
//
// This is NOT a full GGML buffer backend yet.
// It is only the first buffer layer for ggml-cuda8.

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

#define GGML_CUDA8_BUFFER_CHECK(call)                                               \
    do {                                                                            \
        cudaError_t err__ = (call);                                                 \
        if (err__ != cudaSuccess) {                                                 \
            std::fprintf(stderr,                                                    \
                "ggml-cuda8/buffer: CUDA error %s (%d) at %s:%d\n",                 \
                cudaGetErrorString(err__), (int) err__, __FILE__, __LINE__);        \
            return -1;                                                              \
        }                                                                           \
    } while (0)

extern "C" int ggml_cuda8_buffer_malloc(void ** dev_ptr, size_t size) {
    if (dev_ptr == NULL || size == 0) {
        std::fprintf(stderr, "ggml-cuda8/buffer: invalid malloc args\n");
        return -1;
    }

    cudaError_t err = cudaMalloc(dev_ptr, size);
    if (err != cudaSuccess) {
        std::fprintf(stderr,
            "ggml-cuda8/buffer: cudaMalloc failed: %s (%d), size=%zu\n",
            cudaGetErrorString(err), (int) err, size);
        *dev_ptr = NULL;
        return -1;
    }

    return 0;
}

extern "C" int ggml_cuda8_buffer_free(void * dev_ptr) {
    if (dev_ptr == NULL) {
        return 0;
    }

    cudaError_t err = cudaFree(dev_ptr);
    if (err != cudaSuccess) {
        std::fprintf(stderr,
            "ggml-cuda8/buffer: cudaFree failed: %s (%d)\n",
            cudaGetErrorString(err), (int) err);
        return -1;
    }

    return 0;
}

extern "C" int ggml_cuda8_buffer_upload(void * dev_ptr, const void * host_ptr, size_t size) {
    if (dev_ptr == NULL || host_ptr == NULL || size == 0) {
        std::fprintf(stderr, "ggml-cuda8/buffer: invalid upload args\n");
        return -1;
    }

    GGML_CUDA8_BUFFER_CHECK(cudaMemcpy(dev_ptr, host_ptr, size, cudaMemcpyHostToDevice));
    return 0;
}

extern "C" int ggml_cuda8_buffer_download(void * host_ptr, const void * dev_ptr, size_t size) {
    if (host_ptr == NULL || dev_ptr == NULL || size == 0) {
        std::fprintf(stderr, "ggml-cuda8/buffer: invalid download args\n");
        return -1;
    }

    GGML_CUDA8_BUFFER_CHECK(cudaMemcpy(host_ptr, dev_ptr, size, cudaMemcpyDeviceToHost));
    return 0;
}

extern "C" int ggml_cuda8_buffer_memset(void * dev_ptr, int value, size_t size) {
    if (dev_ptr == NULL || size == 0) {
        std::fprintf(stderr, "ggml-cuda8/buffer: invalid memset args\n");
        return -1;
    }

    GGML_CUDA8_BUFFER_CHECK(cudaMemset(dev_ptr, value, size));
    return 0;
}

extern "C" int ggml_cuda8_buffer_roundtrip_test(size_t n_floats) {
    if (n_floats == 0) {
        std::fprintf(stderr, "ggml-cuda8/buffer: roundtrip_test n_floats=0\n");
        return -1;
    }

    const size_t bytes = n_floats * sizeof(float);

    float * host_in  = (float *) std::malloc(bytes);
    float * host_out = (float *) std::malloc(bytes);
    void  * dev_ptr  = NULL;

    if (host_in == NULL || host_out == NULL) {
        std::fprintf(stderr, "ggml-cuda8/buffer: host malloc failed\n");
        std::free(host_in);
        std::free(host_out);
        return -1;
    }

    for (size_t i = 0; i < n_floats; ++i) {
        host_in[i] = (float) i * 0.5f;
        host_out[i] = 0.0f;
    }

    if (ggml_cuda8_buffer_malloc(&dev_ptr, bytes) != 0) {
        std::free(host_in);
        std::free(host_out);
        return -1;
    }

    if (ggml_cuda8_buffer_upload(dev_ptr, host_in, bytes) != 0) {
        ggml_cuda8_buffer_free(dev_ptr);
        std::free(host_in);
        std::free(host_out);
        return -1;
    }

    if (ggml_cuda8_buffer_download(host_out, dev_ptr, bytes) != 0) {
        ggml_cuda8_buffer_free(dev_ptr);
        std::free(host_in);
        std::free(host_out);
        return -1;
    }

    int ok = 1;
    for (size_t i = 0; i < n_floats; ++i) {
        if (host_in[i] != host_out[i]) {
            std::fprintf(stderr,
                "ggml-cuda8/buffer: roundtrip mismatch at %zu: in=%f out=%f\n",
                i, host_in[i], host_out[i]);
            ok = 0;
            break;
        }
    }

    ggml_cuda8_buffer_free(dev_ptr);
    std::free(host_in);
    std::free(host_out);

    if (!ok) {
        return -1;
    }

    std::printf("ggml-cuda8/buffer: roundtrip OK for %zu floats\n", n_floats);
    return 0;
}
