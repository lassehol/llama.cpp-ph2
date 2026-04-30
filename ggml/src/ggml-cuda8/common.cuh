// ggml/src/ggml-cuda8/common.cuh
#pragma once

#include <cstdio>
#include <cuda_runtime.h>

// Simple CUDA8-safe error helper for the legacy backend
#define GGML_CUDA8_CHECK(call)                                                      \
    do {                                                                            \
        cudaError_t err__ = (call);                                                 \
        if (err__ != cudaSuccess) {                                                 \
            std::fprintf(stderr,                                                    \
                "ggml-cuda8: CUDA error %s (%d) at %s:%d\n",                        \
                cudaGetErrorString(err__), (int) err__, __FILE__, __LINE__);        \
            return -1;                                                              \
        }                                                                           \
    } while (0)
