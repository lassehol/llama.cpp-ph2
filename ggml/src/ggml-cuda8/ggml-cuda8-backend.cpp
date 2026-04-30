// ggml/src/ggml-cuda8/ggml-cuda8-backend.cpp
//
// G3 CUDA8 backend identity skeleton.

#include "ggml-cuda8-backend.h"

#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

const char * ggml_cuda8_backend_name(void) {
    return "CUDA8";
}

int ggml_cuda8_backend_available(void) {
    int count = 0;

    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return 0;
    }

    return count > 0 ? 1 : 0;
}

int ggml_cuda8_backend_get_device_count(void) {
    int count = 0;

    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        std::fprintf(stderr,
            "ggml-cuda8/backend: cudaGetDeviceCount failed: %s (%d)\n",
            cudaGetErrorString(err), (int) err);
        return -1;
    }

    return count;
}

int ggml_cuda8_backend_get_device_info(
    int index,
    struct ggml_cuda8_device_info * info
) {
    if (info == NULL) {
        std::fprintf(stderr, "ggml-cuda8/backend: null device info pointer\n");
        return -1;
    }

    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);

    if (err != cudaSuccess) {
        std::fprintf(stderr,
            "ggml-cuda8/backend: cudaGetDeviceCount failed: %s (%d)\n",
            cudaGetErrorString(err), (int) err);
        return -1;
    }

    if (index < 0 || index >= count) {
        std::fprintf(stderr,
            "ggml-cuda8/backend: invalid device index %d, count=%d\n",
            index, count);
        return -1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, index);

    if (err != cudaSuccess) {
        std::fprintf(stderr,
            "ggml-cuda8/backend: cudaGetDeviceProperties(%d) failed: %s (%d)\n",
            index, cudaGetErrorString(err), (int) err);
        return -1;
    }

    std::memset(info, 0, sizeof(*info));

    info->index                 = index;
    info->cc_major              = prop.major;
    info->cc_minor              = prop.minor;
    info->warp_size             = prop.warpSize;
    info->multi_processor_count = prop.multiProcessorCount;
    info->clock_rate_khz        = prop.clockRate;
    info->memory_clock_rate_khz = prop.memoryClockRate;
    info->memory_bus_width_bits = prop.memoryBusWidth;
    info->total_global_mem      = (unsigned long long) prop.totalGlobalMem;

    std::snprintf(
        info->name,
        GGML_CUDA8_BACKEND_NAME_MAX,
        "%s",
        prop.name
    );

    return 0;
}

int ggml_cuda8_backend_print_devices(void) {
    const int count = ggml_cuda8_backend_get_device_count();

    if (count < 0) {
        return -1;
    }

    std::printf("ggml-cuda8/backend: backend name: %s\n", ggml_cuda8_backend_name());
    std::printf("ggml-cuda8/backend: detected %d CUDA device(s)\n", count);

    for (int i = 0; i < count; ++i) {
        ggml_cuda8_device_info info;

        if (ggml_cuda8_backend_get_device_info(i, &info) != 0) {
            return -1;
        }

        std::printf(
            "ggml-cuda8/backend: device %d: %s | cc %d.%d | SMs %d | warp %d | global mem %.1f MiB\n",
            info.index,
            info.name,
            info.cc_major,
            info.cc_minor,
            info.multi_processor_count,
            info.warp_size,
            (double) info.total_global_mem / (1024.0 * 1024.0)
        );

        std::printf(
            "ggml-cuda8/backend: device %d clocks: core %.1f MHz | mem %.1f MHz | bus %d-bit\n",
            info.index,
            (double) info.clock_rate_khz / 1000.0,
            (double) info.memory_clock_rate_khz / 1000.0,
            info.memory_bus_width_bits
        );
    }

    return 0;
}
