// ggml/src/ggml-cuda8/ggml-cuda8.cu

#include "common.cuh"
#include "buffer.cuh"
#include "mmv.cuh"
#include "q8_0-mmv.cuh"

static __global__ void ggml_cuda8_nop_kernel(void) {
    // intentionally empty
}

extern "C" int ggml_cuda8_probe(void) {
    int count = 0;
    GGML_CUDA8_CHECK(cudaGetDeviceCount(&count));

    std::printf("ggml-cuda8: detected %d CUDA device(s)\n", count);

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop;
        GGML_CUDA8_CHECK(cudaGetDeviceProperties(&prop, i));

        std::printf(
            "ggml-cuda8: device %d: %s | cc %d.%d | global mem %.1f MiB\n",
            i,
            prop.name,
            prop.major,
            prop.minor,
            (double) prop.totalGlobalMem / (1024.0 * 1024.0)
        );
    }

    if (count <= 0) {
        std::fprintf(stderr, "ggml-cuda8: no CUDA devices found\n");
        return -1;
    }

    GGML_CUDA8_CHECK(cudaSetDevice(0));

    ggml_cuda8_nop_kernel<<<1, 1>>>();
    GGML_CUDA8_CHECK(cudaGetLastError());
    GGML_CUDA8_CHECK(cudaDeviceSynchronize());

    if (ggml_cuda8_buffer_roundtrip_test(16) != 0) {
        std::fprintf(stderr, "ggml-cuda8: buffer roundtrip failed\n");
        return -1;
    }

    if (ggml_cuda8_mmv_f32_roundtrip_test() != 0) {
        std::fprintf(stderr, "ggml-cuda8: mmv roundtrip failed\n");
        return -1;
    }

    if (ggml_cuda8_q8_0_mmv_roundtrip_test() != 0) {
        std::fprintf(stderr, "ggml-cuda8: q8_0 mmv roundtrip failed\n");
        return -1;
    }

    std::printf("ggml-cuda8: probe OK\n");
    return 0;
}
