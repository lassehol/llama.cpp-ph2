// ggml/src/ggml-cuda8/ggml-cuda8-backend-smoke.cpp
//
// G3 backend identity smoke test.

#include "ggml-cuda8-backend.h"

#include <cstdio>

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    std::printf("ggml-cuda8-backend-smoke: starting\n");

    const char * name = ggml_cuda8_backend_name();

    if (name == NULL) {
        std::fprintf(stderr, "ggml-cuda8-backend-smoke: backend name is NULL\n");
        return 1;
    }

    std::printf("backend name: %s\n", name);

    if (!ggml_cuda8_backend_available()) {
        std::fprintf(stderr, "ggml-cuda8-backend-smoke: backend not available\n");
        return 1;
    }

    const int count = ggml_cuda8_backend_get_device_count();

    if (count <= 0) {
        std::fprintf(stderr,
            "ggml-cuda8-backend-smoke: invalid device count: %d\n",
            count);
        return 1;
    }

    std::printf("device count: %d\n", count);

    if (ggml_cuda8_backend_print_devices() != 0) {
        std::fprintf(stderr, "ggml-cuda8-backend-smoke: failed to print devices\n");
        return 1;
    }

    ggml_cuda8_device_info info;

    if (ggml_cuda8_backend_get_device_info(0, &info) != 0) {
        std::fprintf(stderr, "ggml-cuda8-backend-smoke: failed to query device 0\n");
        return 1;
    }

    if (info.cc_major < 2) {
        std::fprintf(stderr,
            "ggml-cuda8-backend-smoke: unexpected compute capability %d.%d\n",
            info.cc_major,
            info.cc_minor);
        return 1;
    }

    std::printf("ggml-cuda8-backend-smoke: SUCCESS\n");
    return 0;
}
