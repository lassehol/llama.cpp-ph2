// ggml/src/ggml-cuda8/smoke-main.cpp
//
// Minimal standalone smoke executable for the legacy CUDA8 / Fermi backend.
// Expected behavior:
//   - calls ggml_cuda8_probe()
//   - prints success/failure
//   - returns 0 on success, nonzero on failure

#include <cstdio>

// Provided by ggml-cuda8.cu
extern "C" int ggml_cuda8_probe(void);

int main() {
    std::printf("ggml-cuda8-smoke: starting probe...\n");

    const int rc = ggml_cuda8_probe();
    if (rc != 0) {
        std::fprintf(stderr, "ggml-cuda8-smoke: probe FAILED (rc=%d)\n", rc);
        return 1;
    }

    std::printf("ggml-cuda8-smoke: SUCCESS\n");
    return 0;
}
