// ggml/src/ggml-cuda8/ggml-cuda8-ggml-api-probe.cpp
//
// G11A-0: GGML backend API compile probe.
//
// Purpose:
//   Verify that the CUDA8 subdir can include ggml backend API headers
//   without linking full ggml-base.
//
// This does not create a backend yet.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include "ggml-cuda8-backend.h"
#include "ggml-cuda8-backend-buffer.h"
#include "ggml-cuda8-context.h"

#include <cstdio>

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    std::printf("ggml-cuda8-ggml-api-probe: starting\n");

    std::printf("backend name: %s\n", ggml_cuda8_backend_name());
    std::printf("buffer type:  %s\n", ggml_cuda8_backend_buffer_type_name());
    std::printf("alignment:    %zu\n", ggml_cuda8_backend_buffer_alignment());

    // We intentionally do not instantiate ggml_backend_buffer_t here yet.
    // This target only validates header/API visibility from ggml-cuda8.

    std::printf("ggml-cuda8-ggml-api-probe: SUCCESS\n");
    return 0;
}
