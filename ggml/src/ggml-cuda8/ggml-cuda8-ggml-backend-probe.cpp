// ggml/src/ggml-cuda8/ggml-cuda8-ggml-backend-probe.cpp
// G12A smoke: minimal ggml_backend_t registration-shaped probe.

#include "ggml-cuda8-ggml-backend.h"
#include "ggml-cuda8-ggml-buffer.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstring>
#include <vector>

static void setup_f32_vector(ggml_tensor & t, int64_t n, void * data) {
    std::memset(&t, 0, sizeof(t));
    t.type = GGML_TYPE_F32;
    t.ne[0] = n; t.ne[1] = 1; t.ne[2] = 1; t.ne[3] = 1;
    t.nb[0] = sizeof(float);
    t.nb[1] = (size_t) n * sizeof(float);
    t.nb[2] = t.nb[1];
    t.nb[3] = t.nb[1];
    t.data = data;
}

int main(int argc, char ** argv) {
    (void) argc; (void) argv;
    std::printf("ggml-cuda8-ggml-backend-probe: starting\n");
    ggml_backend_t backend = ggml_cuda8_ggml_backend_init(0);
    if (backend == NULL) { std::fprintf(stderr, "backend init failed\n"); return 1; }
    if (!ggml_cuda8_ggml_backend_is_cuda8(backend)) { std::fprintf(stderr, "backend CUDA8 identity check failed\n"); backend->iface.free(backend); return 1; }
    const char * backend_name = backend->iface.get_name(backend);
    std::printf("backend name: %s\n", backend_name != NULL ? backend_name : "<null>");

    ggml_backend_buffer_type_t buft = ggml_cuda8_ggml_backend_get_default_buffer_type(backend);
    if (buft == NULL) { std::fprintf(stderr, "backend default buffer type is NULL\n"); backend->iface.free(backend); return 1; }
    std::printf("default buffer type: %s\n", buft->iface.get_name(buft));
    std::printf("alignment:           %zu\n", buft->iface.get_alignment(buft));
    std::printf("is_host:             %d\n", buft->iface.is_host(buft) ? 1 : 0);

    const int n = 64;
    const size_t bytes = (size_t) n * sizeof(float);
    ggml_backend_buffer_t buffer = buft->iface.alloc_buffer(buft, bytes);
    if (buffer == NULL) { std::fprintf(stderr, "default buffer allocation failed\n"); backend->iface.free(backend); return 1; }
    if (!ggml_cuda8_ggml_buffer_is_cuda8(buffer)) { std::fprintf(stderr, "allocated buffer is not CUDA8\n"); buffer->iface.free_buffer(buffer); backend->iface.free(backend); return 1; }
    void * base = buffer->iface.get_base(buffer);
    if (base == NULL) { std::fprintf(stderr, "buffer base is NULL\n"); buffer->iface.free_buffer(buffer); backend->iface.free(backend); return 1; }
    std::printf("allocated buffer size: %zu\n", buffer->size);
    std::printf("buffer base:           %p\n", base);

    ggml_tensor t;
    setup_f32_vector(t, n, base);
    if (buffer->iface.init_tensor(buffer, &t) != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "init_tensor failed\n"); buffer->iface.free_buffer(buffer); backend->iface.free(backend); return 1; }
    std::vector<float> in(n), out(n, 0.0f);
    for (int i = 0; i < n; ++i) in[i] = (float) i * 0.125f - 4.0f;
    buffer->iface.set_tensor(buffer, &t, &in[0], 0, bytes);
    buffer->iface.get_tensor(buffer, &t, &out[0], 0, bytes);
    for (int i = 0; i < n; ++i) {
        if (out[i] != in[i]) { std::fprintf(stderr, "roundtrip mismatch i=%d got=%f expected=%f\n", i, out[i], in[i]); buffer->iface.free_buffer(buffer); backend->iface.free(backend); return 1; }
    }
    std::printf("backend default buffer set/get PASS\n");
    buffer->iface.free_buffer(buffer);
    backend->iface.free(backend);
    std::printf("ggml-cuda8-ggml-backend-probe: SUCCESS\n");
    return 0;
}
