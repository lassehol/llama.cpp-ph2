// ggml/src/ggml-cuda8/ggml-cuda8-ggml-buffer-smoke.cpp
//
// G11A-1 smoke for minimal ggml_backend_buffer_t wrapper.

#include "ggml-cuda8-ggml-buffer.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstring>
#include <vector>

static void setup_f32_tensor(
    ggml_tensor & t,
    int64_t n,
    void * data
) {
    std::memset(&t, 0, sizeof(t));

    t.type = GGML_TYPE_F32;

    t.ne[0] = n;
    t.ne[1] = 1;
    t.ne[2] = 1;
    t.ne[3] = 1;

    t.nb[0] = sizeof(float);
    t.nb[1] = (size_t) n * sizeof(float);
    t.nb[2] = t.nb[1];
    t.nb[3] = t.nb[1];

    t.data = data;
}

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    std::printf("ggml-cuda8-ggml-buffer-smoke: starting\n");

    ggml_backend_buffer_type_t buft = ggml_cuda8_ggml_buffer_type();

    if (buft == NULL) {
        std::fprintf(stderr, "buffer type is NULL\n");
        return 1;
    }

    std::printf("buffer type name: %s\n", buft->iface.get_name(buft));
    std::printf("alignment:        %zu\n", buft->iface.get_alignment(buft));
    std::printf("is_host:          %d\n", buft->iface.is_host(buft) ? 1 : 0);

    const int n = 64;
    const size_t bytes = (size_t) n * sizeof(float);

    ggml_backend_buffer_t buffer = ggml_cuda8_ggml_buffer_alloc(bytes);

    if (buffer == NULL) {
        std::fprintf(stderr, "buffer allocation failed\n");
        return 1;
    }

    if (!ggml_cuda8_ggml_buffer_is_cuda8(buffer)) {
        std::fprintf(stderr, "buffer type check failed\n");
        buffer->iface.free_buffer(buffer);
        return 1;
    }

    std::printf("allocated buffer size: %zu\n", buffer->size);

    void * base = buffer->iface.get_base(buffer);

    if (base == NULL) {
        std::fprintf(stderr, "buffer base is NULL\n");
        buffer->iface.free_buffer(buffer);
        return 1;
    }

    std::printf("buffer base: %p\n", base);

    ggml_tensor t;
    setup_f32_tensor(t, n, base);

    if (buffer->iface.init_tensor(buffer, &t) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "init_tensor failed\n");
        buffer->iface.free_buffer(buffer);
        return 1;
    }

    std::vector<float> h_in(n);
    std::vector<float> h_out(n, 0.0f);

    for (int i = 0; i < n; ++i) {
        h_in[i] = (float) i * 0.25f - 7.0f;
    }

    buffer->iface.set_tensor(buffer, &t, &h_in[0], 0, bytes);
    buffer->iface.get_tensor(buffer, &t, &h_out[0], 0, bytes);

    for (int i = 0; i < n; ++i) {
        if (h_out[i] != h_in[i]) {
            std::fprintf(stderr,
                "roundtrip mismatch i=%d got=%f expected=%f\n",
                i, h_out[i], h_in[i]);
            buffer->iface.free_buffer(buffer);
            return 1;
        }
    }

    std::printf("set/get tensor roundtrip PASS\n");

    buffer->iface.memset_tensor(buffer, &t, 0, 0, bytes);
    buffer->iface.get_tensor(buffer, &t, &h_out[0], 0, bytes);

    for (int i = 0; i < n; ++i) {
        if (h_out[i] != 0.0f) {
            std::fprintf(stderr,
                "memset mismatch i=%d got=%f expected=0\n",
                i, h_out[i]);
            buffer->iface.free_buffer(buffer);
            return 1;
        }
    }

    std::printf("memset tensor PASS\n");

    buffer->iface.clear(buffer, 0);
    buffer->iface.get_tensor(buffer, &t, &h_out[0], 0, bytes);

    for (int i = 0; i < n; ++i) {
        if (h_out[i] != 0.0f) {
            std::fprintf(stderr,
                "clear mismatch i=%d got=%f expected=0\n",
                i, h_out[i]);
            buffer->iface.free_buffer(buffer);
            return 1;
        }
    }

    std::printf("clear buffer PASS\n");

    buffer->iface.free_buffer(buffer);

    std::printf("ggml-cuda8-ggml-buffer-smoke: SUCCESS\n");
    return 0;
}
