// ggml/src/ggml-cuda8/ggml-cuda8-ggml-buffer-dispatch-smoke.cpp
//
// G11A-3 smoke: minimal ggml_backend_buffer_t wrapper + dispatcher interop.
//
// This is intentionally NOT zero-copy dispatcher execution yet.
//
// Current dispatcher ops expect host-backed tensor->data and upload internally.
// Therefore this smoke validates a safe staging path:
//
//   CUDA8 ggml_backend_buffer_t
//       tensor A at base + off_a
//       tensor B at base + off_b
//       tensor C at base + off_c
//
//   host A/B
//       -> buffer->iface.set_tensor(A/B)
//       -> buffer->iface.get_tensor(A/B) into host staging
//       -> ggml_cuda8_dispatch_execute(ADD_F32) using host staging tensors
//       -> buffer->iface.set_tensor(C)
//       -> buffer->iface.get_tensor(C)
//       -> verify C == A + B
//
// This proves:
//   - ggml_backend_buffer_t wrapper can store multiple tensor offsets
//   - set/get tensor interop works
//   - dispatcher result can be round-tripped through the GGML-shaped CUDA8 buffer

#include "ggml-cuda8-ggml-buffer.h"
#include "ggml-cuda8-dispatch.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstring>
#include <stdint.h>
#include <vector>

static void setup_f32_vector(
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

static bool check_exact(
    const std::vector<float> & got,
    const std::vector<float> & ref,
    const char * label
) {
    if (got.size() != ref.size()) {
        std::fprintf(stderr, "%s: size mismatch\n", label);
        return false;
    }

    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] != ref[i]) {
            std::fprintf(stderr,
                "%s: mismatch i=%zu got=%f expected=%f\n",
                label,
                i,
                got[i],
                ref[i]);
            return false;
        }
    }

    return true;
}

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    std::printf("ggml-cuda8-ggml-buffer-dispatch-smoke: starting\n");

    ggml_backend_buffer_type_t buft = ggml_cuda8_ggml_buffer_type();

    if (buft == NULL) {
        std::fprintf(stderr, "buffer type is NULL\n");
        return 1;
    }

    std::printf("buffer type name: %s\n", buft->iface.get_name(buft));
    std::printf("alignment:        %zu\n", buft->iface.get_alignment(buft));
    std::printf("is_host:          %d\n", buft->iface.is_host(buft) ? 1 : 0);

    ggml_cuda8_context * ctx = NULL;

    if (ggml_cuda8_context_create(0, &ctx) != 0 || ctx == NULL) {
        std::fprintf(stderr, "failed to create CUDA8 context\n");
        return 1;
    }

    ggml_cuda8_context_print(ctx);

    const int n = 256;
    const size_t bytes = (size_t) n * sizeof(float);

    // Keep tensor offsets separated and 256-byte aligned.
    const size_t off_a = 0;
    const size_t off_b = 2048;
    const size_t off_c = 4096;
    const size_t total_size = 8192;

    if (off_a + bytes > total_size ||
        off_b + bytes > total_size ||
        off_c + bytes > total_size) {
        std::fprintf(stderr, "internal test layout invalid\n");
        ggml_cuda8_context_destroy(ctx);
        return 1;
    }

    ggml_backend_buffer_t buffer =
        ggml_cuda8_ggml_buffer_alloc(total_size);

    if (buffer == NULL) {
        std::fprintf(stderr, "buffer allocation failed\n");
        ggml_cuda8_context_destroy(ctx);
        return 1;
    }

    if (!ggml_cuda8_ggml_buffer_is_cuda8(buffer)) {
        std::fprintf(stderr, "buffer type check failed\n");
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        return 1;
    }

    void * base = buffer->iface.get_base(buffer);

    if (base == NULL) {
        std::fprintf(stderr, "buffer base is NULL\n");
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        return 1;
    }

    std::printf("allocated buffer size: %zu\n", buffer->size);
    std::printf("buffer base:           %p\n", base);
    std::printf("tensor A offset:       %zu\n", off_a);
    std::printf("tensor B offset:       %zu\n", off_b);
    std::printf("tensor C offset:       %zu\n", off_c);

    uint8_t * base_u8 = (uint8_t *) base;

    ggml_tensor t_a_dev;
    ggml_tensor t_b_dev;
    ggml_tensor t_c_dev;

    setup_f32_vector(t_a_dev, n, base_u8 + off_a);
    setup_f32_vector(t_b_dev, n, base_u8 + off_b);
    setup_f32_vector(t_c_dev, n, base_u8 + off_c);

    if (buffer->iface.init_tensor(buffer, &t_a_dev) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &t_b_dev) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &t_c_dev) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "init_tensor failed\n");
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        return 1;
    }

    std::vector<float> a_host(n);
    std::vector<float> b_host(n);
    std::vector<float> c_ref(n);

    for (int i = 0; i < n; ++i) {
        a_host[i] = (float) i * 0.25f - 5.0f;
        b_host[i] = (float) i * -0.125f + 3.0f;
        c_ref[i]  = a_host[i] + b_host[i];
    }

    buffer->iface.clear(buffer, 0);

    buffer->iface.set_tensor(buffer, &t_a_dev, &a_host[0], 0, bytes);
    buffer->iface.set_tensor(buffer, &t_b_dev, &b_host[0], 0, bytes);

    std::vector<float> a_stage(n, 0.0f);
    std::vector<float> b_stage(n, 0.0f);
    std::vector<float> c_stage(n, 0.0f);
    std::vector<float> c_out(n, 0.0f);

    buffer->iface.get_tensor(buffer, &t_a_dev, &a_stage[0], 0, bytes);
    buffer->iface.get_tensor(buffer, &t_b_dev, &b_stage[0], 0, bytes);

    if (!check_exact(a_stage, a_host, "A staging get")) {
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        return 1;
    }

    if (!check_exact(b_stage, b_host, "B staging get")) {
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        return 1;
    }

    std::printf("A/B buffer -> host staging PASS\n");

    // Existing dispatcher path uses host-backed tensors.
    ggml_tensor t_a_host;
    ggml_tensor t_b_host;
    ggml_tensor t_c_host;

    setup_f32_vector(t_a_host, n, &a_stage[0]);
    setup_f32_vector(t_b_host, n, &b_stage[0]);
    setup_f32_vector(t_c_host, n, &c_stage[0]);

    const int op = GGML_CUDA8_OP_ADD_F32;

    if (!ggml_cuda8_dispatch_supported(ctx, op, &t_a_host, &t_b_host, &t_c_host)) {
        std::fprintf(stderr, "ADD_F32 unsupported for host staging tensors\n");
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        return 1;
    }

    if (ggml_cuda8_dispatch_execute(ctx, op, &t_a_host, &t_b_host, &t_c_host) != 0) {
        std::fprintf(stderr, "ADD_F32 dispatch execute failed\n");
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        return 1;
    }

    if (!check_exact(c_stage, c_ref, "ADD_F32 dispatcher result")) {
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        return 1;
    }

    std::printf("ADD_F32 dispatcher on staged tensors PASS\n");

    // Store dispatcher result into device-backed tensor C.
    buffer->iface.set_tensor(buffer, &t_c_dev, &c_stage[0], 0, bytes);
    buffer->iface.get_tensor(buffer, &t_c_dev, &c_out[0], 0, bytes);

    if (!check_exact(c_out, c_ref, "C device result roundtrip")) {
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        return 1;
    }

    std::printf("dispatcher result -> GGML CUDA8 buffer C PASS\n");

    // Verify A/B still unchanged after writing C.
    std::fill(a_stage.begin(), a_stage.end(), 0.0f);
    std::fill(b_stage.begin(), b_stage.end(), 0.0f);

    buffer->iface.get_tensor(buffer, &t_a_dev, &a_stage[0], 0, bytes);
    buffer->iface.get_tensor(buffer, &t_b_dev, &b_stage[0], 0, bytes);

    if (!check_exact(a_stage, a_host, "A unchanged after C write")) {
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        return 1;
    }

    if (!check_exact(b_stage, b_host, "B unchanged after C write")) {
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        return 1;
    }

    std::printf("A/B isolation after C write PASS\n");

    buffer->iface.free_buffer(buffer);
    ggml_cuda8_context_destroy(ctx);

    std::printf("ggml-cuda8-ggml-buffer-dispatch-smoke: SUCCESS\n");
    return 0;
}
