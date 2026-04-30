// ggml/src/ggml-cuda8/ggml-cuda8-ggml-buffer-device-graph-smoke.cpp
//
// G11B smoke: device-resident mini graph.
//
// Graph:
//   C = A + B
//   D = C * scalar
//
// A/B/C/D/scalar are all inside one CUDA8 ggml_backend_buffer_t.
// Only final D is copied back for verification.

#include "ggml-cuda8-ggml-buffer.h"
#include "ggml-cuda8-dispatch.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstring>
#include <stdint.h>
#include <vector>

static void setup_f32_vector(ggml_tensor & t, int64_t n, void * data) {
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

static void setup_f32_scalar(ggml_tensor & t, void * data) {
    std::memset(&t, 0, sizeof(t));

    t.type = GGML_TYPE_F32;

    t.ne[0] = 1;
    t.ne[1] = 1;
    t.ne[2] = 1;
    t.ne[3] = 1;

    t.nb[0] = sizeof(float);
    t.nb[1] = sizeof(float);
    t.nb[2] = sizeof(float);
    t.nb[3] = sizeof(float);

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
                label, i, got[i], ref[i]);
            return false;
        }
    }

    return true;
}

static bool expect_resident(
    const char * label,
    const ggml_tensor * t,
    size_t bytes,
    ggml_backend_buffer_t expected,
    size_t expected_offset
) {
    ggml_backend_buffer_t owner = NULL;
    size_t offset = 0;

    if (!ggml_cuda8_ggml_tensor_is_device_resident(t, bytes, &owner, &offset)) {
        std::fprintf(stderr, "%s: residency not found\n", label);
        return false;
    }

    if (owner != expected) {
        std::fprintf(stderr, "%s: owner mismatch\n", label);
        return false;
    }

    if (offset != expected_offset) {
        std::fprintf(stderr,
            "%s: offset mismatch got=%zu expected=%zu\n",
            label, offset, expected_offset);
        return false;
    }

    std::printf("%s residency PASS offset=%zu\n", label, offset);
    return true;
}

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    std::printf("ggml-cuda8-ggml-buffer-device-graph-smoke: starting\n");

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

    const size_t off_a      = 0;
    const size_t off_b      = 2048;
    const size_t off_c      = 4096;
    const size_t off_d      = 6144;
    const size_t off_scalar = 7680;
    const size_t total_size = 8192;

    if (off_a + bytes > total_size ||
        off_b + bytes > total_size ||
        off_c + bytes > total_size ||
        off_d + bytes > total_size ||
        off_scalar + sizeof(float) > total_size) {
        std::fprintf(stderr, "internal layout invalid\n");
        ggml_cuda8_context_destroy(ctx);
        return 1;
    }

    ggml_backend_buffer_t buffer = ggml_cuda8_ggml_buffer_alloc(total_size);

    if (buffer == NULL) {
        std::fprintf(stderr, "buffer allocation failed\n");
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
    std::printf("tensor D offset:       %zu\n", off_d);
    std::printf("scalar offset:         %zu\n", off_scalar);

    uint8_t * base_u8 = (uint8_t *) base;

    ggml_tensor t_a;
    ggml_tensor t_b;
    ggml_tensor t_c;
    ggml_tensor t_d;
    ggml_tensor t_scalar;

    setup_f32_vector(t_a, n, base_u8 + off_a);
    setup_f32_vector(t_b, n, base_u8 + off_b);
    setup_f32_vector(t_c, n, base_u8 + off_c);
    setup_f32_vector(t_d, n, base_u8 + off_d);
    setup_f32_scalar(t_scalar, base_u8 + off_scalar);

    if (buffer->iface.init_tensor(buffer, &t_a) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &t_b) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &t_c) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &t_d) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &t_scalar) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "init_tensor failed\n");
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        return 1;
    }

    if (!expect_resident("tensor A", &t_a, bytes, buffer, off_a) ||
        !expect_resident("tensor B", &t_b, bytes, buffer, off_b) ||
        !expect_resident("tensor C", &t_c, bytes, buffer, off_c) ||
        !expect_resident("tensor D", &t_d, bytes, buffer, off_d) ||
        !expect_resident("scalar", &t_scalar, sizeof(float), buffer, off_scalar)) {
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        return 1;
    }

    std::vector<float> a_host(n);
    std::vector<float> b_host(n);
    std::vector<float> c_ref(n);
    std::vector<float> d_ref(n);
    std::vector<float> c_out(n, 0.0f);
    std::vector<float> d_out(n, 0.0f);
    std::vector<float> a_out(n, 0.0f);
    std::vector<float> b_out(n, 0.0f);

    const float scalar = -2.0f;

    for (int i = 0; i < n; ++i) {
        a_host[i] = (float) i * 0.25f - 5.0f;
        b_host[i] = (float) i * -0.125f + 3.0f;
        c_ref[i]  = a_host[i] + b_host[i];
        d_ref[i]  = c_ref[i] * scalar;
    }

    buffer->iface.clear(buffer, 0);

    buffer->iface.set_tensor(buffer, &t_a, &a_host[0], 0, bytes);
    buffer->iface.set_tensor(buffer, &t_b, &b_host[0], 0, bytes);
    buffer->iface.set_tensor(buffer, &t_scalar, &scalar, 0, sizeof(float));

    const int add_op = GGML_CUDA8_OP_ADD_F32;

    if (!ggml_cuda8_dispatch_supported(ctx, add_op, &t_a, &t_b, &t_c)) {
        std::fprintf(stderr, "ADD_F32 unsupported\n");
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        return 1;
    }

    if (ggml_cuda8_dispatch_execute(ctx, add_op, &t_a, &t_b, &t_c) != 0) {
        std::fprintf(stderr, "ADD_F32 graph op failed\n");
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        return 1;
    }

    buffer->iface.get_tensor(buffer, &t_c, &c_out[0], 0, bytes);

    if (!check_exact(c_out, c_ref, "graph intermediate C = A+B")) {
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        return 1;
    }

    std::printf("graph op 1: C = A + B PASS\n");

    const int mul_scalar_op = GGML_CUDA8_OP_MUL_SCALAR_F32;

    if (!ggml_cuda8_dispatch_supported(ctx, mul_scalar_op, &t_c, &t_scalar, &t_d)) {
        std::fprintf(stderr, "MUL_SCALAR_F32 unsupported\n");
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        return 1;
    }

    if (ggml_cuda8_dispatch_execute(ctx, mul_scalar_op, &t_c, &t_scalar, &t_d) != 0) {
        std::fprintf(stderr, "MUL_SCALAR_F32 graph op failed\n");
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        return 1;
    }

    buffer->iface.get_tensor(buffer, &t_d, &d_out[0], 0, bytes);

    if (!check_exact(d_out, d_ref, "graph final D = C*scalar")) {
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        return 1;
    }

    std::printf("graph op 2: D = C * scalar PASS\n");

    buffer->iface.get_tensor(buffer, &t_a, &a_out[0], 0, bytes);
    buffer->iface.get_tensor(buffer, &t_b, &b_out[0], 0, bytes);

    if (!check_exact(a_out, a_host, "graph A unchanged")) {
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        return 1;
    }

    if (!check_exact(b_out, b_host, "graph B unchanged")) {
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        return 1;
    }

    std::printf("graph input isolation PASS\n");

    buffer->iface.free_buffer(buffer);
    ggml_cuda8_context_destroy(ctx);

    std::printf("ggml-cuda8-ggml-buffer-device-graph-smoke: SUCCESS\n");
    return 0;
}
