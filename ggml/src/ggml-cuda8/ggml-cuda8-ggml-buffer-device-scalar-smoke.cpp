// ggml/src/ggml-cuda8/ggml-cuda8-ggml-buffer-device-scalar-smoke.cpp
//
// G11A-4D smoke: device-resident scalar ADD/MUL dispatcher path.
//
// Validates:
//   - src/dst tensors are CUDA8-resident inside ggml_backend_buffer_t
//   - scalar tensor is also CUDA8-resident
//   - dispatcher executes ADD_SCALAR_F32 and MUL_SCALAR_F32 directly on
//     device src/dst pointers
//   - scalar value is read from CUDA8 buffer via get_tensor()
//   - result is read back only for verification

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

static void setup_f32_scalar(
    ggml_tensor & t,
    void * data
) {
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
                label,
                i,
                got[i],
                ref[i]);
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
            label,
            offset,
            expected_offset);
        return false;
    }

    std::printf("%s residency PASS offset=%zu\n", label, offset);
    return true;
}

static bool run_one(
    ggml_cuda8_context * ctx,
    ggml_backend_buffer_t buffer,
    ggml_tensor * t_src,
    ggml_tensor * t_scalar,
    ggml_tensor * t_dst,
    const std::vector<float> & src,
    float scalar,
    int op,
    const char * label
) {
    const int n = (int) src.size();
    const size_t bytes = (size_t) n * sizeof(float);

    std::vector<float> ref(n);
    std::vector<float> out(n, 0.0f);

    for (int i = 0; i < n; ++i) {
        if (op == GGML_CUDA8_OP_ADD_SCALAR_F32) {
            ref[i] = src[i] + scalar;
        } else {
            ref[i] = src[i] * scalar;
        }
    }

    buffer->iface.set_tensor(buffer, t_src, &src[0], 0, bytes);
    buffer->iface.set_tensor(buffer, t_scalar, &scalar, 0, sizeof(float));
    buffer->iface.memset_tensor(buffer, t_dst, 0, 0, bytes);

    if (!ggml_cuda8_dispatch_supported(ctx, op, t_src, t_scalar, t_dst)) {
        std::fprintf(stderr, "%s unsupported\n", label);
        return false;
    }

    if (ggml_cuda8_dispatch_execute(ctx, op, t_src, t_scalar, t_dst) != 0) {
        std::fprintf(stderr, "%s dispatch failed\n", label);
        return false;
    }

    buffer->iface.get_tensor(buffer, t_dst, &out[0], 0, bytes);

    if (!check_exact(out, ref, label)) {
        return false;
    }

    std::printf("%s PASS\n", label);
    return true;
}

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    std::printf("ggml-cuda8-ggml-buffer-device-scalar-smoke: starting\n");

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

    const size_t off_src    = 0;
    const size_t off_dst    = 2048;
    const size_t off_scalar = 4096;
    const size_t total_size = 8192;

    ggml_backend_buffer_t buffer =
        ggml_cuda8_ggml_buffer_alloc(total_size);

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
    std::printf("tensor src offset:     %zu\n", off_src);
    std::printf("tensor dst offset:     %zu\n", off_dst);
    std::printf("tensor scalar offset:  %zu\n", off_scalar);

    uint8_t * base_u8 = (uint8_t *) base;

    ggml_tensor t_src;
    ggml_tensor t_dst;
    ggml_tensor t_scalar;

    setup_f32_vector(t_src, n, base_u8 + off_src);
    setup_f32_vector(t_dst, n, base_u8 + off_dst);
    setup_f32_scalar(t_scalar, base_u8 + off_scalar);

    if (buffer->iface.init_tensor(buffer, &t_src) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &t_dst) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &t_scalar) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "init_tensor failed\n");
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        return 1;
    }

    if (!expect_resident("src tensor", &t_src, bytes, buffer, off_src) ||
        !expect_resident("dst tensor", &t_dst, bytes, buffer, off_dst) ||
        !expect_resident("scalar tensor", &t_scalar, sizeof(float), buffer, off_scalar)) {
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        return 1;
    }

    std::vector<float> src(n);

    for (int i = 0; i < n; ++i) {
        src[i] = (float) i * 0.125f - 9.0f;
    }

    buffer->iface.clear(buffer, 0);

    bool ok = true;

    ok = ok && run_one(
        ctx,
        buffer,
        &t_src,
        &t_scalar,
        &t_dst,
        src,
        2.0f,
        GGML_CUDA8_OP_ADD_SCALAR_F32,
        "device-resident ADD_SCALAR_F32"
    );

    ok = ok && run_one(
        ctx,
        buffer,
        &t_src,
        &t_scalar,
        &t_dst,
        src,
        -3.0f,
        GGML_CUDA8_OP_MUL_SCALAR_F32,
        "device-resident MUL_SCALAR_F32"
    );

    buffer->iface.free_buffer(buffer);
    ggml_cuda8_context_destroy(ctx);

    if (!ok) {
        std::fprintf(stderr, "ggml-cuda8-ggml-buffer-device-scalar-smoke: FAILED\n");
        return 1;
    }

    std::printf("ggml-cuda8-ggml-buffer-device-scalar-smoke: SUCCESS\n");
    return 0;
}
