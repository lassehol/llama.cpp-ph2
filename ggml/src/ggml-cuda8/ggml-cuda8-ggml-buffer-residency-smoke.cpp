// ggml/src/ggml-cuda8/ggml-cuda8-ggml-buffer-residency-smoke.cpp
//
// G11A-4A smoke: CUDA8 GGML buffer residency registry.
//
// Validates:
//   - registered CUDA8 ggml_backend_buffer_t can be found by pointer
//   - pointer offsets are reported correctly
//   - host pointers are rejected
//   - out-of-range device pointer ranges are rejected
//   - tensor->data residency helper works
//   - free unregisters buffer

#include "ggml-cuda8-ggml-buffer.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstring>
#include <stdint.h>
#include <vector>

static void setup_f32_tensor_at(
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

static bool expect_found(
    const char * label,
    const void * ptr,
    size_t size,
    ggml_backend_buffer_t expected_buffer,
    size_t expected_offset
) {
    ggml_backend_buffer_t owner = NULL;
    size_t offset = 0;

    const int found =
        ggml_cuda8_ggml_find_buffer_for_ptr(ptr, size, &owner, &offset);

    if (!found) {
        std::fprintf(stderr, "%s: expected found, got not found\n", label);
        return false;
    }

    if (owner != expected_buffer) {
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

    std::printf("%s PASS offset=%zu\n", label, offset);
    return true;
}

static bool expect_not_found(
    const char * label,
    const void * ptr,
    size_t size
) {
    ggml_backend_buffer_t owner = NULL;
    size_t offset = 0;

    const int found =
        ggml_cuda8_ggml_find_buffer_for_ptr(ptr, size, &owner, &offset);

    if (found) {
        std::fprintf(stderr,
            "%s: expected not found, got owner=%p offset=%zu\n",
            label,
            (void *) owner,
            offset);
        return false;
    }

    std::printf("%s PASS\n", label);
    return true;
}

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    std::printf("ggml-cuda8-ggml-buffer-residency-smoke: starting\n");

    ggml_backend_buffer_type_t buft = ggml_cuda8_ggml_buffer_type();

    if (buft == NULL) {
        std::fprintf(stderr, "buffer type is NULL\n");
        return 1;
    }

    std::printf("buffer type name: %s\n", buft->iface.get_name(buft));
    std::printf("alignment:        %zu\n", buft->iface.get_alignment(buft));
    std::printf("is_host:          %d\n", buft->iface.is_host(buft) ? 1 : 0);

    const size_t total_size = 8192;

    ggml_backend_buffer_t buffer =
        ggml_cuda8_ggml_buffer_alloc(total_size);

    if (buffer == NULL) {
        std::fprintf(stderr, "buffer allocation failed\n");
        return 1;
    }

    void * base = buffer->iface.get_base(buffer);

    if (base == NULL) {
        std::fprintf(stderr, "buffer base is NULL\n");
        buffer->iface.free_buffer(buffer);
        return 1;
    }

    std::printf("allocated buffer size: %zu\n", buffer->size);
    std::printf("buffer base:           %p\n", base);

    uint8_t * base_u8 = (uint8_t *) base;

    if (!expect_found("contains base+0 size=1024", base_u8 + 0, 1024, buffer, 0)) {
        buffer->iface.free_buffer(buffer);
        return 1;
    }

    if (!expect_found("contains base+2048 size=1024", base_u8 + 2048, 1024, buffer, 2048)) {
        buffer->iface.free_buffer(buffer);
        return 1;
    }

    if (!expect_found("contains base+4096 size=1024", base_u8 + 4096, 1024, buffer, 4096)) {
        buffer->iface.free_buffer(buffer);
        return 1;
    }

    if (!ggml_cuda8_ggml_buffer_contains_ptr(buffer, base_u8 + 512, 256)) {
        std::fprintf(stderr, "direct contains_ptr failed\n");
        buffer->iface.free_buffer(buffer);
        return 1;
    }

    std::printf("direct contains_ptr PASS\n");

    std::vector<float> host(64, 1.0f);

    if (!expect_not_found("host pointer rejected", &host[0], host.size() * sizeof(float))) {
        buffer->iface.free_buffer(buffer);
        return 1;
    }

    if (!expect_not_found("out-of-range base+8000 size=512", base_u8 + 8000, 512)) {
        buffer->iface.free_buffer(buffer);
        return 1;
    }

    if (!expect_not_found("out-of-range base+8192 size=1", base_u8 + 8192, 1)) {
        buffer->iface.free_buffer(buffer);
        return 1;
    }

    ggml_tensor t_a;
    ggml_tensor t_b;

    setup_f32_tensor_at(t_a, 256, base_u8 + 0);
    setup_f32_tensor_at(t_b, 256, base_u8 + 2048);

    ggml_backend_buffer_t owner = NULL;
    size_t offset = 0;

    if (!ggml_cuda8_ggml_tensor_is_device_resident(&t_a, 256 * sizeof(float), &owner, &offset)) {
        std::fprintf(stderr, "tensor A residency not found\n");
        buffer->iface.free_buffer(buffer);
        return 1;
    }

    if (owner != buffer || offset != 0) {
        std::fprintf(stderr, "tensor A residency mismatch offset=%zu\n", offset);
        buffer->iface.free_buffer(buffer);
        return 1;
    }

    std::printf("tensor A residency PASS offset=%zu\n", offset);

    owner = NULL;
    offset = 0;

    if (!ggml_cuda8_ggml_tensor_is_device_resident(&t_b, 256 * sizeof(float), &owner, &offset)) {
        std::fprintf(stderr, "tensor B residency not found\n");
        buffer->iface.free_buffer(buffer);
        return 1;
    }

    if (owner != buffer || offset != 2048) {
        std::fprintf(stderr, "tensor B residency mismatch offset=%zu\n", offset);
        buffer->iface.free_buffer(buffer);
        return 1;
    }

    std::printf("tensor B residency PASS offset=%zu\n", offset);

    const void * saved_base = base;

    buffer->iface.free_buffer(buffer);

    if (!expect_not_found("pointer rejected after buffer free", saved_base, 1)) {
        return 1;
    }

    std::printf("ggml-cuda8-ggml-buffer-residency-smoke: SUCCESS\n");
    return 0;
}
