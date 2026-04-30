// ggml/src/ggml-cuda8/ggml-cuda8-ggml-buffer-offset-smoke.cpp
//
// G11A-2 smoke for tensor offsets inside one minimal ggml_backend_buffer_t.
//
// Validates:
//   - one CUDA8 ggml_backend_buffer_t allocation
//   - two ggml_tensor metadata objects pointing to different device offsets
//   - set/get tensor A
//   - set/get tensor B
//   - memset tensor B only
//   - tensor A remains unchanged
//   - partial set/get using tensor-relative offset

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

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    std::printf("ggml-cuda8-ggml-buffer-offset-smoke: starting\n");

    ggml_backend_buffer_type_t buft = ggml_cuda8_ggml_buffer_type();

    if (buft == NULL) {
        std::fprintf(stderr, "buffer type is NULL\n");
        return 1;
    }

    std::printf("buffer type name: %s\n", buft->iface.get_name(buft));
    std::printf("alignment:        %zu\n", buft->iface.get_alignment(buft));
    std::printf("is_host:          %d\n", buft->iface.is_host(buft) ? 1 : 0);

    const int n_a = 64;
    const int n_b = 64;

    const size_t bytes_a = (size_t) n_a * sizeof(float);
    const size_t bytes_b = (size_t) n_b * sizeof(float);

    // Keep offsets deliberately separated and aligned.
    const size_t off_a = 0;
    const size_t off_b = 512;

    const size_t total_size = 1024;

    if (off_a + bytes_a > total_size || off_b + bytes_b > total_size) {
        std::fprintf(stderr, "internal test layout invalid\n");
        return 1;
    }

    ggml_backend_buffer_t buffer = ggml_cuda8_ggml_buffer_alloc(total_size);

    if (buffer == NULL) {
        std::fprintf(stderr, "buffer allocation failed\n");
        return 1;
    }

    if (!ggml_cuda8_ggml_buffer_is_cuda8(buffer)) {
        std::fprintf(stderr, "buffer type check failed\n");
        buffer->iface.free_buffer(buffer);
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
    std::printf("tensor A offset:       %zu\n", off_a);
    std::printf("tensor B offset:       %zu\n", off_b);

    uint8_t * base_u8 = (uint8_t *) base;

    ggml_tensor t_a;
    ggml_tensor t_b;

    setup_f32_tensor_at(t_a, n_a, base_u8 + off_a);
    setup_f32_tensor_at(t_b, n_b, base_u8 + off_b);

    if (buffer->iface.init_tensor(buffer, &t_a) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "init_tensor A failed\n");
        buffer->iface.free_buffer(buffer);
        return 1;
    }

    if (buffer->iface.init_tensor(buffer, &t_b) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "init_tensor B failed\n");
        buffer->iface.free_buffer(buffer);
        return 1;
    }

    std::vector<float> a_in(n_a);
    std::vector<float> b_in(n_b);
    std::vector<float> a_out(n_a, 0.0f);
    std::vector<float> b_out(n_b, 0.0f);

    for (int i = 0; i < n_a; ++i) {
        a_in[i] = (float) i * 0.25f - 3.0f;
    }

    for (int i = 0; i < n_b; ++i) {
        b_in[i] = 100.0f + (float) i * 0.5f;
    }

    buffer->iface.clear(buffer, 0);

    // Full tensor A/B set/get.
    buffer->iface.set_tensor(buffer, &t_a, &a_in[0], 0, bytes_a);
    buffer->iface.set_tensor(buffer, &t_b, &b_in[0], 0, bytes_b);

    buffer->iface.get_tensor(buffer, &t_a, &a_out[0], 0, bytes_a);
    buffer->iface.get_tensor(buffer, &t_b, &b_out[0], 0, bytes_b);

    if (!check_exact(a_out, a_in, "tensor A full roundtrip")) {
        buffer->iface.free_buffer(buffer);
        return 1;
    }

    if (!check_exact(b_out, b_in, "tensor B full roundtrip")) {
        buffer->iface.free_buffer(buffer);
        return 1;
    }

    std::printf("A/B full offset roundtrip PASS\n");

    // Clear B only.
    buffer->iface.memset_tensor(buffer, &t_b, 0, 0, bytes_b);

    std::fill(a_out.begin(), a_out.end(), -1.0f);
    std::fill(b_out.begin(), b_out.end(), -1.0f);

    buffer->iface.get_tensor(buffer, &t_a, &a_out[0], 0, bytes_a);
    buffer->iface.get_tensor(buffer, &t_b, &b_out[0], 0, bytes_b);

    std::vector<float> b_zero(n_b, 0.0f);

    if (!check_exact(a_out, a_in, "tensor A unchanged after B memset")) {
        buffer->iface.free_buffer(buffer);
        return 1;
    }

    if (!check_exact(b_out, b_zero, "tensor B memset zero")) {
        buffer->iface.free_buffer(buffer);
        return 1;
    }

    std::printf("B-only memset isolation PASS\n");

    // Partial write into B using tensor-relative offset.
    const size_t part_start = 16;
    const size_t part_count = 8;
    const size_t part_offset_bytes = part_start * sizeof(float);
    const size_t part_bytes = part_count * sizeof(float);

    std::vector<float> b_ref = b_zero;
    std::vector<float> b_patch(part_count);

    for (size_t i = 0; i < part_count; ++i) {
        b_patch[i] = -50.0f - (float) i;
        b_ref[part_start + i] = b_patch[i];
    }

    buffer->iface.set_tensor(buffer, &t_b, &b_patch[0], part_offset_bytes, part_bytes);

    std::fill(b_out.begin(), b_out.end(), -1.0f);
    buffer->iface.get_tensor(buffer, &t_b, &b_out[0], 0, bytes_b);

    if (!check_exact(b_out, b_ref, "tensor B partial write")) {
        buffer->iface.free_buffer(buffer);
        return 1;
    }

    // Verify A still unchanged after B partial write.
    std::fill(a_out.begin(), a_out.end(), -1.0f);
    buffer->iface.get_tensor(buffer, &t_a, &a_out[0], 0, bytes_a);

    if (!check_exact(a_out, a_in, "tensor A unchanged after B partial write")) {
        buffer->iface.free_buffer(buffer);
        return 1;
    }

    std::printf("B partial offset write isolation PASS\n");

    buffer->iface.free_buffer(buffer);

    std::printf("ggml-cuda8-ggml-buffer-offset-smoke: SUCCESS\n");
    return 0;
}
