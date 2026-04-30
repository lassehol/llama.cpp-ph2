// ggml/src/ggml-cuda8/ggml-cuda8-buffer-smoke.cpp
//
// G5 backend buffer layout smoke test.
//
// Validates:
//   - buffer type name
//   - alignment metadata
//   - device buffer allocation
//   - upload/download
//   - offset upload/download
//   - clear/memset
//   - size/device/base queries

#include "ggml-cuda8-backend-buffer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static bool check_float_equal(
    const std::vector<float> & a,
    const std::vector<float> & b,
    const char * label
) {
    if (a.size() != b.size()) {
        std::fprintf(stderr, "%s: size mismatch\n", label);
        return false;
    }

    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            std::fprintf(stderr,
                "%s: mismatch at %zu got=%f expected=%f\n",
                label, i, b[i], a[i]);
            return false;
        }
    }

    return true;
}

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    std::printf("ggml-cuda8-buffer-smoke: starting\n");

    std::printf("buffer type: %s\n", ggml_cuda8_backend_buffer_type_name());
    std::printf("alignment:   %zu bytes\n", ggml_cuda8_backend_buffer_alignment());

    const int device = 0;
    const size_t n_floats = 64;
    const size_t bytes = n_floats * sizeof(float);

    ggml_cuda8_backend_buffer * buf = NULL;

    if (ggml_cuda8_backend_buffer_alloc(device, bytes, &buf) != 0) {
        std::fprintf(stderr, "ggml-cuda8-buffer-smoke: alloc failed\n");
        return 1;
    }

    if (buf == NULL) {
        std::fprintf(stderr, "ggml-cuda8-buffer-smoke: alloc returned NULL\n");
        return 1;
    }

    std::printf("allocated buffer: device=%d size=%zu base=%p\n",
        ggml_cuda8_backend_buffer_get_device(buf),
        ggml_cuda8_backend_buffer_get_size(buf),
        ggml_cuda8_backend_buffer_get_base(buf)
    );

    if (ggml_cuda8_backend_buffer_get_size(buf) != bytes) {
        std::fprintf(stderr, "ggml-cuda8-buffer-smoke: size query mismatch\n");
        ggml_cuda8_backend_buffer_free(buf);
        return 1;
    }

    if (ggml_cuda8_backend_buffer_get_device(buf) != device) {
        std::fprintf(stderr, "ggml-cuda8-buffer-smoke: device query mismatch\n");
        ggml_cuda8_backend_buffer_free(buf);
        return 1;
    }

    if (ggml_cuda8_backend_buffer_get_base(buf) == NULL) {
        std::fprintf(stderr, "ggml-cuda8-buffer-smoke: base pointer is NULL\n");
        ggml_cuda8_backend_buffer_free(buf);
        return 1;
    }

    std::vector<float> h_in(n_floats);
    std::vector<float> h_out(n_floats, 0.0f);

    for (size_t i = 0; i < n_floats; ++i) {
        h_in[i] = (float) i * 0.25f - 3.0f;
    }

    if (ggml_cuda8_backend_buffer_upload(buf, 0, &h_in[0], bytes) != 0) {
        std::fprintf(stderr, "ggml-cuda8-buffer-smoke: upload failed\n");
        ggml_cuda8_backend_buffer_free(buf);
        return 1;
    }

    if (ggml_cuda8_backend_buffer_download(buf, 0, &h_out[0], bytes) != 0) {
        std::fprintf(stderr, "ggml-cuda8-buffer-smoke: download failed\n");
        ggml_cuda8_backend_buffer_free(buf);
        return 1;
    }

    if (!check_float_equal(h_in, h_out, "full roundtrip")) {
        ggml_cuda8_backend_buffer_free(buf);
        return 1;
    }

    std::printf("full upload/download PASS\n");

    // Offset upload/download test.
    const size_t offset_floats = 16;
    const size_t sub_floats = 8;
    const size_t offset_bytes = offset_floats * sizeof(float);
    const size_t sub_bytes = sub_floats * sizeof(float);

    std::vector<float> h_sub_in(sub_floats);
    std::vector<float> h_sub_out(sub_floats, 0.0f);

    for (size_t i = 0; i < sub_floats; ++i) {
        h_sub_in[i] = 100.0f + (float) i;
    }

    if (ggml_cuda8_backend_buffer_upload(buf, offset_bytes, &h_sub_in[0], sub_bytes) != 0) {
        std::fprintf(stderr, "ggml-cuda8-buffer-smoke: offset upload failed\n");
        ggml_cuda8_backend_buffer_free(buf);
        return 1;
    }

    if (ggml_cuda8_backend_buffer_download(buf, offset_bytes, &h_sub_out[0], sub_bytes) != 0) {
        std::fprintf(stderr, "ggml-cuda8-buffer-smoke: offset download failed\n");
        ggml_cuda8_backend_buffer_free(buf);
        return 1;
    }

    if (!check_float_equal(h_sub_in, h_sub_out, "offset roundtrip")) {
        ggml_cuda8_backend_buffer_free(buf);
        return 1;
    }

    std::printf("offset upload/download PASS\n");

    // Clear test.
    if (ggml_cuda8_backend_buffer_clear(buf, 0) != 0) {
        std::fprintf(stderr, "ggml-cuda8-buffer-smoke: clear failed\n");
        ggml_cuda8_backend_buffer_free(buf);
        return 1;
    }

    std::vector<float> h_zero(n_floats, 0.0f);
    std::fill(h_out.begin(), h_out.end(), -1.0f);

    if (ggml_cuda8_backend_buffer_download(buf, 0, &h_out[0], bytes) != 0) {
        std::fprintf(stderr, "ggml-cuda8-buffer-smoke: clear download failed\n");
        ggml_cuda8_backend_buffer_free(buf);
        return 1;
    }

    if (!check_float_equal(h_zero, h_out, "clear")) {
        ggml_cuda8_backend_buffer_free(buf);
        return 1;
    }

    std::printf("clear PASS\n");

    if (ggml_cuda8_backend_buffer_free(buf) != 0) {
        std::fprintf(stderr, "ggml-cuda8-buffer-smoke: free failed\n");
        return 1;
    }

    std::printf("ggml-cuda8-buffer-smoke: SUCCESS\n");
    return 0;
}
