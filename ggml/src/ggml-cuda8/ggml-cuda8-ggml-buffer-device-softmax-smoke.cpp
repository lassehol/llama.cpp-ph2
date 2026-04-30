// ggml/src/ggml-cuda8/ggml-cuda8-ggml-buffer-device-softmax-smoke.cpp
//
// G11A-4E smoke: device-resident SOFTMAX_ROWS_F32 dispatcher path.

#include "ggml-cuda8-ggml-buffer.h"
#include "ggml-cuda8-dispatch.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstring>
#include <stdint.h>
#include <vector>
#include <cmath>
#include <float.h>

static void setup_f32_matrix_2d(
    ggml_tensor & t,
    int64_t cols,
    int64_t rows,
    void * data
) {
    std::memset(&t, 0, sizeof(t));

    t.type = GGML_TYPE_F32;

    t.ne[0] = cols;
    t.ne[1] = rows;
    t.ne[2] = 1;
    t.ne[3] = 1;

    t.nb[0] = sizeof(float);
    t.nb[1] = (size_t) cols * sizeof(float);
    t.nb[2] = t.nb[1] * (size_t) rows;
    t.nb[3] = t.nb[2];

    t.data = data;
}

static void cpu_softmax_rows(
    const std::vector<float> & src,
    std::vector<float> & dst,
    int rows,
    int cols
) {
    for (int r = 0; r < rows; ++r) {
        const float * row_src = &src[(size_t) r * cols];
        float * row_dst = &dst[(size_t) r * cols];

        float vmax = -FLT_MAX;

        for (int c = 0; c < cols; ++c) {
            vmax = vmax > row_src[c] ? vmax : row_src[c];
        }

        float sum = 0.0f;

        for (int c = 0; c < cols; ++c) {
            const float e = std::exp(row_src[c] - vmax);
            row_dst[c] = e;
            sum += e;
        }

        const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;

        for (int c = 0; c < cols; ++c) {
            row_dst[c] *= inv_sum;
        }
    }
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

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    std::printf("ggml-cuda8-ggml-buffer-device-softmax-smoke: starting\n");

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

    const int rows = 64;
    const int cols = 128;
    const size_t n = (size_t) rows * (size_t) cols;
    const size_t bytes = n * sizeof(float);

    const size_t off_src = 0;
    const size_t off_dst = 65536;
    const size_t total_size = 131072;

    if (off_src + bytes > total_size || off_dst + bytes > total_size) {
        std::fprintf(stderr, "internal layout invalid\n");
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

    void * base = buffer->iface.get_base(buffer);

    if (base == NULL) {
        std::fprintf(stderr, "buffer base is NULL\n");
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        return 1;
    }

    std::printf("allocated buffer size: %zu\n", buffer->size);
    std::printf("buffer base:           %p\n", base);
    std::printf("src offset:            %zu\n", off_src);
    std::printf("dst offset:            %zu\n", off_dst);

    uint8_t * base_u8 = (uint8_t *) base;

    ggml_tensor t_src;
    ggml_tensor t_dst;
    ggml_tensor t_dummy;

    setup_f32_matrix_2d(t_src, cols, rows, base_u8 + off_src);
    setup_f32_matrix_2d(t_dst, cols, rows, base_u8 + off_dst);

    setup_f32_matrix_2d(t_dummy, 1, 1, base_u8 + off_dst);

    if (buffer->iface.init_tensor(buffer, &t_src) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &t_dst) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &t_dummy) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "init_tensor failed\n");
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        return 1;
    }

    if (!expect_resident("src matrix", &t_src, bytes, buffer, off_src) ||
        !expect_resident("dst matrix", &t_dst, bytes, buffer, off_dst)) {
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        return 1;
    }

    std::vector<float> src(n);
    std::vector<float> ref(n, 0.0f);
    std::vector<float> out(n, 0.0f);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const int v = (r * 17 + c * 31 + 7) % 101;
            float x = ((float) v - 50.0f) * 0.03125f;

            if ((c % 29) == 0) {
                x += (float) (r % 7) * 0.02f;
            }

            src[(size_t) r * cols + c] = x;
        }
    }

    cpu_softmax_rows(src, ref, rows, cols);

    buffer->iface.clear(buffer, 0);
    buffer->iface.set_tensor(buffer, &t_src, &src[0], 0, bytes);

    const int op = GGML_CUDA8_OP_SOFTMAX_ROWS_F32;

    if (!ggml_cuda8_dispatch_supported(ctx, op, &t_src, &t_dummy, &t_dst)) {
        std::fprintf(stderr, "SOFTMAX_ROWS_F32 unsupported for device tensors\n");
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        return 1;
    }

    if (ggml_cuda8_dispatch_execute(ctx, op, &t_src, &t_dummy, &t_dst) != 0) {
        std::fprintf(stderr, "SOFTMAX_ROWS_F32 device dispatch failed\n");
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        return 1;
    }

    buffer->iface.get_tensor(buffer, &t_dst, &out[0], 0, bytes);

    double max_abs_err = 0.0;
    double max_row_sum_abs_err = 0.0;

    for (int r = 0; r < rows; ++r) {
        double row_sum = 0.0;

        for (int c = 0; c < cols; ++c) {
            const size_t i = (size_t) r * cols + c;

            const double diff = (double) out[i] - (double) ref[i];
            const double abs_err = diff < 0.0 ? -diff : diff;

            if (abs_err > max_abs_err) max_abs_err = abs_err;
            row_sum += (double) out[i];
        }

        const double sum_diff = row_sum - 1.0;
        const double sum_abs_err = sum_diff < 0.0 ? -sum_diff : sum_diff;

        if (sum_abs_err > max_row_sum_abs_err) {
            max_row_sum_abs_err = sum_abs_err;
        }
    }

    std::printf("device SOFTMAX_ROWS_F32: max_abs_err=%.9g row_sum_abs=%.9g\n",
        max_abs_err,
        max_row_sum_abs_err);

    if (max_abs_err > 2e-5 || max_row_sum_abs_err > 2e-5) {
        std::fprintf(stderr, "device SOFTMAX_ROWS_F32 FAIL\n");
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        return 1;
    }

    std::printf("device-resident SOFTMAX_ROWS_F32 PASS\n");

    buffer->iface.free_buffer(buffer);
    ggml_cuda8_context_destroy(ctx);

    std::printf("ggml-cuda8-ggml-buffer-device-softmax-smoke: SUCCESS\n");
    return 0;
}
