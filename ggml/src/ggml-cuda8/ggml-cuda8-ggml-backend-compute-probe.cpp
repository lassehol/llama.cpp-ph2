// ggml/src/ggml-cuda8/ggml-cuda8-ggml-backend-compute-probe.cpp
//
// G13A smoke: graph-compute-shaped backend callback probe.
//
// This does not wire into full ggml_backend_i graph compute yet. Instead it
// validates a backend-owned compute callback shape:
//
//   backend_dispatch_op(backend, ctx, op, src0, src1, dst)
//
// The callback validates the backend object and forwards supported ops to the
// existing CUDA8 residency-aware dispatcher.
//
// Backend-owned buffer graph:
//   C = A + B
//   D = C * scalar
//   S = softmax_rows(D)

#include "ggml-cuda8-ggml-backend.h"
#include "ggml-cuda8-ggml-buffer.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstring>
#include <stdint.h>
#include <vector>
#include <cmath>
#include <float.h>

static void setup_f32_matrix_2d(ggml_tensor & t, int64_t cols, int64_t rows, void * data) {
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

static void cpu_softmax_rows(const std::vector<float> & src, std::vector<float> & dst, int rows, int cols) {
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

static bool expect_resident(const char * label, const ggml_tensor * t, size_t bytes, ggml_backend_buffer_t expected, size_t expected_offset) {
    ggml_backend_buffer_t owner = NULL;
    size_t offset = 0;
    if (!ggml_cuda8_ggml_tensor_is_device_resident(t, bytes, &owner, &offset)) {
        std::fprintf(stderr, "%s: residency not found\n", label);
        return false;
    }
    if (owner != expected || offset != expected_offset) {
        std::fprintf(stderr, "%s: residency mismatch offset=%zu expected=%zu\n", label, offset, expected_offset);
        return false;
    }
    std::printf("%s residency PASS offset=%zu\n", label, offset);
    return true;
}

static bool check_softmax_matrix(const std::vector<float> & got, const std::vector<float> & ref, int rows, int cols) {
    double max_abs_err = 0.0;
    double max_row_sum_abs_err = 0.0;
    for (int r = 0; r < rows; ++r) {
        double row_sum = 0.0;
        for (int c = 0; c < cols; ++c) {
            const size_t i = (size_t) r * cols + c;
            const double diff = (double) got[i] - (double) ref[i];
            const double abs_err = diff < 0.0 ? -diff : diff;
            if (abs_err > max_abs_err) max_abs_err = abs_err;
            row_sum += (double) got[i];
        }
        const double sum_diff = row_sum - 1.0;
        const double sum_abs_err = sum_diff < 0.0 ? -sum_diff : sum_diff;
        if (sum_abs_err > max_row_sum_abs_err) max_row_sum_abs_err = sum_abs_err;
    }
    std::printf("backend compute-shaped softmax graph: max_abs_err=%.9g row_sum_abs=%.9g\n", max_abs_err, max_row_sum_abs_err);
    if (max_abs_err > 3e-5 || max_row_sum_abs_err > 3e-5) {
        std::fprintf(stderr, "backend compute-shaped softmax graph FAIL\n");
        return false;
    }
    std::printf("backend compute-shaped softmax graph PASS\n");
    return true;
}

static int run_backend_op(
    ggml_backend_t backend,
    ggml_cuda8_context * ctx,
    int op,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst,
    const char * label
) {
    std::printf("backend compute callback: %s\n", label);
    const int rc = ggml_cuda8_ggml_backend_dispatch_op(backend, ctx, op, src0, src1, dst);
    if (rc != 0) {
        std::fprintf(stderr, "backend compute callback failed: %s\n", label);
        return -1;
    }
    std::printf("backend compute callback PASS: %s\n", label);
    return 0;
}

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    std::printf("ggml-cuda8-ggml-backend-compute-probe: starting\n");

    ggml_backend_t backend = ggml_cuda8_ggml_backend_init(0);
    if (backend == NULL) {
        std::fprintf(stderr, "backend init failed\n");
        return 1;
    }

    if (!ggml_cuda8_ggml_backend_is_cuda8(backend)) {
        std::fprintf(stderr, "backend identity check failed\n");
        backend->iface.free(backend);
        return 1;
    }

    std::printf("backend name: %s\n", backend->iface.get_name(backend));

    ggml_backend_buffer_type_t buft = ggml_cuda8_ggml_backend_get_default_buffer_type(backend);
    if (buft == NULL) {
        std::fprintf(stderr, "backend default buffer type is NULL\n");
        backend->iface.free(backend);
        return 1;
    }

    std::printf("backend default buffer type: %s\n", buft->iface.get_name(buft));
    std::printf("alignment:                   %zu\n", buft->iface.get_alignment(buft));
    std::printf("is_host:                     %d\n", buft->iface.is_host(buft) ? 1 : 0);

    ggml_cuda8_context * ctx = NULL;
    if (ggml_cuda8_context_create(0, &ctx) != 0 || ctx == NULL) {
        std::fprintf(stderr, "failed to create CUDA8 dispatcher context\n");
        backend->iface.free(backend);
        return 1;
    }
    ggml_cuda8_context_print(ctx);

    const int rows = 32;
    const int cols = 128;
    const size_t n = (size_t) rows * (size_t) cols;
    const size_t matrix_bytes = n * sizeof(float);

    const size_t off_a      = 0;
    const size_t off_b      = 32768;
    const size_t off_c      = 65536;
    const size_t off_d      = 98304;
    const size_t off_s      = 131072;
    const size_t off_scalar = 196608;
    const size_t total_size = 200704;

    ggml_backend_buffer_t buffer = buft->iface.alloc_buffer(buft, total_size);
    if (buffer == NULL) {
        std::fprintf(stderr, "backend-owned buffer allocation failed\n");
        ggml_cuda8_context_destroy(ctx);
        backend->iface.free(backend);
        return 1;
    }

    if (!ggml_cuda8_ggml_buffer_is_cuda8(buffer)) {
        std::fprintf(stderr, "backend-owned buffer is not CUDA8\n");
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        backend->iface.free(backend);
        return 1;
    }

    void * base = buffer->iface.get_base(buffer);
    if (base == NULL) {
        std::fprintf(stderr, "buffer base is NULL\n");
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        backend->iface.free(backend);
        return 1;
    }

    std::printf("backend-owned compute buffer size: %zu\n", buffer->size);
    std::printf("buffer base:                       %p\n", base);

    uint8_t * base_u8 = (uint8_t *) base;

    ggml_tensor t_a;
    ggml_tensor t_b;
    ggml_tensor t_c;
    ggml_tensor t_d;
    ggml_tensor t_s;
    ggml_tensor t_scalar;
    ggml_tensor t_dummy;
    ggml_tensor t_a_vec;
    ggml_tensor t_b_vec;
    ggml_tensor t_c_vec;
    ggml_tensor t_d_vec;

    setup_f32_matrix_2d(t_a, cols, rows, base_u8 + off_a);
    setup_f32_matrix_2d(t_b, cols, rows, base_u8 + off_b);
    setup_f32_matrix_2d(t_c, cols, rows, base_u8 + off_c);
    setup_f32_matrix_2d(t_d, cols, rows, base_u8 + off_d);
    setup_f32_matrix_2d(t_s, cols, rows, base_u8 + off_s);
    setup_f32_scalar(t_scalar, base_u8 + off_scalar);
    setup_f32_matrix_2d(t_dummy, 1, 1, base_u8 + off_s);
    setup_f32_matrix_2d(t_a_vec, (int64_t) n, 1, base_u8 + off_a);
    setup_f32_matrix_2d(t_b_vec, (int64_t) n, 1, base_u8 + off_b);
    setup_f32_matrix_2d(t_c_vec, (int64_t) n, 1, base_u8 + off_c);
    setup_f32_matrix_2d(t_d_vec, (int64_t) n, 1, base_u8 + off_d);

    if (buffer->iface.init_tensor(buffer, &t_a) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &t_b) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &t_c) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &t_d) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &t_s) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &t_scalar) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &t_dummy) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &t_a_vec) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &t_b_vec) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &t_c_vec) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &t_d_vec) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "init_tensor failed\n");
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        backend->iface.free(backend);
        return 1;
    }

    if (!expect_resident("A", &t_a, matrix_bytes, buffer, off_a) ||
        !expect_resident("B", &t_b, matrix_bytes, buffer, off_b) ||
        !expect_resident("C", &t_c, matrix_bytes, buffer, off_c) ||
        !expect_resident("D", &t_d, matrix_bytes, buffer, off_d) ||
        !expect_resident("S", &t_s, matrix_bytes, buffer, off_s) ||
        !expect_resident("scalar", &t_scalar, sizeof(float), buffer, off_scalar)) {
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        backend->iface.free(backend);
        return 1;
    }

    std::vector<float> a_host(n);
    std::vector<float> b_host(n);
    std::vector<float> c_ref(n);
    std::vector<float> d_ref(n);
    std::vector<float> s_ref(n);
    std::vector<float> s_out(n, 0.0f);

    const float scalar = -1.5f;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const size_t i = (size_t) r * cols + c;
            const int va = (r * 17 + c * 31 + 7) % 101;
            const int vb = (r * 13 + c * 19 + 5) % 97;
            a_host[i] = ((float) va - 50.0f) * 0.03125f;
            b_host[i] = ((float) vb - 48.0f) * 0.015625f;
            c_ref[i] = a_host[i] + b_host[i];
            d_ref[i] = c_ref[i] * scalar;
        }
    }

    cpu_softmax_rows(d_ref, s_ref, rows, cols);

    buffer->iface.clear(buffer, 0);
    buffer->iface.set_tensor(buffer, &t_a, &a_host[0], 0, matrix_bytes);
    buffer->iface.set_tensor(buffer, &t_b, &b_host[0], 0, matrix_bytes);
    buffer->iface.set_tensor(buffer, &t_scalar, &scalar, 0, sizeof(float));

    if (run_backend_op(backend, ctx, GGML_CUDA8_OP_ADD_F32, &t_a_vec, &t_b_vec, &t_c_vec, "C = A + B") != 0 ||
        run_backend_op(backend, ctx, GGML_CUDA8_OP_MUL_SCALAR_F32, &t_c_vec, &t_scalar, &t_d_vec, "D = C * scalar") != 0 ||
        run_backend_op(backend, ctx, GGML_CUDA8_OP_SOFTMAX_ROWS_F32, &t_d, &t_dummy, &t_s, "S = softmax_rows(D)") != 0) {
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        backend->iface.free(backend);
        return 1;
    }

    buffer->iface.get_tensor(buffer, &t_s, &s_out[0], 0, matrix_bytes);

    if (!check_softmax_matrix(s_out, s_ref, rows, cols)) {
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        backend->iface.free(backend);
        return 1;
    }

    std::printf("backend compute-shaped device graph PASS\n");

    buffer->iface.free_buffer(buffer);
    ggml_cuda8_context_destroy(ctx);
    backend->iface.free(backend);

    std::printf("ggml-cuda8-ggml-backend-compute-probe: SUCCESS\n");
    return 0;
}
