// ggml/src/ggml-cuda8/ggml-cuda8-ggml-backend-attnlike-smoke.cpp
//
// G12C smoke: backend-owned attention-like device-resident micrograph.
//
// Validates:
//   - minimal CUDA8 ggml_backend_t can be created
//   - backend default buffer type helper returns CUDA8 buffer type
//   - attention-like graph buffer is allocated through backend default buffer type
//   - tensors in that backend-owned buffer are CUDA8-resident
//   - existing residency-aware dispatcher can run the attention-like graph
//
// Graph:
//   scores  = A + B
//   scaled  = scores * scale
//   row_max = reduce_max_rows(scaled)
//   probs   = softmax_rows(scaled)
//   row_sum = reduce_sum_rows(probs)

#include "ggml-cuda8-ggml-backend.h"
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

static bool check_close_vector(const std::vector<float> & got, const std::vector<float> & ref, double abs_tol, const char * label) {
    double max_abs_err = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double diff = (double) got[i] - (double) ref[i];
        const double abs_err = diff < 0.0 ? -diff : diff;
        if (abs_err > max_abs_err) max_abs_err = abs_err;
    }
    std::printf("%s: max_abs_err=%.9g\n", label, max_abs_err);
    if (max_abs_err > abs_tol) {
        std::fprintf(stderr, "%s FAIL\n", label);
        return false;
    }
    std::printf("%s PASS\n", label);
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
    std::printf("backend-owned attention probs: max_abs_err=%.9g row_sum_abs=%.9g\n", max_abs_err, max_row_sum_abs_err);
    if (max_abs_err > 3e-5 || max_row_sum_abs_err > 3e-5) {
        std::fprintf(stderr, "backend-owned attention probs FAIL\n");
        return false;
    }
    std::printf("backend-owned attention probs PASS\n");
    return true;
}

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    std::printf("ggml-cuda8-ggml-backend-attnlike-smoke: starting\n");

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
    const size_t vector_bytes = (size_t) rows * sizeof(float);

    const size_t off_a       = 0;
    const size_t off_b       = 32768;
    const size_t off_scores  = 65536;
    const size_t off_scaled  = 98304;
    const size_t off_probs   = 131072;
    const size_t off_row_max = 163840;
    const size_t off_row_sum = 164352;
    const size_t off_scalar  = 164864;
    const size_t total_size  = 196608;

    if (off_a + matrix_bytes > total_size ||
        off_b + matrix_bytes > total_size ||
        off_scores + matrix_bytes > total_size ||
        off_scaled + matrix_bytes > total_size ||
        off_probs + matrix_bytes > total_size ||
        off_row_max + vector_bytes > total_size ||
        off_row_sum + vector_bytes > total_size ||
        off_scalar + sizeof(float) > total_size) {
        std::fprintf(stderr, "internal layout invalid\n");
        ggml_cuda8_context_destroy(ctx);
        backend->iface.free(backend);
        return 1;
    }

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

    std::printf("backend-owned buffer size: %zu\n", buffer->size);
    std::printf("buffer base:               %p\n", base);
    std::printf("A offset:                  %zu\n", off_a);
    std::printf("B offset:                  %zu\n", off_b);
    std::printf("scores offset:             %zu\n", off_scores);
    std::printf("scaled offset:             %zu\n", off_scaled);
    std::printf("probs offset:              %zu\n", off_probs);
    std::printf("row_max offset:            %zu\n", off_row_max);
    std::printf("row_sum offset:            %zu\n", off_row_sum);
    std::printf("scalar offset:             %zu\n", off_scalar);

    uint8_t * base_u8 = (uint8_t *) base;

    ggml_tensor t_a;
    ggml_tensor t_b;
    ggml_tensor t_scores;
    ggml_tensor t_scaled;
    ggml_tensor t_probs;
    ggml_tensor t_row_max;
    ggml_tensor t_row_sum;
    ggml_tensor t_scalar;
    ggml_tensor t_dummy;

    ggml_tensor t_a_vec;
    ggml_tensor t_b_vec;
    ggml_tensor t_scores_vec;
    ggml_tensor t_scaled_vec;

    setup_f32_matrix_2d(t_a, cols, rows, base_u8 + off_a);
    setup_f32_matrix_2d(t_b, cols, rows, base_u8 + off_b);
    setup_f32_matrix_2d(t_scores, cols, rows, base_u8 + off_scores);
    setup_f32_matrix_2d(t_scaled, cols, rows, base_u8 + off_scaled);
    setup_f32_matrix_2d(t_probs, cols, rows, base_u8 + off_probs);
    setup_f32_vector(t_row_max, rows, base_u8 + off_row_max);
    setup_f32_vector(t_row_sum, rows, base_u8 + off_row_sum);
    setup_f32_scalar(t_scalar, base_u8 + off_scalar);
    setup_f32_vector(t_dummy, 1, base_u8 + off_row_sum);

    setup_f32_matrix_2d(t_a_vec, (int64_t) n, 1, base_u8 + off_a);
    setup_f32_matrix_2d(t_b_vec, (int64_t) n, 1, base_u8 + off_b);
    setup_f32_matrix_2d(t_scores_vec, (int64_t) n, 1, base_u8 + off_scores);
    setup_f32_matrix_2d(t_scaled_vec, (int64_t) n, 1, base_u8 + off_scaled);

    if (buffer->iface.init_tensor(buffer, &t_a) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &t_b) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &t_scores) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &t_scaled) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &t_probs) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &t_row_max) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &t_row_sum) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &t_scalar) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &t_dummy) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &t_a_vec) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &t_b_vec) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &t_scores_vec) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &t_scaled_vec) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "init_tensor failed\n");
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        backend->iface.free(backend);
        return 1;
    }

    if (!expect_resident("A", &t_a, matrix_bytes, buffer, off_a) ||
        !expect_resident("B", &t_b, matrix_bytes, buffer, off_b) ||
        !expect_resident("scores", &t_scores, matrix_bytes, buffer, off_scores) ||
        !expect_resident("scaled", &t_scaled, matrix_bytes, buffer, off_scaled) ||
        !expect_resident("probs", &t_probs, matrix_bytes, buffer, off_probs) ||
        !expect_resident("row_max", &t_row_max, vector_bytes, buffer, off_row_max) ||
        !expect_resident("row_sum", &t_row_sum, vector_bytes, buffer, off_row_sum) ||
        !expect_resident("scalar", &t_scalar, sizeof(float), buffer, off_scalar)) {
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        backend->iface.free(backend);
        return 1;
    }

    std::vector<float> a_host(n);
    std::vector<float> b_host(n);
    std::vector<float> scores_ref(n);
    std::vector<float> scaled_ref(n);
    std::vector<float> probs_ref(n);
    std::vector<float> probs_out(n, 0.0f);
    std::vector<float> row_max_ref(rows, -FLT_MAX);
    std::vector<float> row_sum_ref(rows, 1.0f);
    std::vector<float> row_max_out(rows, 0.0f);
    std::vector<float> row_sum_out(rows, 0.0f);

    const float scale = 0.125f;

    for (int r = 0; r < rows; ++r) {
        float vmax = -FLT_MAX;
        for (int c = 0; c < cols; ++c) {
            const size_t i = (size_t) r * cols + c;
            const int va = (r * 17 + c * 31 + 7) % 101;
            const int vb = (r * 13 + c * 19 + 5) % 97;
            a_host[i] = ((float) va - 50.0f) * 0.03125f;
            b_host[i] = ((float) vb - 48.0f) * 0.015625f;
            scores_ref[i] = a_host[i] + b_host[i];
            scaled_ref[i] = scores_ref[i] * scale;
            vmax = vmax > scaled_ref[i] ? vmax : scaled_ref[i];
        }
        row_max_ref[r] = vmax;
    }

    cpu_softmax_rows(scaled_ref, probs_ref, rows, cols);

    buffer->iface.clear(buffer, 0);
    buffer->iface.set_tensor(buffer, &t_a, &a_host[0], 0, matrix_bytes);
    buffer->iface.set_tensor(buffer, &t_b, &b_host[0], 0, matrix_bytes);
    buffer->iface.set_tensor(buffer, &t_scalar, &scale, 0, sizeof(float));

    if (!ggml_cuda8_dispatch_supported(ctx, GGML_CUDA8_OP_ADD_F32, &t_a_vec, &t_b_vec, &t_scores_vec)) {
        std::fprintf(stderr, "ADD_F32 unsupported\n");
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        backend->iface.free(backend);
        return 1;
    }
    if (ggml_cuda8_dispatch_execute(ctx, GGML_CUDA8_OP_ADD_F32, &t_a_vec, &t_b_vec, &t_scores_vec) != 0) {
        std::fprintf(stderr, "ADD_F32 failed\n");
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        backend->iface.free(backend);
        return 1;
    }
    std::printf("backend attention op 1: scores = A + B PASS\n");

    if (!ggml_cuda8_dispatch_supported(ctx, GGML_CUDA8_OP_MUL_SCALAR_F32, &t_scores_vec, &t_scalar, &t_scaled_vec)) {
        std::fprintf(stderr, "MUL_SCALAR_F32 unsupported\n");
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        backend->iface.free(backend);
        return 1;
    }
    if (ggml_cuda8_dispatch_execute(ctx, GGML_CUDA8_OP_MUL_SCALAR_F32, &t_scores_vec, &t_scalar, &t_scaled_vec) != 0) {
        std::fprintf(stderr, "MUL_SCALAR_F32 failed\n");
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        backend->iface.free(backend);
        return 1;
    }
    std::printf("backend attention op 2: scaled = scores * scale PASS\n");

    if (!ggml_cuda8_dispatch_supported(ctx, GGML_CUDA8_OP_REDUCE_MAX_ROWS_F32, &t_scaled, &t_dummy, &t_row_max)) {
        std::fprintf(stderr, "REDUCE_MAX_ROWS_F32 unsupported\n");
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        backend->iface.free(backend);
        return 1;
    }
    if (ggml_cuda8_dispatch_execute(ctx, GGML_CUDA8_OP_REDUCE_MAX_ROWS_F32, &t_scaled, &t_dummy, &t_row_max) != 0) {
        std::fprintf(stderr, "REDUCE_MAX_ROWS_F32 failed\n");
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        backend->iface.free(backend);
        return 1;
    }
    std::printf("backend attention op 3: row_max = reduce_max_rows(scaled) PASS\n");

    if (!ggml_cuda8_dispatch_supported(ctx, GGML_CUDA8_OP_SOFTMAX_ROWS_F32, &t_scaled, &t_dummy, &t_probs)) {
        std::fprintf(stderr, "SOFTMAX_ROWS_F32 unsupported\n");
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        backend->iface.free(backend);
        return 1;
    }
    if (ggml_cuda8_dispatch_execute(ctx, GGML_CUDA8_OP_SOFTMAX_ROWS_F32, &t_scaled, &t_dummy, &t_probs) != 0) {
        std::fprintf(stderr, "SOFTMAX_ROWS_F32 failed\n");
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        backend->iface.free(backend);
        return 1;
    }
    std::printf("backend attention op 4: probs = softmax_rows(scaled) PASS\n");

    if (!ggml_cuda8_dispatch_supported(ctx, GGML_CUDA8_OP_REDUCE_SUM_ROWS_F32, &t_probs, &t_dummy, &t_row_sum)) {
        std::fprintf(stderr, "REDUCE_SUM_ROWS_F32 unsupported\n");
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        backend->iface.free(backend);
        return 1;
    }
    if (ggml_cuda8_dispatch_execute(ctx, GGML_CUDA8_OP_REDUCE_SUM_ROWS_F32, &t_probs, &t_dummy, &t_row_sum) != 0) {
        std::fprintf(stderr, "REDUCE_SUM_ROWS_F32 failed\n");
        buffer->iface.free_buffer(buffer);
        ggml_cuda8_context_destroy(ctx);
        backend->iface.free(backend);
        return 1;
    }
    std::printf("backend attention op 5: row_sum = reduce_sum_rows(probs) PASS\n");

    buffer->iface.get_tensor(buffer, &t_probs, &probs_out[0], 0, matrix_bytes);
    buffer->iface.get_tensor(buffer, &t_row_max, &row_max_out[0], 0, vector_bytes);
    buffer->iface.get_tensor(buffer, &t_row_sum, &row_sum_out[0], 0, vector_bytes);

    bool ok = true;
    ok = ok && check_close_vector(row_max_out, row_max_ref, 1e-6, "backend attention row_max");
    ok = ok && check_softmax_matrix(probs_out, probs_ref, rows, cols);
    ok = ok && check_close_vector(row_sum_out, row_sum_ref, 3e-5, "backend attention row_sum(probs)");

    buffer->iface.free_buffer(buffer);
    ggml_cuda8_context_destroy(ctx);
    backend->iface.free(backend);

    if (!ok) {
        std::fprintf(stderr, "ggml-cuda8-ggml-backend-attnlike-smoke: FAILED\n");
        return 1;
    }

    std::printf("backend-owned attention-like graph PASS\n");
    std::printf("ggml-cuda8-ggml-backend-attnlike-smoke: SUCCESS\n");
    return 0;
}
