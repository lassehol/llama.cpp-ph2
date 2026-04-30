// ggml/src/ggml-cuda8/ggml-cuda8-ggml-buffer-device-reduce-smoke.cpp
// G11D smoke: device-resident REDUCE_SUM_ROWS_F32 / REDUCE_MAX_ROWS_F32.

#include "ggml-cuda8-ggml-buffer.h"
#include "ggml-cuda8-dispatch.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstring>
#include <stdint.h>
#include <vector>
#include <float.h>

static void setup_f32_matrix_2d(ggml_tensor & t, int64_t cols, int64_t rows, void * data) {
    std::memset(&t, 0, sizeof(t));
    t.type = GGML_TYPE_F32;
    t.ne[0] = cols; t.ne[1] = rows; t.ne[2] = 1; t.ne[3] = 1;
    t.nb[0] = sizeof(float);
    t.nb[1] = (size_t) cols * sizeof(float);
    t.nb[2] = t.nb[1] * (size_t) rows;
    t.nb[3] = t.nb[2];
    t.data = data;
}

static void setup_f32_vector(ggml_tensor & t, int64_t n, void * data) {
    std::memset(&t, 0, sizeof(t));
    t.type = GGML_TYPE_F32;
    t.ne[0] = n; t.ne[1] = 1; t.ne[2] = 1; t.ne[3] = 1;
    t.nb[0] = sizeof(float);
    t.nb[1] = (size_t) n * sizeof(float);
    t.nb[2] = t.nb[1]; t.nb[3] = t.nb[1];
    t.data = data;
}

static bool expect_resident(const char * label, const ggml_tensor * t, size_t bytes, ggml_backend_buffer_t expected, size_t expected_offset) {
    ggml_backend_buffer_t owner = NULL;
    size_t offset = 0;
    if (!ggml_cuda8_ggml_tensor_is_device_resident(t, bytes, &owner, &offset)) { std::fprintf(stderr, "%s: residency not found\n", label); return false; }
    if (owner != expected || offset != expected_offset) { std::fprintf(stderr, "%s: residency mismatch offset=%zu expected=%zu\n", label, offset, expected_offset); return false; }
    std::printf("%s residency PASS offset=%zu\n", label, offset);
    return true;
}

static bool check_close(const std::vector<float> & got, const std::vector<float> & ref, double abs_tol, const char * label) {
    double max_abs_err = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double diff = (double) got[i] - (double) ref[i];
        const double abs_err = diff < 0.0 ? -diff : diff;
        if (abs_err > max_abs_err) max_abs_err = abs_err;
    }
    std::printf("%s: max_abs_err=%.9g\n", label, max_abs_err);
    if (max_abs_err > abs_tol) { std::fprintf(stderr, "%s FAIL\n", label); return false; }
    std::printf("%s PASS\n", label);
    return true;
}

int main(int argc, char ** argv) {
    (void) argc; (void) argv;
    std::printf("ggml-cuda8-ggml-buffer-device-reduce-smoke: starting\n");
    ggml_backend_buffer_type_t buft = ggml_cuda8_ggml_buffer_type();
    if (buft == NULL) return 1;
    std::printf("buffer type name: %s\n", buft->iface.get_name(buft));
    std::printf("alignment:        %zu\n", buft->iface.get_alignment(buft));
    std::printf("is_host:          %d\n", buft->iface.is_host(buft) ? 1 : 0);
    ggml_cuda8_context * ctx = NULL;
    if (ggml_cuda8_context_create(0, &ctx) != 0 || ctx == NULL) return 1;
    ggml_cuda8_context_print(ctx);

    const int rows = 64;
    const int cols = 128;
    const size_t src_bytes = (size_t) rows * (size_t) cols * sizeof(float);
    const size_t dst_bytes = (size_t) rows * sizeof(float);
    const size_t off_src = 0, off_sum = 65536, off_max = 66048, total_size = 131072;

    ggml_backend_buffer_t buffer = ggml_cuda8_ggml_buffer_alloc(total_size);
    if (buffer == NULL) { ggml_cuda8_context_destroy(ctx); return 1; }
    void * base = buffer->iface.get_base(buffer);
    if (base == NULL) { buffer->iface.free_buffer(buffer); ggml_cuda8_context_destroy(ctx); return 1; }
    std::printf("allocated buffer size: %zu\n", buffer->size);
    std::printf("buffer base:           %p\n", base);
    std::printf("src offset:            %zu\n", off_src);
    std::printf("sum offset:            %zu\n", off_sum);
    std::printf("max offset:            %zu\n", off_max);

    uint8_t * base_u8 = (uint8_t *) base;
    ggml_tensor t_src, t_sum, t_max, t_dummy;
    setup_f32_matrix_2d(t_src, cols, rows, base_u8 + off_src);
    setup_f32_vector(t_sum, rows, base_u8 + off_sum);
    setup_f32_vector(t_max, rows, base_u8 + off_max);
    setup_f32_vector(t_dummy, 1, base_u8 + off_sum);
    if (buffer->iface.init_tensor(buffer, &t_src) != GGML_STATUS_SUCCESS || buffer->iface.init_tensor(buffer, &t_sum) != GGML_STATUS_SUCCESS || buffer->iface.init_tensor(buffer, &t_max) != GGML_STATUS_SUCCESS || buffer->iface.init_tensor(buffer, &t_dummy) != GGML_STATUS_SUCCESS) { buffer->iface.free_buffer(buffer); ggml_cuda8_context_destroy(ctx); return 1; }
    if (!expect_resident("src matrix", &t_src, src_bytes, buffer, off_src) || !expect_resident("sum vector", &t_sum, dst_bytes, buffer, off_sum) || !expect_resident("max vector", &t_max, dst_bytes, buffer, off_max)) { buffer->iface.free_buffer(buffer); ggml_cuda8_context_destroy(ctx); return 1; }

    std::vector<float> src((size_t) rows * cols);
    std::vector<float> sum_ref(rows, 0.0f), max_ref(rows, -FLT_MAX), sum_out(rows, 0.0f), max_out(rows, 0.0f);
    for (int r = 0; r < rows; ++r) {
        float sum = 0.0f, vmax = -FLT_MAX;
        for (int c = 0; c < cols; ++c) {
            const int v = (r * 17 + c * 31 + 7) % 101;
            float x = ((float) v - 50.0f) * 0.015625f;
            if ((c % 37) == 0) x += (float) (r % 13) * 0.01f;
            src[(size_t) r * cols + c] = x;
            sum += x;
            vmax = vmax > x ? vmax : x;
        }
        sum_ref[r] = sum;
        max_ref[r] = vmax;
    }

    buffer->iface.clear(buffer, 0);
    buffer->iface.set_tensor(buffer, &t_src, &src[0], 0, src_bytes);
    if (!ggml_cuda8_dispatch_supported(ctx, GGML_CUDA8_OP_REDUCE_SUM_ROWS_F32, &t_src, &t_dummy, &t_sum)) return 1;
    if (ggml_cuda8_dispatch_execute(ctx, GGML_CUDA8_OP_REDUCE_SUM_ROWS_F32, &t_src, &t_dummy, &t_sum) != 0) return 1;
    if (!ggml_cuda8_dispatch_supported(ctx, GGML_CUDA8_OP_REDUCE_MAX_ROWS_F32, &t_src, &t_dummy, &t_max)) return 1;
    if (ggml_cuda8_dispatch_execute(ctx, GGML_CUDA8_OP_REDUCE_MAX_ROWS_F32, &t_src, &t_dummy, &t_max) != 0) return 1;
    buffer->iface.get_tensor(buffer, &t_sum, &sum_out[0], 0, dst_bytes);
    buffer->iface.get_tensor(buffer, &t_max, &max_out[0], 0, dst_bytes);

    bool ok = true;
    ok = ok && check_close(sum_out, sum_ref, 2e-4, "device REDUCE_SUM_ROWS_F32");
    ok = ok && check_close(max_out, max_ref, 1e-6, "device REDUCE_MAX_ROWS_F32");
    buffer->iface.free_buffer(buffer);
    ggml_cuda8_context_destroy(ctx);
    if (!ok) { std::fprintf(stderr, "ggml-cuda8-ggml-buffer-device-reduce-smoke: FAILED\n"); return 1; }
    std::printf("ggml-cuda8-ggml-buffer-device-reduce-smoke: SUCCESS\n");
    return 0;
}
