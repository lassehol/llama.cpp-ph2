// ggml/src/ggml-cuda8/ggml-cuda8-ggml-backend-graph-builder-softmax-smoke.cpp
// G16C smoke: real GGML graph-builder ADD -> scalar MUL -> SOFTMAX graph.

#include "ggml-cuda8-ggml-backend.h"
#include "ggml-cuda8-ggml-buffer.h"
#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <vector>

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

static void force_2d_f32_data_layout(ggml_tensor * t, int64_t cols, int64_t rows, void * data) {
    t->type = GGML_TYPE_F32;
    t->ne[0] = cols;
    t->ne[1] = rows;
    t->ne[2] = 1;
    t->ne[3] = 1;
    t->nb[0] = sizeof(float);
    t->nb[1] = (size_t) cols * sizeof(float);
    t->nb[2] = (size_t) cols * (size_t) rows * sizeof(float);
    t->nb[3] = t->nb[2];
    t->data = data;
}

static void force_1d_f32_data_layout(ggml_tensor * t, int64_t n, void * data) {
    t->type = GGML_TYPE_F32;
    t->ne[0] = n;
    t->ne[1] = 1;
    t->ne[2] = 1;
    t->ne[3] = 1;
    t->nb[0] = sizeof(float);
    t->nb[1] = (size_t) n * sizeof(float);
    t->nb[2] = t->nb[1];
    t->nb[3] = t->nb[1];
    t->data = data;
}

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    std::printf("ggml-cuda8-ggml-backend-graph-builder-softmax-smoke: starting\n");

    const int cols = 128;
    const int rows = 64;
    const int n = cols * rows;
    const size_t mat_bytes = (size_t) n * sizeof(float);
    const size_t scalar_bytes = sizeof(float);

    std::vector<uint8_t> ggml_mem(4 * 1024 * 1024);
    struct ggml_init_params params;
    std::memset(&params, 0, sizeof(params));
    params.mem_size = ggml_mem.size();
    params.mem_buffer = ggml_mem.data();
    params.no_alloc = false;

    ggml_context * gctx = ggml_init(params);
    if (gctx == NULL) {
        std::fprintf(stderr, "ggml_init failed\n");
        return 1;
    }

    ggml_tensor * a = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, cols, rows);
    ggml_tensor * b = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, cols, rows);
    ggml_tensor * scale = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 1);
    ggml_tensor * c = ggml_add(gctx, a, b);
    ggml_tensor * d = ggml_mul(gctx, c, scale);
    ggml_tensor * s = ggml_soft_max(gctx, d);

    if (a == NULL || b == NULL || scale == NULL || c == NULL || d == NULL || s == NULL) {
        std::fprintf(stderr, "ggml graph tensor creation failed\n");
        ggml_free(gctx);
        return 1;
    }

    ggml_cgraph * graph = ggml_new_graph(gctx);
    if (graph == NULL) {
        std::fprintf(stderr, "ggml_new_graph failed\n");
        ggml_free(gctx);
        return 1;
    }

    ggml_build_forward_expand(graph, s);

    std::printf("real ggml graph n_nodes=%d n_leafs=%d\n", graph->n_nodes, graph->n_leafs);
    for (int i = 0; i < graph->n_nodes; ++i) {
        std::printf("real graph node %d op=%d\n", i, graph->nodes[i] ? (int) graph->nodes[i]->op : -1);
    }

    if (graph->n_nodes != 3) {
        std::fprintf(stderr, "expected real ggml graph to have 3 nodes, got %d\n", graph->n_nodes);
        ggml_free(gctx);
        return 1;
    }

    ggml_backend_t backend = ggml_cuda8_ggml_backend_init(0);
    if (backend == NULL) {
        std::fprintf(stderr, "backend init failed\n");
        ggml_free(gctx);
        return 1;
    }
    if (!ggml_cuda8_ggml_backend_is_cuda8(backend)) {
        std::fprintf(stderr, "backend identity check failed\n");
        backend->iface.free(backend);
        ggml_free(gctx);
        return 1;
    }
    if (backend->iface.graph_compute == NULL) {
        std::fprintf(stderr, "backend graph_compute callback is NULL\n");
        backend->iface.free(backend);
        ggml_free(gctx);
        return 1;
    }

    std::printf("backend name: %s\n", backend->iface.get_name(backend));

    ggml_backend_buffer_type_t buft = ggml_cuda8_ggml_backend_get_default_buffer_type(backend);
    if (buft == NULL) {
        std::fprintf(stderr, "backend default buffer type is NULL\n");
        backend->iface.free(backend);
        ggml_free(gctx);
        return 1;
    }
    std::printf("backend default buffer type: %s\n", buft->iface.get_name(buft));

    const size_t off_a     = 0;
    const size_t off_b     = 32768;
    const size_t off_c     = 65536;
    const size_t off_d     = 98304;
    const size_t off_s     = 131072;
    const size_t off_scale = 163840;
    const size_t total_size = 168960;

    ggml_backend_buffer_t buffer = buft->iface.alloc_buffer(buft, total_size);
    if (buffer == NULL) {
        std::fprintf(stderr, "buffer alloc failed\n");
        backend->iface.free(backend);
        ggml_free(gctx);
        return 1;
    }

    void * base_ptr = buffer->iface.get_base(buffer);
    if (base_ptr == NULL) {
        std::fprintf(stderr, "buffer base is NULL\n");
        buffer->iface.free_buffer(buffer);
        backend->iface.free(backend);
        ggml_free(gctx);
        return 1;
    }
    std::printf("backend-owned graph-builder softmax buffer size: %zu\n", buffer->size);
    std::printf("buffer base:                            %p\n", base_ptr);

    uint8_t * base_u8 = (uint8_t *) base_ptr;
    force_2d_f32_data_layout(a, cols, rows, base_u8 + off_a);
    force_2d_f32_data_layout(b, cols, rows, base_u8 + off_b);
    force_2d_f32_data_layout(c, cols, rows, base_u8 + off_c);
    force_2d_f32_data_layout(d, cols, rows, base_u8 + off_d);
    force_2d_f32_data_layout(s, cols, rows, base_u8 + off_s);
    force_1d_f32_data_layout(scale, 1, base_u8 + off_scale);

    if (buffer->iface.init_tensor(buffer, a) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, b) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, c) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, d) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, s) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, scale) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "init_tensor failed\n");
        buffer->iface.free_buffer(buffer);
        backend->iface.free(backend);
        ggml_free(gctx);
        return 1;
    }

    if (!expect_resident("A", a, mat_bytes, buffer, off_a) ||
        !expect_resident("B", b, mat_bytes, buffer, off_b) ||
        !expect_resident("C", c, mat_bytes, buffer, off_c) ||
        !expect_resident("D", d, mat_bytes, buffer, off_d) ||
        !expect_resident("S", s, mat_bytes, buffer, off_s) ||
        !expect_resident("scale", scale, scalar_bytes, buffer, off_scale)) {
        buffer->iface.free_buffer(buffer);
        backend->iface.free(backend);
        ggml_free(gctx);
        return 1;
    }

    std::vector<float> a_host(n), b_host(n), s_ref(n), s_out(n, 0.0f);
    const float scale_host = 0.375f;
    for (int r = 0; r < rows; ++r) {
        float row_max = -1.0e30f;
        for (int ccol = 0; ccol < cols; ++ccol) {
            const int idx = r * cols + ccol;
            a_host[idx] = ((float) ((idx * 17 + 3) % 101) - 50.0f) * 0.03125f;
            b_host[idx] = ((float) ((idx * 19 + 5) % 97) - 48.0f) * 0.015625f;
            const float v = (a_host[idx] + b_host[idx]) * scale_host;
            if (v > row_max) row_max = v;
        }
        double sum = 0.0;
        for (int ccol = 0; ccol < cols; ++ccol) {
            const int idx = r * cols + ccol;
            const float v = (a_host[idx] + b_host[idx]) * scale_host;
            const double e = std::exp((double) v - (double) row_max);
            s_ref[idx] = (float) e;
            sum += e;
        }
        for (int ccol = 0; ccol < cols; ++ccol) {
            const int idx = r * cols + ccol;
            s_ref[idx] = (float) ((double) s_ref[idx] / sum);
        }
    }

    buffer->iface.clear(buffer, 0);
    buffer->iface.set_tensor(buffer, a, &a_host[0], 0, mat_bytes);
    buffer->iface.set_tensor(buffer, b, &b_host[0], 0, mat_bytes);
    buffer->iface.set_tensor(buffer, scale, &scale_host, 0, scalar_bytes);

    enum ggml_status status = backend->iface.graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "backend graph_compute returned %d\n", (int) status);
        buffer->iface.free_buffer(buffer);
        backend->iface.free(backend);
        ggml_free(gctx);
        return 1;
    }

    buffer->iface.get_tensor(buffer, s, &s_out[0], 0, mat_bytes);

    double max_abs_err = 0.0;
    double row_sum_abs = 0.0;
    for (int r = 0; r < rows; ++r) {
        double row_sum = 0.0;
        for (int ccol = 0; ccol < cols; ++ccol) {
            const int idx = r * cols + ccol;
            const double diff = (double) s_out[idx] - (double) s_ref[idx];
            const double abs_err = diff < 0.0 ? -diff : diff;
            if (abs_err > max_abs_err) max_abs_err = abs_err;
            row_sum += (double) s_out[idx];
        }
        const double row_err = std::fabs(row_sum - 1.0);
        if (row_err > row_sum_abs) row_sum_abs = row_err;
    }

    std::printf("real graph-builder ADD->MUL_SCALAR->SOFTMAX max_abs_err=%.9g row_sum_abs=%.9g\n", max_abs_err, row_sum_abs);
    if (max_abs_err > 5e-6 || row_sum_abs > 5e-6) {
        std::fprintf(stderr, "real graph-builder ADD->MUL_SCALAR->SOFTMAX verification FAIL\n");
        buffer->iface.free_buffer(buffer);
        backend->iface.free(backend);
        ggml_free(gctx);
        return 1;
    }

    std::printf("real graph-builder ADD->MUL_SCALAR->SOFTMAX verification PASS\n");

    buffer->iface.free_buffer(buffer);
    backend->iface.free(backend);
    ggml_free(gctx);

    std::printf("ggml-cuda8-ggml-backend-graph-builder-softmax-smoke: SUCCESS\n");
    return 0;
}
