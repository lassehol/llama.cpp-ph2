// ggml/src/ggml-cuda8/ggml-cuda8-ggml-backend-graph-compute-softmax-smoke.cpp
//
// G15C smoke: real backend->iface.graph_compute dispatch of a three-node graph:
//   C = A + B
//   D = C * scalar
//   S = softmax_rows(D)

#include "ggml-cuda8-ggml-backend.h"
#include "ggml-cuda8-ggml-buffer.h"

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstdlib>
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
    setup_f32_matrix_2d(t, 1, 1, data);
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
    std::printf("graph_compute softmax final: max_abs_err=%.9g row_sum_abs=%.9g\n", max_abs_err, max_row_sum_abs_err);
    if (max_abs_err > 3e-5 || max_row_sum_abs_err > 3e-5) {
        std::fprintf(stderr, "graph_compute softmax verification FAIL\n");
        return false;
    }
    std::printf("graph_compute softmax verification PASS\n");
    return true;
}

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    std::printf("ggml-cuda8-ggml-backend-graph-compute-softmax-smoke: starting\n");

    ggml_backend_t backend = ggml_cuda8_ggml_backend_init(0);
    if (backend == NULL) { std::fprintf(stderr, "backend init failed\n"); return 1; }
    if (!ggml_cuda8_ggml_backend_is_cuda8(backend)) { std::fprintf(stderr, "backend identity check failed\n"); backend->iface.free(backend); return 1; }
    if (backend->iface.graph_compute == NULL) { std::fprintf(stderr, "backend graph_compute callback is NULL\n"); backend->iface.free(backend); return 1; }

    std::printf("backend name: %s\n", backend->iface.get_name(backend));

    ggml_backend_buffer_type_t buft = ggml_cuda8_ggml_backend_get_default_buffer_type(backend);
    if (buft == NULL) { std::fprintf(stderr, "backend default buffer type is NULL\n"); backend->iface.free(backend); return 1; }
    std::printf("backend default buffer type: %s\n", buft->iface.get_name(buft));

    const int rows = 32;
    const int cols = 128;
    const size_t n = (size_t) rows * (size_t) cols;
    const size_t matrix_bytes = n * sizeof(float);

    const size_t off_a = 0;
    const size_t off_b = 32768;
    const size_t off_c = 65536;
    const size_t off_d = 98304;
    const size_t off_s = 131072;
    const size_t off_scalar = 196608;
    const size_t off_dummy = 196864;
    const size_t total_size = 200704;

    ggml_backend_buffer_t buffer = buft->iface.alloc_buffer(buft, total_size);
    if (buffer == NULL) { std::fprintf(stderr, "buffer alloc failed\n"); backend->iface.free(backend); return 1; }
    void * base = buffer->iface.get_base(buffer);
    if (base == NULL) { std::fprintf(stderr, "buffer base is NULL\n"); buffer->iface.free_buffer(buffer); backend->iface.free(backend); return 1; }

    std::printf("backend-owned graph buffer size: %zu\n", buffer->size);
    std::printf("buffer base:                     %p\n", base);

    uint8_t * base_u8 = (uint8_t *) base;

    ggml_tensor a, b, c, d, s, scalar, dummy;
    setup_f32_matrix_2d(a, cols, rows, base_u8 + off_a);
    setup_f32_matrix_2d(b, cols, rows, base_u8 + off_b);
    setup_f32_matrix_2d(c, cols, rows, base_u8 + off_c);
    setup_f32_matrix_2d(d, cols, rows, base_u8 + off_d);
    setup_f32_matrix_2d(s, cols, rows, base_u8 + off_s);
    setup_f32_scalar(scalar, base_u8 + off_scalar);
    setup_f32_scalar(dummy, base_u8 + off_dummy);

    c.op = GGML_OP_ADD;
    c.src[0] = &a;
    c.src[1] = &b;

    d.op = GGML_OP_MUL;
    d.src[0] = &c;
    d.src[1] = &scalar;

    // G37: SOFT_MAX must mirror what ggml_soft_max() emits - src[1] is the mask
    // slot, not a filler, and op_params is { scale, max_bias }. A dummy in src[1]
    // or a zeroed op_params block now reads as a soft_max_ext node and is refused.
    s.op = GGML_OP_SOFT_MAX;
    s.src[0] = &d;
    s.src[1] = NULL;
    {
        const float sm_params[2] = { 1.0f, 0.0f };
        std::memcpy(s.op_params, sm_params, sizeof(sm_params));
    }

    if (buffer->iface.init_tensor(buffer, &a) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &b) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &c) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &d) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &s) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &scalar) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &dummy) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "init_tensor failed\n");
        buffer->iface.free_buffer(buffer);
        backend->iface.free(backend);
        return 1;
    }

    if (!expect_resident("A", &a, matrix_bytes, buffer, off_a) ||
        !expect_resident("B", &b, matrix_bytes, buffer, off_b) ||
        !expect_resident("C", &c, matrix_bytes, buffer, off_c) ||
        !expect_resident("D", &d, matrix_bytes, buffer, off_d) ||
        !expect_resident("S", &s, matrix_bytes, buffer, off_s) ||
        !expect_resident("scalar", &scalar, sizeof(float), buffer, off_scalar) ||
        !expect_resident("dummy", &dummy, sizeof(float), buffer, off_dummy)) {
        buffer->iface.free_buffer(buffer);
        backend->iface.free(backend);
        return 1;
    }

    std::vector<float> a_host(n), b_host(n), d_ref(n), s_ref(n), s_out(n, 0.0f);
    const float scale = -1.5f;
    const float zero = 0.0f;
    for (int r = 0; r < rows; ++r) {
        for (int col = 0; col < cols; ++col) {
            const size_t i = (size_t) r * cols + col;
            a_host[i] = ((float) ((r * 17 + col * 31 + 7) % 101) - 50.0f) * 0.03125f;
            b_host[i] = ((float) ((r * 13 + col * 19 + 5) % 97) - 48.0f) * 0.015625f;
            d_ref[i] = (a_host[i] + b_host[i]) * scale;
        }
    }
    cpu_softmax_rows(d_ref, s_ref, rows, cols);

    buffer->iface.clear(buffer, 0);
    buffer->iface.set_tensor(buffer, &a, &a_host[0], 0, matrix_bytes);
    buffer->iface.set_tensor(buffer, &b, &b_host[0], 0, matrix_bytes);
    buffer->iface.set_tensor(buffer, &scalar, &scale, 0, sizeof(float));
    buffer->iface.set_tensor(buffer, &dummy, &zero, 0, sizeof(float));

    ggml_tensor * nodes[3];
    nodes[0] = &c;
    nodes[1] = &d;
    nodes[2] = &s;

    ggml_cgraph graph;
    std::memset(&graph, 0, sizeof(graph));
    graph.n_nodes = 3;
    graph.nodes = nodes;

    std::printf("synthetic graph n_nodes=%d\n", graph.n_nodes);
    for (int i = 0; i < graph.n_nodes; ++i) {
        std::printf("graph node %d op=%d\n", i, (int) graph.nodes[i]->op);
    }

    enum ggml_status status = backend->iface.graph_compute(backend, &graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "backend graph_compute returned %d\n", (int) status);
        buffer->iface.free_buffer(buffer);
        backend->iface.free(backend);
        return 1;
    }

    buffer->iface.get_tensor(buffer, &s, &s_out[0], 0, matrix_bytes);

    if (!check_softmax_matrix(s_out, s_ref, rows, cols)) {
        buffer->iface.free_buffer(buffer);
        backend->iface.free(backend);
        return 1;
    }

    buffer->iface.free_buffer(buffer);
    backend->iface.free(backend);

    std::printf("ggml-cuda8-ggml-backend-graph-compute-softmax-smoke: SUCCESS\n");
    return 0;
}
