// ggml/src/ggml-cuda8/ggml-cuda8-ggml-backend-graph-compute-add-smoke.cpp
//
// G15A smoke, link-safe variant: real backend->iface.graph_compute dispatch
// of one tiny synthetic ADD_F32 graph.
//
// This variant deliberately avoids calling ggml_init / ggml_new_tensor_1d /
// ggml_add / ggml_new_graph so the CUDA8-only smoke executable does not need
// to link full ggml-base. It constructs the minimal ggml_tensor metadata and
// ggml_cgraph object directly, using the internal ggml-impl.h layout already
// required by the backend graph walker.

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

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    std::printf("ggml-cuda8-ggml-backend-graph-compute-add-smoke: starting\n");

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

    if (backend->iface.graph_compute == NULL) {
        std::fprintf(stderr, "backend graph_compute callback is NULL\n");
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

    const int n = 512;
    const size_t bytes = (size_t) n * sizeof(float);
    const size_t off_a = 0;
    const size_t off_b = 4096;
    const size_t off_c = 8192;
    const size_t total_size = 12288;

    ggml_backend_buffer_t buffer = buft->iface.alloc_buffer(buft, total_size);
    if (buffer == NULL) {
        std::fprintf(stderr, "backend-owned buffer allocation failed\n");
        backend->iface.free(backend);
        return 1;
    }

    void * base = buffer->iface.get_base(buffer);
    if (base == NULL) {
        std::fprintf(stderr, "buffer base is NULL\n");
        buffer->iface.free_buffer(buffer);
        backend->iface.free(backend);
        return 1;
    }

    std::printf("backend-owned graph buffer size: %zu\n", buffer->size);
    std::printf("buffer base:                     %p\n", base);
    std::printf("A offset:                        %zu\n", off_a);
    std::printf("B offset:                        %zu\n", off_b);
    std::printf("C offset:                        %zu\n", off_c);

    uint8_t * base_u8 = (uint8_t *) base;

    ggml_tensor a;
    ggml_tensor b;
    ggml_tensor c;

    setup_f32_vector(a, n, base_u8 + off_a);
    setup_f32_vector(b, n, base_u8 + off_b);
    setup_f32_vector(c, n, base_u8 + off_c);

    c.op = GGML_OP_ADD;
    c.src[0] = &a;
    c.src[1] = &b;

    if (buffer->iface.init_tensor(buffer, &a) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &b) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, &c) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "init_tensor failed\n");
        buffer->iface.free_buffer(buffer);
        backend->iface.free(backend);
        return 1;
    }

    if (!expect_resident("A", &a, bytes, buffer, off_a) ||
        !expect_resident("B", &b, bytes, buffer, off_b) ||
        !expect_resident("C", &c, bytes, buffer, off_c)) {
        buffer->iface.free_buffer(buffer);
        backend->iface.free(backend);
        return 1;
    }

    std::vector<float> a_host(n);
    std::vector<float> b_host(n);
    std::vector<float> c_ref(n);
    std::vector<float> c_out(n, 0.0f);

    for (int i = 0; i < n; ++i) {
        a_host[i] = ((float) ((i * 17 + 3) % 101) - 50.0f) * 0.03125f;
        b_host[i] = ((float) ((i * 19 + 5) % 97) - 48.0f) * 0.015625f;
        c_ref[i] = a_host[i] + b_host[i];
    }

    buffer->iface.clear(buffer, 0);
    buffer->iface.set_tensor(buffer, &a, &a_host[0], 0, bytes);
    buffer->iface.set_tensor(buffer, &b, &b_host[0], 0, bytes);

    ggml_tensor * nodes[1];
    nodes[0] = &c;

    ggml_cgraph graph;
    std::memset(&graph, 0, sizeof(graph));
    graph.n_nodes = 1;
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

    buffer->iface.get_tensor(buffer, &c, &c_out[0], 0, bytes);

    double max_abs_err = 0.0;
    for (int i = 0; i < n; ++i) {
        const double diff = (double) c_out[i] - (double) c_ref[i];
        const double abs_err = diff < 0.0 ? -diff : diff;
        if (abs_err > max_abs_err) max_abs_err = abs_err;
    }

    std::printf("graph_compute ADD_F32 max_abs_err=%.9g\n", max_abs_err);
    if (max_abs_err > 1e-6) {
        std::fprintf(stderr, "graph_compute ADD_F32 verification FAIL\n");
        buffer->iface.free_buffer(buffer);
        backend->iface.free(backend);
        return 1;
    }

    std::printf("graph_compute ADD_F32 verification PASS\n");

    buffer->iface.free_buffer(buffer);
    backend->iface.free(backend);

    std::printf("ggml-cuda8-ggml-backend-graph-compute-add-smoke: SUCCESS\n");
    return 0;
}
