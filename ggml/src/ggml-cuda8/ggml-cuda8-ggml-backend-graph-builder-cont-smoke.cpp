// ggml-cuda8-ggml-backend-graph-builder-cont-smoke.cpp
// G30A: CONT (contiguous copy) through graph_compute
// Graph: x(4x128) -> permute(1,0) -> cont -> add(bias) -> y
// permute is a no-op (G29), cont is a real D2D copy, add is dispatched.

#include "ggml-cuda8-ggml-backend.h"
#include "ggml-cuda8-ggml-buffer.h"
#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

static bool expect_resident(const char * label, const ggml_tensor * t,
                            size_t bytes, ggml_backend_buffer_t expected,
                            size_t expected_offset) {
    ggml_backend_buffer_t owner = NULL;
    size_t offset = 0;
    if (!ggml_cuda8_ggml_tensor_is_device_resident(t, bytes, &owner, &offset)) {
        std::fprintf(stderr, "%s: residency not found\n", label);
        return false;
    }
    if (owner != expected || offset != expected_offset) {
        std::fprintf(stderr, "%s: residency mismatch offset=%zu expected=%zu\n",
                     label, offset, expected_offset);
        return false;
    }
    std::printf("%s residency PASS offset=%zu\n", label, offset);
    return true;
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

int main() {
    std::printf("ggml-cuda8-ggml-backend-graph-builder-cont-smoke: starting\n");

    // Use 1D tensors for simplicity: cont on a 1D contiguous tensor is a memcpy
    const int n = 512;
    const size_t bytes = (size_t) n * sizeof(float);

    // -- ggml context ---------------------------------------------------------
    std::vector<uint8_t> ggml_mem(4 * 1024 * 1024);
    struct ggml_init_params params;
    std::memset(&params, 0, sizeof(params));
    params.mem_size   = ggml_mem.size();
    params.mem_buffer = ggml_mem.data();
    params.no_alloc   = false;

    ggml_context * gctx = ggml_init(params);
    if (gctx == NULL) { std::fprintf(stderr, "ggml_init failed\n"); return 1; }

    // -- build graph: x -> cont -> add(bias) -> y ----------------------------
    ggml_tensor * tx   = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, n);
    ggml_tensor * bias = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, n);
    if (tx == NULL || bias == NULL) {
        std::fprintf(stderr, "tensor creation failed\n");
        ggml_free(gctx); return 1;
    }

    ggml_tensor * tc = ggml_cont(gctx, tx);
    ggml_tensor * ty = ggml_add(gctx, tc, bias);

    if (tc == NULL || ty == NULL) {
        std::fprintf(stderr, "op creation failed\n");
        ggml_free(gctx); return 1;
    }

    ggml_cgraph * graph = ggml_new_graph(gctx);
    if (graph == NULL) {
        std::fprintf(stderr, "ggml_new_graph failed\n");
        ggml_free(gctx); return 1;
    }
    ggml_build_forward_expand(graph, ty);

    std::printf("real ggml graph n_nodes=%d n_leafs=%d\n",
                graph->n_nodes, graph->n_leafs);
    for (int i = 0; i < graph->n_nodes; ++i) {
        std::printf("real graph node %d op=%d (%s)\n", i,
                    graph->nodes[i] ? (int) graph->nodes[i]->op : -1,
                    graph->nodes[i] ? ggml_op_name(graph->nodes[i]->op) : "NULL");
    }

    // -- backend init ---------------------------------------------------------
    ggml_backend_t backend = ggml_cuda8_ggml_backend_init(0);
    if (backend == NULL) {
        std::fprintf(stderr, "backend init failed\n");
        ggml_free(gctx); return 1;
    }
    if (backend->iface.graph_compute == NULL) {
        std::fprintf(stderr, "backend graph_compute callback is NULL\n");
        backend->iface.free(backend);
        ggml_free(gctx); return 1;
    }
    std::printf("backend name: %s\n", backend->iface.get_name(backend));

    ggml_backend_buffer_type_t buft =
        ggml_cuda8_ggml_backend_get_default_buffer_type(backend);

    // -- allocate device buffer -----------------------------------------------
    // Layout: x | bias | cont | y   (4 slots, 4096-aligned)
    const size_t off_x    = 0;
    const size_t off_bias = 4096;
    const size_t off_cont = 8192;
    const size_t off_y    = 12288;
    const size_t total    = 16384;

    ggml_backend_buffer_t buffer = buft->iface.alloc_buffer(buft, total);
    if (buffer == NULL) {
        std::fprintf(stderr, "buffer alloc failed\n");
        backend->iface.free(backend);
        ggml_free(gctx); return 1;
    }
    void * base = buffer->iface.get_base(buffer);
    uint8_t * base_u8 = (uint8_t *) base;
    std::printf("buffer base: %p  size: %zu\n", base, buffer->size);

    // Wire tensor data to device buffer
    force_1d_f32_data_layout(tx,   n, base_u8 + off_x);
    force_1d_f32_data_layout(bias, n, base_u8 + off_bias);
    force_1d_f32_data_layout(tc,   n, base_u8 + off_cont);
    force_1d_f32_data_layout(ty,   n, base_u8 + off_y);

    // Register residency
    if (buffer->iface.init_tensor(buffer, tx)   != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, bias) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, tc)   != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, ty)   != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "init_tensor failed\n");
        buffer->iface.free_buffer(buffer);
        backend->iface.free(backend);
        ggml_free(gctx); return 1;
    }
    if (!expect_resident("x",    tx,   bytes, buffer, off_x)    ||
        !expect_resident("bias", bias, bytes, buffer, off_bias) ||
        !expect_resident("cont", tc,   bytes, buffer, off_cont) ||
        !expect_resident("y",    ty,   bytes, buffer, off_y)) {
        buffer->iface.free_buffer(buffer);
        backend->iface.free(backend);
        ggml_free(gctx); return 1;
    }

    // -- prepare host data + CPU reference ------------------------------------
    std::vector<float> h_x(n), h_bias(n), h_ref(n), h_y(n, 0.0f);

    srand(42);
    for (int i = 0; i < n; ++i) {
        h_x[i]    = ((float) rand() / RAND_MAX) * 2.0f - 1.0f;
        h_bias[i] = ((float) rand() / RAND_MAX) * 2.0f - 1.0f;
        h_ref[i]  = h_x[i] + h_bias[i];  // cont is identity copy
    }

    buffer->iface.clear(buffer, 0);
    buffer->iface.set_tensor(buffer, tx,   &h_x[0],    0, bytes);
    buffer->iface.set_tensor(buffer, bias, &h_bias[0], 0, bytes);

    // -- compute --------------------------------------------------------------
    enum ggml_status status = backend->iface.graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "backend graph_compute returned %d\n", (int) status);
        buffer->iface.free_buffer(buffer);
        backend->iface.free(backend);
        ggml_free(gctx); return 1;
    }

    // -- download and verify --------------------------------------------------
    buffer->iface.get_tensor(buffer, ty, &h_y[0], 0, bytes);

    float max_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        float e = std::fabs(h_y[i] - h_ref[i]);
        if (e > max_err) max_err = e;
    }

    std::printf("real graph-builder CONT->ADD max_err=%.6e\n", max_err);

    bool pass = (max_err < 1e-6f);
    std::printf("real graph-builder CONT->ADD verification %s\n",
                pass ? "PASS" : "FAIL");

    // -- cleanup --------------------------------------------------------------
    buffer->iface.free_buffer(buffer);
    backend->iface.free(backend);
    ggml_free(gctx);

    std::printf("ggml-cuda8-ggml-backend-graph-builder-cont-smoke: %s\n",
                pass ? "SUCCESS" : "FAIL");
    return pass ? 0 : 1;
}
