// ggml/src/ggml-cuda8/ggml-cuda8-ggml-backend-graph-api-probe.cpp
//
// G14B smoke: compile-checked real ggml_backend_i graph callback stubs.
//
// Validates that CUDA8 backend populates actual graph callback fields matching
// this checkout's ggml-backend-impl.h signatures:
//   graph_plan_create
//   graph_plan_free
//   graph_plan_update
//   graph_plan_compute
//   graph_compute
//   graph_optimize
//
// This is intentionally not real graph execution yet. It is an ABI/API-shape
// probe that calls the safe no-op/success stubs with NULL graph/plan inputs.

#include "ggml-cuda8-ggml-backend.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <cstdio>

static int require_ptr(const char * name, const void * ptr) {
    if (ptr == NULL) {
        std::fprintf(stderr, "%s is NULL\n", name);
        return 1;
    }
    std::printf("%s present PASS\n", name);
    return 0;
}

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    std::printf("ggml-cuda8-ggml-backend-graph-api-probe: starting\n");

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

    int missing = 0;
    missing += require_ptr("graph_plan_create", (const void *) backend->iface.graph_plan_create);
    missing += require_ptr("graph_plan_free", (const void *) backend->iface.graph_plan_free);
    missing += require_ptr("graph_plan_update", (const void *) backend->iface.graph_plan_update);
    missing += require_ptr("graph_plan_compute", (const void *) backend->iface.graph_plan_compute);
    missing += require_ptr("graph_compute", (const void *) backend->iface.graph_compute);
    missing += require_ptr("graph_optimize", (const void *) backend->iface.graph_optimize);
    missing += require_ptr("synchronize", (const void *) backend->iface.synchronize);

    if (missing != 0) {
        backend->iface.free(backend);
        return 1;
    }

    std::printf("calling graph callback stubs with NULL graph/plan inputs\n");

    ggml_backend_graph_plan_t plan = backend->iface.graph_plan_create(backend, NULL);
    backend->iface.graph_plan_update(backend, plan, NULL);

    enum ggml_status st_plan = backend->iface.graph_plan_compute(backend, plan);
    if (st_plan != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "graph_plan_compute returned %d\n", (int) st_plan);
        backend->iface.graph_plan_free(backend, plan);
        backend->iface.free(backend);
        return 1;
    }

    enum ggml_status st_graph = backend->iface.graph_compute(backend, NULL);
    if (st_graph != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "graph_compute returned %d\n", (int) st_graph);
        backend->iface.graph_plan_free(backend, plan);
        backend->iface.free(backend);
        return 1;
    }

    backend->iface.graph_optimize(backend, NULL);
    backend->iface.synchronize(backend);
    backend->iface.graph_plan_free(backend, plan);

    backend->iface.free(backend);

    std::printf("ggml-cuda8-ggml-backend-graph-api-probe: SUCCESS\n");
    return 0;
}
