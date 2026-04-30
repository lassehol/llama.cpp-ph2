// ggml/src/ggml-cuda8/ggml-cuda8-ggml-buffer-registry.cpp
//
// G11A-4A: lightweight CUDA8 GGML buffer residency registry.
//
// Prototype-only:
//   - static process-local registry
//   - no mutex/threading yet
//   - sufficient for smoke tests and initial dispatcher residency probes

#include "ggml-cuda8-ggml-buffer.h"

#include "ggml-backend-impl.h"

#include <cstdio>
#include <stdint.h>
#include <vector>

struct ggml_cuda8_registered_buffer {
    ggml_backend_buffer_t buffer;
    void * base;
    size_t size;
};

static std::vector<ggml_cuda8_registered_buffer> & registry(void) {
    static std::vector<ggml_cuda8_registered_buffer> r;
    return r;
}

extern "C" int ggml_cuda8_ggml_buffer_contains_ptr(
    ggml_backend_buffer_t buffer,
    const void * ptr,
    size_t size
) {
    if (buffer == NULL || ptr == NULL) {
        return 0;
    }

    if (!ggml_cuda8_ggml_buffer_is_cuda8(buffer)) {
        return 0;
    }

    void * base_ptr = buffer->iface.get_base(buffer);

    if (base_ptr == NULL) {
        return 0;
    }

    const uintptr_t base = (uintptr_t) base_ptr;
    const uintptr_t p    = (uintptr_t) ptr;

    if (p < base) {
        return 0;
    }

    const size_t off = (size_t) (p - base);

    if (off > buffer->size) {
        return 0;
    }

    if (size > buffer->size - off) {
        return 0;
    }

    return 1;
}

extern "C" int ggml_cuda8_ggml_register_buffer(
    ggml_backend_buffer_t buffer
) {
    if (buffer == NULL) {
        return 0;
    }

    if (!ggml_cuda8_ggml_buffer_is_cuda8(buffer)) {
        return 0;
    }

    void * base = buffer->iface.get_base(buffer);

    if (base == NULL) {
        return 0;
    }

    std::vector<ggml_cuda8_registered_buffer> & r = registry();

    for (size_t i = 0; i < r.size(); ++i) {
        if (r[i].buffer == buffer) {
            r[i].base = base;
            r[i].size = buffer->size;
            return 1;
        }
    }

    ggml_cuda8_registered_buffer rec;
    rec.buffer = buffer;
    rec.base   = base;
    rec.size   = buffer->size;

    r.push_back(rec);
    return 1;
}

extern "C" int ggml_cuda8_ggml_unregister_buffer(
    ggml_backend_buffer_t buffer
) {
    if (buffer == NULL) {
        return 0;
    }

    std::vector<ggml_cuda8_registered_buffer> & r = registry();

    for (size_t i = 0; i < r.size(); ++i) {
        if (r[i].buffer == buffer) {
            r.erase(r.begin() + i);
            return 1;
        }
    }

    return 0;
}

extern "C" int ggml_cuda8_ggml_find_buffer_for_ptr(
    const void * ptr,
    size_t size,
    ggml_backend_buffer_t * out_buffer,
    size_t * out_offset
) {
    if (out_buffer != NULL) {
        *out_buffer = NULL;
    }

    if (out_offset != NULL) {
        *out_offset = 0;
    }

    if (ptr == NULL) {
        return 0;
    }

    std::vector<ggml_cuda8_registered_buffer> & r = registry();

    const uintptr_t p = (uintptr_t) ptr;

    for (size_t i = 0; i < r.size(); ++i) {
        const uintptr_t base = (uintptr_t) r[i].base;

        if (p < base) {
            continue;
        }

        const size_t off = (size_t) (p - base);

        if (off > r[i].size) {
            continue;
        }

        if (size > r[i].size - off) {
            continue;
        }

        if (out_buffer != NULL) {
            *out_buffer = r[i].buffer;
        }

        if (out_offset != NULL) {
            *out_offset = off;
        }

        return 1;
    }

    return 0;
}

extern "C" int ggml_cuda8_ggml_tensor_is_device_resident(
    const struct ggml_tensor * tensor,
    size_t nbytes,
    ggml_backend_buffer_t * out_buffer,
    size_t * out_offset
) {
    if (tensor == NULL || tensor->data == NULL) {
        if (out_buffer != NULL) {
            *out_buffer = NULL;
        }

        if (out_offset != NULL) {
            *out_offset = 0;
        }

        return 0;
    }

    return ggml_cuda8_ggml_find_buffer_for_ptr(
        tensor->data,
        nbytes,
        out_buffer,
        out_offset
    );
}
