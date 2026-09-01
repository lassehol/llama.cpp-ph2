// ggml/src/ggml-cuda8/ggml-cuda8-ggml-buffer.cpp
//
// G11A-1: minimal ggml_backend_buffer_t wrapper for CUDA8.
//
// Transitional wrapper:
// - avoids ggml_backend_buffer_init()
// - avoids linking full ggml-base
// - directly allocates ggml_backend_buffer using ggml-backend-impl.h layout

#include "ggml-cuda8-ggml-buffer.h"

#include "ggml-backend-impl.h"
#include "ggml-cuda8-backend-buffer.h"
#include "buffer.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>

struct ggml_cuda8_ggml_buffer_context {
    ggml_cuda8_backend_buffer * cuda8_buf;
};

static ggml_cuda8_ggml_buffer_context * ctx_from_buffer(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) return NULL;
    return (ggml_cuda8_ggml_buffer_context *) buffer->context;
}

static size_t tensor_offset(
    ggml_backend_buffer_t buffer,
    const struct ggml_tensor * tensor,
    size_t offset
) {
    ggml_cuda8_ggml_buffer_context * ctx = ctx_from_buffer(buffer);

    if (ctx == NULL || ctx->cuda8_buf == NULL || tensor == NULL || tensor->data == NULL) {
        return (size_t) -1;
    }

    const uint8_t * base =
        (const uint8_t *) ggml_cuda8_backend_buffer_get_base_const(ctx->cuda8_buf);

    const uint8_t * ptr =
        (const uint8_t *) tensor->data;

    if (base == NULL || ptr < base) {
        return (size_t) -1;
    }

    return (size_t) (ptr - base) + offset;
}

// ---------------------------------------------------------------------
// Buffer iface
// ---------------------------------------------------------------------

static void cuda8_free_buffer(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) return;

    ggml_cuda8_ggml_unregister_buffer(buffer);

    ggml_cuda8_ggml_buffer_context * ctx = ctx_from_buffer(buffer);

    if (ctx != NULL) {
        if (ctx->cuda8_buf != NULL) {
            ggml_cuda8_backend_buffer_free(ctx->cuda8_buf);
            ctx->cuda8_buf = NULL;
        }

        std::free(ctx);
    }

    // G39: do NOT free `buffer` here.
    //
    // ggml_backend_buffer_free() calls this hook and then does `delete buffer`
    // itself (ggml-backend.cpp). Freeing it here made that a double free, which
    // glibc reports as "double free or corruption (fasttop)" during teardown.
    // The struct also belongs to ggml, which allocates it with `new` in
    // ggml_backend_buffer_init() - so releasing it with std::free() was a
    // mismatched deallocation on top of the double free.
    //
    // This hook owns the device allocation and our own context, nothing else.
}

static void * cuda8_get_base(ggml_backend_buffer_t buffer) {
    ggml_cuda8_ggml_buffer_context * ctx = ctx_from_buffer(buffer);

    if (ctx == NULL || ctx->cuda8_buf == NULL) {
        return NULL;
    }

    return ggml_cuda8_backend_buffer_get_base(ctx->cuda8_buf);
}

static enum ggml_status cuda8_init_tensor(
    ggml_backend_buffer_t buffer,
    struct ggml_tensor * tensor
) {
    if (buffer == NULL || tensor == NULL) {
        return GGML_STATUS_FAILED;
    }

    if (tensor->data == NULL) {
        tensor->data = cuda8_get_base(buffer);
    }

    return GGML_STATUS_SUCCESS;
}

static void cuda8_memset_tensor(
    ggml_backend_buffer_t buffer,
    struct ggml_tensor * tensor,
    uint8_t value,
    size_t offset,
    size_t size
) {
    ggml_cuda8_ggml_buffer_context * ctx = ctx_from_buffer(buffer);

    if (ctx == NULL || ctx->cuda8_buf == NULL || tensor == NULL) return;

    const size_t abs_off = tensor_offset(buffer, tensor, offset);
    if (abs_off == (size_t) -1) return;

    if (abs_off + size > ggml_cuda8_backend_buffer_get_size(ctx->cuda8_buf)) return;

    uint8_t * ptr =
        (uint8_t *) ggml_cuda8_backend_buffer_get_base(ctx->cuda8_buf) + abs_off;

    ggml_cuda8_buffer_memset(ptr, (int) value, size);
}

static void cuda8_set_tensor(
    ggml_backend_buffer_t buffer,
    struct ggml_tensor * tensor,
    const void * data,
    size_t offset,
    size_t size
) {
    ggml_cuda8_ggml_buffer_context * ctx = ctx_from_buffer(buffer);

    if (ctx == NULL || ctx->cuda8_buf == NULL || tensor == NULL || data == NULL) return;

    const size_t abs_off = tensor_offset(buffer, tensor, offset);
    if (abs_off == (size_t) -1) return;

    ggml_cuda8_backend_buffer_upload(ctx->cuda8_buf, abs_off, data, size);
}

static void cuda8_get_tensor(
    ggml_backend_buffer_t buffer,
    const struct ggml_tensor * tensor,
    void * data,
    size_t offset,
    size_t size
) {
    ggml_cuda8_ggml_buffer_context * ctx = ctx_from_buffer(buffer);

    if (ctx == NULL || ctx->cuda8_buf == NULL || tensor == NULL || data == NULL) return;

    const size_t abs_off = tensor_offset(buffer, tensor, offset);
    if (abs_off == (size_t) -1) return;

    ggml_cuda8_backend_buffer_download(ctx->cuda8_buf, abs_off, data, size);
}

static bool cuda8_cpy_tensor(
    ggml_backend_buffer_t buffer,
    const struct ggml_tensor * src,
    struct ggml_tensor * dst
) {
    (void) buffer;
    (void) src;
    (void) dst;
    return false;
}

static void cuda8_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_cuda8_ggml_buffer_context * ctx = ctx_from_buffer(buffer);

    if (ctx == NULL || ctx->cuda8_buf == NULL) return;

    ggml_cuda8_backend_buffer_clear(ctx->cuda8_buf, (int) value);
}

static struct ggml_backend_buffer_i cuda8_buffer_i = {
    cuda8_free_buffer,
    cuda8_get_base,
    cuda8_init_tensor,
    cuda8_memset_tensor,
    cuda8_set_tensor,
    cuda8_get_tensor,
    NULL, // set_tensor_2d
    NULL, // get_tensor_2d
    cuda8_cpy_tensor,
    cuda8_clear,
    NULL  // reset
};

// ---------------------------------------------------------------------
// Buffer type iface
// ---------------------------------------------------------------------

static const char * cuda8_buft_get_name(ggml_backend_buffer_type_t buft) {
    (void) buft;
    return "CUDA8";
}

static size_t cuda8_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    (void) buft;
    return ggml_cuda8_backend_buffer_alignment();
}

static size_t cuda8_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    (void) buft;
    return (size_t) -1;
}

static bool cuda8_buft_is_host(ggml_backend_buffer_type_t buft) {
    (void) buft;
    return false;
}

static ggml_backend_buffer_t cuda8_buft_alloc_buffer(
    ggml_backend_buffer_type_t buft,
    size_t size
) {
    if (size == 0) {
        std::fprintf(stderr, "ggml-cuda8/ggml-buffer: zero-size allocation\n");
        return NULL;
    }

    ggml_cuda8_backend_buffer * raw = NULL;

    if (ggml_cuda8_backend_buffer_alloc(0, size, &raw) != 0) {
        return NULL;
    }

    ggml_cuda8_ggml_buffer_context * ctx =
        (ggml_cuda8_ggml_buffer_context *) std::malloc(sizeof(ggml_cuda8_ggml_buffer_context));

    if (ctx == NULL) {
        ggml_cuda8_backend_buffer_free(raw);
        return NULL;
    }

    ctx->cuda8_buf = raw;

    // G39: let ggml allocate and own the buffer struct.
    //
    // Previously this malloc'd the struct by hand. ggml_backend_buffer_free()
    // deletes it, so a hand-rolled malloc was a mismatched deallocation, and
    // cuda8_free_buffer() freeing it too made it a double free. Going through
    // ggml_backend_buffer_init() keeps ownership in one place and picks up any
    // future fields added to the struct.
    ggml_backend_buffer_t buffer =
        ggml_backend_buffer_init(buft, cuda8_buffer_i, ctx, size);

    if (buffer == NULL) {
        ggml_cuda8_backend_buffer_free(raw);
        std::free(ctx);
        return NULL;
    }

    if (!ggml_cuda8_ggml_register_buffer(buffer)) {
        // Full teardown: our hook releases the device memory and ctx, then
        // ggml deletes the struct.
        ggml_backend_buffer_free(buffer);
        return NULL;
    }

    return buffer;
}

static struct ggml_backend_buffer_type_i cuda8_buft_i = {
    cuda8_buft_get_name,
    cuda8_buft_alloc_buffer,
    cuda8_buft_get_alignment,
    cuda8_buft_get_max_size,
    NULL, // get_alloc_size
    cuda8_buft_is_host
};

static struct ggml_backend_buffer_type cuda8_buft = {
    cuda8_buft_i,
    NULL,
    NULL
};

extern "C" ggml_backend_buffer_type_t ggml_cuda8_ggml_buffer_type(void) {
    return &cuda8_buft;
}

extern "C" ggml_backend_buffer_t ggml_cuda8_ggml_buffer_alloc(size_t size) {
    return cuda8_buft_alloc_buffer(ggml_cuda8_ggml_buffer_type(), size);
}

extern "C" int ggml_cuda8_ggml_buffer_is_cuda8(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) return 0;
    return buffer->buft == ggml_cuda8_ggml_buffer_type();
}

// G36: Set the device pointer on the buffer type (called from backend-reg.cpp)
extern "C" void ggml_cuda8_ggml_buffer_type_set_device(ggml_backend_dev_t dev) {
    cuda8_buft.device = dev;
}
