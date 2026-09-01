// ggml/src/ggml-cuda8/ggml-cuda8-add.cpp
//
// G9B-2: F32 vector ADD dispatcher helper
//
// G54: device-resident ADD_F32 fast path.
//   Adds the same residency-aware branch that ggml-cuda8-softmax.cpp
//   (G11A-4E) and ggml-cuda8-scalar.cpp (G11A-4D) already use. If src0,
//   src1 and dst are all registered CUDA8 ggml_backend_buffer_t residents
//   - which is the case for every ADD node under graph_compute - the
//   kernel launches directly on the device pointers with no host<->device
//   staging. The host-staging path is kept for the standalone add-smoke
//   test, which drives this function with plain host-backed tensors.
//
//   Motivation: ADD_F32 is one of the most frequently dispatched ops
//   (every residual/bias-add node). On a real Qwen3-0.6B generation it was
//   ~1900 dispatches, each doing 3x cudaMalloc + 3x cudaMemcpy + 3x
//   cudaFree while every other hot-path op (RMS_NORM, ROPE, MUL, the
//   K-quant matmuls, ...) already ran device-resident. This was the last
//   host-staging op left in that model's hot path.
#include "ggml-cuda8-add.h"
#include "ggml-cuda8-backend-buffer.h"
#include "ggml-cuda8-ggml-buffer.h"
#include <cstdio>

extern "C" int ggml_cuda8_add_f32_launch(
    const float * a, const float * b, float * c, int n
);

static bool is_contig_f32_1d(const ggml_tensor * t) {
    return t && t->type == GGML_TYPE_F32 && t->ne[1] == 1 && t->nb[0] == sizeof(float);
}

int ggml_cuda8_supported_add_f32(
    const ggml_cuda8_context * ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    const ggml_tensor * dst
) {
    (void) ctx;
    if (!is_contig_f32_1d(src0)) return 0;
    if (!is_contig_f32_1d(src1)) return 0;
    if (!is_contig_f32_1d(dst))  return 0;
    if (src0->ne[0] != src1->ne[0]) return 0;
    if (src0->ne[0] != dst->ne[0])  return 0;
    return 1;
}

// G54: residency probe, identical in spirit to the helper of the same name
// in ggml-cuda8-softmax.cpp / ggml-cuda8-scalar.cpp.
static int tensor_is_cuda8_resident(const ggml_tensor * t, size_t bytes) {
    ggml_backend_buffer_t owner = NULL;
    size_t offset = 0;
    return ggml_cuda8_ggml_tensor_is_device_resident(t, bytes, &owner, &offset);
}

// G54: fast path - all three tensors already live in CUDA8 device buffers,
// so launch straight on their ->data pointers. This is exactly what every
// other hot-path op (RMS_NORM, ROPE, MUL, K-quant matmuls, ...) already
// does; ADD just never got the branch.
static int exec_add_f32_device_resident(
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst,
    int n
) {
    return ggml_cuda8_add_f32_launch(
        (const float *) src0->data,
        (const float *) src1->data,
        (float *)       dst->data,
        n
    );
}

// Host-staging fallback - unchanged from the pre-G54 body (including the
// G51 free-on-alloc-failure cleanup). Used by the standalone add-smoke
// test, which passes plain host-backed tensors that are not in any CUDA8
// buffer.
static int exec_add_f32_host_staging(
    ggml_cuda8_context * ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst,
    int n,
    size_t bytes
) {
    ggml_cuda8_backend_buffer * b0 = NULL;
    ggml_cuda8_backend_buffer * b1 = NULL;
    ggml_cuda8_backend_buffer * bd = NULL;

    // G51 cleanup: free previously-allocated buffers on a later allocation
    // failure instead of leaking them.
    if (ggml_cuda8_context_alloc_buffer(ctx, bytes, &b0) != 0) {
        return -1;
    }
    if (ggml_cuda8_context_alloc_buffer(ctx, bytes, &b1) != 0) {
        ggml_cuda8_backend_buffer_free(b0);
        return -1;
    }
    if (ggml_cuda8_context_alloc_buffer(ctx, bytes, &bd) != 0) {
        ggml_cuda8_backend_buffer_free(b0);
        ggml_cuda8_backend_buffer_free(b1);
        return -1;
    }

    int rc = 0;
    if (ggml_cuda8_backend_buffer_upload(b0, 0, src0->data, bytes) != 0) rc = -1;
    if (rc == 0 && ggml_cuda8_backend_buffer_upload(b1, 0, src1->data, bytes) != 0) rc = -1;
    if (rc == 0) {
        rc = ggml_cuda8_add_f32_launch(
            (const float *) ggml_cuda8_backend_buffer_get_base_const(b0),
            (const float *) ggml_cuda8_backend_buffer_get_base_const(b1),
            (float *)       ggml_cuda8_backend_buffer_get_base(bd),
            n
        );
    }
    if (rc == 0) {
        if (ggml_cuda8_backend_buffer_download(bd, 0, dst->data, bytes) != 0) rc = -1;
    }

    ggml_cuda8_backend_buffer_free(b0);
    ggml_cuda8_backend_buffer_free(b1);
    ggml_cuda8_backend_buffer_free(bd);
    return rc;
}

int ggml_cuda8_exec_add_f32(
    ggml_cuda8_context * ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst
) {
    if (!ggml_cuda8_supported_add_f32(ctx, src0, src1, dst)) {
        std::fprintf(stderr, "ggml-cuda8/add: unsupported layout\n");
        return -1;
    }

    const int n = (int) src0->ne[0];
    const size_t bytes = (size_t) n * sizeof(float);

    // G54: take the device-resident fast path only when ALL THREE tensors
    // are CUDA8-resident (the graph_compute case). If any is host-backed
    // (the add-smoke case), fall back to host staging, which uploads each
    // source and downloads the result and so is robust to host-backed or
    // even mixed inputs. This matches the branch structure of
    // ggml-cuda8-softmax.cpp, except softmax rejects the mixed case
    // outright; here the host-staging fallback handles it correctly, so no
    // rejection is needed.
    const int src0_dev = tensor_is_cuda8_resident(src0, bytes);
    const int src1_dev = tensor_is_cuda8_resident(src1, bytes);
    const int dst_dev  = tensor_is_cuda8_resident(dst,  bytes);

    if (src0_dev && src1_dev && dst_dev) {
        return exec_add_f32_device_resident(src0, src1, dst, n);
    }
    return exec_add_f32_host_staging(ctx, src0, src1, dst, n, bytes);
}
