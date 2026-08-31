// ggml/src/ggml-cuda8/ggml-cuda8-ggml-backend.h
//
// G12A/G13A/G14B/G15A: minimal ggml_backend_t wrapper/probe for CUDA8.

#ifndef GGML_CUDA8_GGML_BACKEND_H
#define GGML_CUDA8_GGML_BACKEND_H

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda8-dispatch.h"

#include <string.h>

// G37: SOFT_MAX support predicate, shared by supports_op and graph_compute.
//
// The CUDA8 SOFTMAX_ROWS_F32 kernel computes a plain row-wise softmax: it takes no
// mask, never reads op_params, and has no attention-sink support. But ggml folds
// four different things into GGML_OP_SOFT_MAX via ggml_soft_max_ext():
//
//     src[1]       - the attention mask
//     src[2]       - attention sinks (ggml_soft_max_add_sinks)
//     params[0]    - scale   (1/sqrt(head_dim) in real attention)
//     params[1]    - max_bias (ALiBi slope)
//
// Claiming such a node would not fail loudly - it would silently compute an
// unmasked, unscaled softmax and return plausible-looking wrong attention
// weights. So anything beyond the plain ggml_soft_max() case must be refused
// here and left to the CPU backend.
//
// Returns 1 only for: no mask, no sinks, scale == 1.0f, max_bias == 0.0f.
static inline int ggml_cuda8_soft_max_is_plain(const struct ggml_tensor * op) {
    float params[2] = { 1.0f, 0.0f };  // { scale, max_bias }

    if (op == NULL) {
        return 0;
    }
    if (op->src[1] != NULL) {  // mask
        return 0;
    }
    if (op->src[2] != NULL) {  // attention sinks
        return 0;
    }

    memcpy(params, op->op_params, sizeof(params));

    if (params[0] != 1.0f) {  // scale
        return 0;
    }
    if (params[1] != 0.0f) {  // max_bias (ALiBi)
        return 0;
    }

    return 1;
}

#ifdef __cplusplus
extern "C" {
#endif

ggml_backend_t ggml_cuda8_ggml_backend_init(int device);
int ggml_cuda8_ggml_backend_is_cuda8(ggml_backend_t backend);
ggml_backend_buffer_type_t ggml_cuda8_ggml_backend_get_default_buffer_type(ggml_backend_t backend);

// G13A/G13B: compute-shaped helper used by backend-owned smoke graphs.
// This helper remains useful as the stable single-op dispatch path.
int ggml_cuda8_ggml_backend_dispatch_op(
    ggml_backend_t backend,
    ggml_cuda8_context * ctx,
    int op,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst
);

#ifdef __cplusplus
}
#endif

#endif // GGML_CUDA8_GGML_BACKEND_H
