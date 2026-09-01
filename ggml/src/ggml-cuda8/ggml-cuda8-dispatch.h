// ggml/src/ggml-cuda8/ggml-cuda8-dispatch.h
//
// Dispatcher:
//   MUL_MAT Q8_0xF32 vec
//   CPY_F32
//   ADD_F32
//   scalar ADD/MUL
//   REDUCE_SUM_ROWS_F32
//   REDUCE_MAX_ROWS_F32

#ifndef GGML_CUDA8_DISPATCH_H
#define GGML_CUDA8_DISPATCH_H

#include "ggml-cuda8-context.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

enum ggml_cuda8_op_id {
    GGML_CUDA8_OP_NONE = 0,

    GGML_CUDA8_OP_MUL_MAT_Q8_0_F32_VEC = 1,

    GGML_CUDA8_OP_CPY_F32,
    GGML_CUDA8_OP_ADD_F32,

    GGML_CUDA8_OP_ADD_SCALAR_F32,
    GGML_CUDA8_OP_MUL_SCALAR_F32,

    GGML_CUDA8_OP_REDUCE_SUM_ROWS_F32,
    GGML_CUDA8_OP_REDUCE_MAX_ROWS_F32,
    GGML_CUDA8_OP_SOFTMAX_ROWS_F32,
	GGML_CUDA8_OP_SOFTMAX_EXT_F32,

    GGML_CUDA8_OP_RMS_NORM_F32,

    GGML_CUDA8_OP_MUL_F32,
    GGML_CUDA8_OP_MUL_BROADCAST_F32,

    GGML_CUDA8_OP_ROPE_F32,
	    // G42: batched, broadcast-aware F32xF32 MUL_MAT - the attention
    // matmuls (K.Q, probs.V). Not to be confused with the quantized-weight
    // MUL_MAT paths (Q8_0/Q4_K/Q6_K) above.
    GGML_CUDA8_OP_MUL_MAT_F32_F32,
	
    GGML_CUDA8_OP_CONT_F32,

    GGML_CUDA8_OP_DIAG_MASK_INF_F32,

    GGML_CUDA8_OP_GET_ROWS_F32,

    GGML_CUDA8_OP_MUL_MAT_Q4_K_F32,
    GGML_CUDA8_OP_MUL_MAT_Q6_K_F32,
    GGML_CUDA8_OP_GET_ROWS_Q4_K,
    GGML_CUDA8_OP_GET_ROWS_Q6_K,

    // G40: SwiGLU gated activation - the FFN. dst = silu(gate) * up.
    GGML_CUDA8_OP_SWIGLU_F32,

    // G43: SET_ROWS scatter - writes the KV cache. dst[idx[i]] = src0[i].
    GGML_CUDA8_OP_SET_ROWS_F32,
};

const char * ggml_cuda8_op_name(int op_id);

int ggml_cuda8_dispatch_supported(
    const struct ggml_cuda8_context * ctx,
    int op_id,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst
);

int ggml_cuda8_dispatch_execute(
    struct ggml_cuda8_context * ctx,
    int op_id,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct ggml_tensor * dst
);

#ifdef __cplusplus
}
#endif

#endif // GGML_CUDA8_DISPATCH_H
