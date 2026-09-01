// ggml/src/ggml-cuda8/ggml-cuda8-softmax-ext.cpp
//
// G41: masked / scaled / ALiBi-biased row-wise softmax dispatcher helper.
//
// Unlike the plain SOFTMAX_ROWS_F32 path (ggml-cuda8-softmax.cpp), which
// still supports host-staging for src0/dst not yet CUDA8-resident, this exec
// function operates directly on ->data pointers with no host<->device
// staging - the same convention already used by ROPE, DIAG_MASK_INF,
// GET_ROWS, SWIGLU and SET_ROWS: these are all "not flattened, tensors
// assumed device-resident" ops introduced after the residency-staging era of
// G11A-4*, on the assumption that ggml's scheduler only ever hands this
// backend tensors that are already in its own buffer type.
#include "ggml-cuda8-softmax-ext.h"
#include <cstring>
#include <cstdio>

extern "C" int ggml_cuda8_softmax_ext_f32_launch(
        const float * src,
        const float * mask,
        float * dst,
        int ne00, int ne01, int ne02, int ne03,
        size_t nb01, size_t nb02, size_t nb03,
        size_t dst_nb1, size_t dst_nb2, size_t dst_nb3,
        int mask_ne1, int mask_ne2, int mask_ne3,
        size_t mask_nb1, size_t mask_nb2, size_t mask_nb3,
        float scale, float max_bias);

int ggml_cuda8_supported_softmax_ext_f32(
        const struct ggml_cuda8_context * ctx,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,   // mask, may be NULL
        const struct ggml_tensor * dst
) {
    (void) ctx;
    if (src0 == NULL || dst == NULL) return 0;
    if (src0->data == NULL || dst->data == NULL) return 0;
    if (src0->type != GGML_TYPE_F32) return 0;
    if (dst->type  != GGML_TYPE_F32) return 0;
    // Rows must be contiguous; ne1/ne2/ne3 strides are taken as given
    // (not assumed packed) rather than flattened, the same contract SWIGLU
    // and DIAG_MASK_INF already rely on.
    if (src0->nb[0] != sizeof(float)) return 0;
    if (dst->nb[0]  != sizeof(float)) return 0;
    if (src0->ne[0] != dst->ne[0]) return 0;
    if (src0->ne[1] != dst->ne[1]) return 0;
    if (src0->ne[2] != dst->ne[2]) return 0;
    if (src0->ne[3] != dst->ne[3]) return 0;
    if (src1 != NULL) {
        if (src1->data == NULL) return 0;
        // F16 mask deferred - ties into the general F16 storage work
        // already tracked as G49, not implemented here. Refused, not
        // silently truncated/reinterpreted.
        if (src1->type != GGML_TYPE_F32) return 0;
        if (src1->nb[0] != sizeof(float)) return 0;
        if (src1->ne[0] != src0->ne[0]) return 0;
        // No broadcast in the query/token dimension - every row needs its
        // own mask row. ne2/ne3 may be 1 (broadcast across all heads/batch,
        // the common case) or match src0 exactly.
        if (src1->ne[1] != src0->ne[1]) return 0;
        if (src1->ne[2] != 1 && src1->ne[2] != src0->ne[2]) return 0;
        if (src1->ne[3] != 1 && src1->ne[3] != src0->ne[3]) return 0;
    }
    return 1;
}

int ggml_cuda8_exec_softmax_ext_f32(
        struct ggml_cuda8_context * ctx,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,   // mask, may be NULL
        struct ggml_tensor * dst
) {
    (void) ctx;
    if (!ggml_cuda8_supported_softmax_ext_f32(ctx, src0, src1, dst)) {
        std::fprintf(stderr, "ggml-cuda8/softmax-ext: unsupported layout\n");
        return -1;
    }
    // op_params[0] = scale, op_params[1] = max_bias - the same layout
    // ggml_soft_max_impl() stamps, and the same fixture trap G37 flagged: a
    // memset-zeroed op_params reads as scale=0.0f, not a plain softmax. Any
    // caller building this node by hand must stamp params explicitly.
    float params[2] = { 1.0f, 0.0f };
    std::memcpy(params, dst->op_params, sizeof(params));
    const float scale    = params[0];
    const float max_bias = params[1];

    const int ne00 = (int) src0->ne[0];
    const int ne01 = (int) src0->ne[1];
    const int ne02 = (int) src0->ne[2];
    const int ne03 = (int) src0->ne[3];

    const float * mask_ptr = src1 ? (const float *) src1->data : NULL;
    int    mask_ne1 = 0, mask_ne2 = 0, mask_ne3 = 0;
    size_t mask_nb1 = 0, mask_nb2 = 0, mask_nb3 = 0;
    if (src1 != NULL) {
        mask_ne1 = (int) src1->ne[1];
        mask_ne2 = (int) src1->ne[2];
        mask_ne3 = (int) src1->ne[3];
        mask_nb1 = src1->nb[1];
        mask_nb2 = src1->nb[2];
        mask_nb3 = src1->nb[3];
    }

    return ggml_cuda8_softmax_ext_f32_launch(
        (const float *) src0->data,
        mask_ptr,
        (float *) dst->data,
        ne00, ne01, ne02, ne03,
        src0->nb[1], src0->nb[2], src0->nb[3],
        dst->nb[1],  dst->nb[2],  dst->nb[3],
        mask_ne1, mask_ne2, mask_ne3,
        mask_nb1, mask_nb2, mask_nb3,
        scale, max_bias
    );
}
