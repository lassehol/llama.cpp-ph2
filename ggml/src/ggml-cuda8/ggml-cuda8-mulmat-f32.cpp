// ggml/src/ggml-cuda8/ggml-cuda8-mulmat-f32.cpp
//
// G42: batched F32xF32 MUL_MAT dispatcher helper - see
// ggml-cuda8-mulmat-f32.h and ggml-cuda8-mulmat-f32.cu for the scope and
// kernel semantics.
//
// Like ROPE/DIAG_MASK_INF/SWIGLU/SET_ROWS, this exec function operates
// directly on ->data pointers with no host<->device staging - the ggml
// scheduler is assumed to only ever hand this backend tensors already
// resident in its own buffer type.
#include "ggml-cuda8-mulmat-f32.h"
#include <cstdio>

extern "C" int ggml_cuda8_mul_mat_f32_f32_launch(
    const float * src0,
    const float * src1,
    float * dst,
    int ne00,
    int ne01, int ne02, int ne03,
    int ne11, int ne12, int ne13,
    size_t nb01, size_t nb02, size_t nb03,
    size_t nb11, size_t nb12, size_t nb13,
    size_t nb1,  size_t nb2,  size_t nb3);

int ggml_cuda8_supported_mul_mat_f32_f32(
        const struct ggml_cuda8_context * ctx,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        const struct ggml_tensor * dst
) {
    (void) ctx;
    if (src0 == NULL || src1 == NULL || dst == NULL) return 0;
    if (src0->data == NULL || src1->data == NULL || dst->data == NULL) return 0;
    if (src0->type != GGML_TYPE_F32) return 0;
    if (src1->type != GGML_TYPE_F32) return 0;
    if (dst->type  != GGML_TYPE_F32) return 0;
    // Dim 0 (the reduction dimension) must be contiguous on src0/src1 - this
    // is the one dimension this kernel does not take an explicit stride for.
    // Dims 1-3 are NOT required to be packed/contiguous: the kernel reads
    // them via explicit byte strides, so permuted views (attention's common
    // case after reshape+permute, which reorders head/token/batch dims but
    // leaves dim 0 alone) are handled directly.
    if (src0->nb[0] != sizeof(float)) return 0;
    if (src1->nb[0] != sizeof(float)) return 0;
    if (dst->nb[0]  != sizeof(float)) return 0;
    // Shared reduction dimension.
    if (src0->ne[0] != src1->ne[0]) return 0;
    // GQA-style broadcast: src0's head/batch dims must divide evenly into
    // src1's. Guard ne02/ne03 > 0 defensively before the modulo (should be
    // unreachable for any real tensor, but avoids a div/mod-by-zero if one
    // ever is).
    if (src0->ne[2] <= 0 || src0->ne[3] <= 0) return 0;
    if (src1->ne[2] % src0->ne[2] != 0) return 0;
    if (src1->ne[3] % src0->ne[3] != 0) return 0;
    // dst shape: ne0 = src0's "n" dim, ne1/ne2/ne3 follow src1 (the batched
    // "query"-side operand), matching ggml_mul_mat()'s own convention.
    if (dst->ne[0] != src0->ne[1]) return 0;
    if (dst->ne[1] != src1->ne[1]) return 0;
    if (dst->ne[2] != src1->ne[2]) return 0;
    if (dst->ne[3] != src1->ne[3]) return 0;
    return 1;
}

int ggml_cuda8_exec_mul_mat_f32_f32(
        struct ggml_cuda8_context * ctx,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst
) {
    (void) ctx;
    if (!ggml_cuda8_supported_mul_mat_f32_f32(ctx, src0, src1, dst)) {
        std::fprintf(stderr, "ggml-cuda8/mulmat-f32: unsupported layout\n");
        return -1;
    }
    const int ne00 = (int) src0->ne[0];
    const int ne01 = (int) src0->ne[1];
    const int ne02 = (int) src0->ne[2];
    const int ne03 = (int) src0->ne[3];
    const int ne11 = (int) src1->ne[1];
    const int ne12 = (int) src1->ne[2];
    const int ne13 = (int) src1->ne[3];

    return ggml_cuda8_mul_mat_f32_f32_launch(
        (const float *) src0->data,
        (const float *) src1->data,
        (float *) dst->data,
        ne00, ne01, ne02, ne03, ne11, ne12, ne13,
        src0->nb[1], src0->nb[2], src0->nb[3],
        src1->nb[1], src1->nb[2], src1->nb[3],
        dst->nb[1],  dst->nb[2],  dst->nb[3]
    );
}
