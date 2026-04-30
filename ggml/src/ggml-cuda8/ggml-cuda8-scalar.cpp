// ggml/src/ggml-cuda8/ggml-cuda8-scalar.cpp
//
// G9B-3: F32 scalar ADD/MUL dispatcher helpers.
//
// G11A-4D:
//   Adds device-resident scalar ADD/MUL path.
//   If src0 and dst are CUDA8-resident, launch directly on device pointers.
//   Scalar src1 may be host-backed or CUDA8-resident.
//   Host-backed src0/dst path remains supported.
//   Mixed host/device src0/dst is rejected.

#include "ggml-cuda8-scalar.h"
#include "ggml-cuda8-backend-buffer.h"
#include "ggml-cuda8-ggml-buffer.h"

#include <cstdio>
#include <cuda_runtime.h>

extern "C" int ggml_cuda8_add_scalar_f32_launch(
    const float * src,
    float scalar,
    float * dst,
    int n
);

extern "C" int ggml_cuda8_mul_scalar_f32_launch(
    const float * src,
    float scalar,
    float * dst,
    int n
);

static int tensor_is_contiguous_f32(const ggml_tensor * t) {
    if (t == NULL) return 0;
    if (t->type != GGML_TYPE_F32) return 0;
    if (t->data == NULL) return 0;
    if (t->nb[0] != sizeof(float)) return 0;

    size_t expected = sizeof(float);

    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (t->ne[i] <= 1) continue;

        if (t->nb[i] != expected) {
            return 0;
        }

        expected *= (size_t) t->ne[i];
    }

    return 1;
}

static int tensor_same_shape(const ggml_tensor * a, const ggml_tensor * b) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (a->ne[i] != b->ne[i]) {
            return 0;
        }
    }

    return 1;
}

static size_t tensor_nelements(const ggml_tensor * t) {
    size_t n = 1;

    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (t->ne[i] > 0) {
            n *= (size_t) t->ne[i];
        }
    }

    return n;
}

static int tensor_is_f32_scalar(const ggml_tensor * t) {
    if (t == NULL) return 0;
    if (t->type != GGML_TYPE_F32) return 0;
    if (t->data == NULL) return 0;

    if (t->ne[0] != 1) return 0;
    if (t->ne[1] != 1) return 0;
    if (t->ne[2] != 1) return 0;
    if (t->ne[3] != 1) return 0;

    if (t->nb[0] != sizeof(float)) return 0;

    return 1;
}

int ggml_cuda8_supported_scalar_f32(
    const ggml_cuda8_context * ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    const ggml_tensor * dst
) {
    (void) ctx;

    if (!tensor_is_contiguous_f32(src0)) return 0;
    if (!tensor_is_f32_scalar(src1)) return 0;
    if (!tensor_is_contiguous_f32(dst)) return 0;

    if (!tensor_same_shape(src0, dst)) return 0;

    return 1;
}

static int tensor_is_cuda8_resident(
    const ggml_tensor * t,
    size_t bytes
) {
    ggml_backend_buffer_t owner = NULL;
    size_t offset = 0;

    return ggml_cuda8_ggml_tensor_is_device_resident(
        t,
        bytes,
        &owner,
        &offset
    );
}

static int read_scalar_value(
    const ggml_tensor * src1,
    float * out_scalar
) {
    if (src1 == NULL || src1->data == NULL || out_scalar == NULL) {
        return -1;
    }

    if (tensor_is_cuda8_resident(src1, sizeof(float))) {
        cudaError_t err = cudaMemcpy(
            out_scalar,
            src1->data,
            sizeof(float),
            cudaMemcpyDeviceToHost
        );

        if (err != cudaSuccess) {
            std::fprintf(stderr,
                "ggml-cuda8/scalar: cudaMemcpy scalar D2H failed: %s (%d)\n",
                cudaGetErrorString(err),
                (int) err);
            return -1;
        }

        return 0;
    }

    *out_scalar = *((const float *) src1->data);
    return 0;
}

static int exec_scalar_f32_device_resident(
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst,
    int is_mul,
    int n
) {
    float scalar = 0.0f;

    if (read_scalar_value(src1, &scalar) != 0) {
        std::fprintf(stderr, "ggml-cuda8/scalar: failed to read scalar\n");
        return -1;
    }

    const float * d_src = (const float *) src0->data;
    float * d_dst = (float *) dst->data;

    if (is_mul) {
        return ggml_cuda8_mul_scalar_f32_launch(d_src, scalar, d_dst, n);
    }

    return ggml_cuda8_add_scalar_f32_launch(d_src, scalar, d_dst, n);
}

static int exec_scalar_f32_host_staging(
    ggml_cuda8_context * ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst,
    int is_mul,
    int n
) {
    const size_t bytes = (size_t) n * sizeof(float);

    float scalar = 0.0f;

    if (read_scalar_value(src1, &scalar) != 0) {
        std::fprintf(stderr, "ggml-cuda8/scalar: failed to read scalar\n");
        return -1;
    }

    ggml_cuda8_backend_buffer * b_src = NULL;
    ggml_cuda8_backend_buffer * b_dst = NULL;

    if (ggml_cuda8_context_alloc_buffer(ctx, bytes, &b_src) != 0) {
        return -1;
    }

    if (ggml_cuda8_context_alloc_buffer(ctx, bytes, &b_dst) != 0) {
        ggml_cuda8_backend_buffer_free(b_src);
        return -1;
    }

    int rc = 0;

    if (ggml_cuda8_backend_buffer_upload(b_src, 0, src0->data, bytes) != 0) {
        rc = -1;
    }

    if (rc == 0) {
        const float * d_src =
            (const float *) ggml_cuda8_backend_buffer_get_base_const(b_src);

        float * d_dst =
            (float *) ggml_cuda8_backend_buffer_get_base(b_dst);

        if (is_mul) {
            rc = ggml_cuda8_mul_scalar_f32_launch(d_src, scalar, d_dst, n);
        } else {
            rc = ggml_cuda8_add_scalar_f32_launch(d_src, scalar, d_dst, n);
        }
    }

    if (rc == 0) {
        if (ggml_cuda8_backend_buffer_download(b_dst, 0, dst->data, bytes) != 0) {
            rc = -1;
        }
    }

    ggml_cuda8_backend_buffer_free(b_src);
    ggml_cuda8_backend_buffer_free(b_dst);

    return rc;
}

static int exec_scalar_f32_common(
    ggml_cuda8_context * ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst,
    int is_mul
) {
    if (!ggml_cuda8_supported_scalar_f32(ctx, src0, src1, dst)) {
        std::fprintf(stderr, "ggml-cuda8/scalar: unsupported layout\n");
        return -1;
    }

    const size_t n_size = tensor_nelements(src0);

    if (n_size > 2147483647u) {
        std::fprintf(stderr, "ggml-cuda8/scalar: too many elements: %zu\n", n_size);
        return -1;
    }

    const int n = (int) n_size;
    const size_t bytes = n_size * sizeof(float);

    const int src0_dev = tensor_is_cuda8_resident(src0, bytes);
    const int dst_dev  = tensor_is_cuda8_resident(dst,  bytes);

    if (src0_dev && dst_dev) {
        return exec_scalar_f32_device_resident(src0, src1, dst, is_mul, n);
    }

    if (src0_dev || dst_dev) {
        std::fprintf(stderr,
            "ggml-cuda8/scalar: mixed host/device src0/dst unsupported in G11A-4D\n");
        return -1;
    }

    return exec_scalar_f32_host_staging(ctx, src0, src1, dst, is_mul, n);
}

int ggml_cuda8_exec_add_scalar_f32(
    ggml_cuda8_context * ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst
) {
    return exec_scalar_f32_common(ctx, src0, src1, dst, 0);
}

int ggml_cuda8_exec_mul_scalar_f32(
    ggml_cuda8_context * ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst
) {
    return exec_scalar_f32_common(ctx, src0, src1, dst, 1);
}
