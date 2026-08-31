// ggml-cuda8-backend-reg.cpp
// G36A: Backend auto-registration for CUDA8/Fermi backend
//
// Implements:
//   ggml_backend_cuda8_reg()       - registry entry point
//   ggml_backend_cuda8_device_*    - device interface
//   supports_op                    - declares supported operations

#include "ggml-cuda8-ggml-backend.h"
#include "ggml-cuda8-ggml-buffer.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

// -- device context -----------------------------------------------------------

struct ggml_backend_cuda8_device_context {
    int device;
    std::string name;
    std::string description;
};

// -- device interface ---------------------------------------------------------

static const char * ggml_backend_cuda8_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cuda8_device_context * ctx =
        (ggml_backend_cuda8_device_context *) dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_cuda8_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cuda8_device_context * ctx =
        (ggml_backend_cuda8_device_context *) dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_cuda8_device_get_memory(ggml_backend_dev_t dev,
                                                   size_t * free, size_t * total) {
    ggml_backend_cuda8_device_context * ctx =
        (ggml_backend_cuda8_device_context *) dev->context;
    cudaSetDevice(ctx->device);
    cudaMemGetInfo(free, total);
}

static enum ggml_backend_dev_type ggml_backend_cuda8_device_get_type(ggml_backend_dev_t dev) {
    (void) dev;
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_cuda8_device_get_props(ggml_backend_dev_t dev,
                                                  struct ggml_backend_dev_props * props) {
    ggml_backend_cuda8_device_context * ctx =
        (ggml_backend_cuda8_device_context *) dev->context;

    props->name        = ctx->name.c_str();
    props->description = ctx->description.c_str();

    cudaSetDevice(ctx->device);
    cudaMemGetInfo(&props->memory_free, &props->memory_total);

    props->type      = GGML_BACKEND_DEVICE_TYPE_GPU;
    props->device_id = NULL;

    props->caps = {
        /* .async              = */ false,
        /* .host_buffer        = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events             = */ false,
    };
}

static ggml_backend_t ggml_backend_cuda8_device_init_backend(ggml_backend_dev_t dev,
                                                               const char * params) {
    (void) params;
    ggml_backend_cuda8_device_context * ctx =
        (ggml_backend_cuda8_device_context *) dev->context;
    ggml_backend_t backend = ggml_cuda8_ggml_backend_init(ctx->device);
    if (backend) {
        backend->device = dev;
    }
    return backend;
}

static ggml_backend_buffer_type_t ggml_backend_cuda8_device_get_buffer_type(ggml_backend_dev_t dev) {
    (void) dev;
    return ggml_cuda8_ggml_buffer_type();
}

static ggml_backend_buffer_type_t ggml_backend_cuda8_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    (void) dev;
    return NULL;  // no pinned host buffer support
}

static bool ggml_backend_cuda8_device_supports_op(ggml_backend_dev_t dev,
                                                    const struct ggml_tensor * op) {
    (void) dev;

    switch (op->op) {
        // no-ops (metadata only)
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        // F32 element-wise
        case GGML_OP_ADD:
            return (op->type == GGML_TYPE_F32 &&
                    op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                    op->src[1] && op->src[1]->type == GGML_TYPE_F32 &&
                    ggml_nelements(op) == ggml_nelements(op->src[1]));

        // MUL: element-wise F32 or scalar F32
        case GGML_OP_MUL:
            return (op->type == GGML_TYPE_F32 &&
                    op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                    op->src[1] && op->src[1]->type == GGML_TYPE_F32);

        // softmax: plain row-wise only.
        // G37: reject the ggml_soft_max_ext() forms (mask / sinks / scale /
        // max_bias) - the kernel ignores all of them, so claiming those nodes
        // would silently produce wrong attention weights. See
        // ggml_cuda8_soft_max_is_plain().
        case GGML_OP_SOFT_MAX:
            return (op->type == GGML_TYPE_F32 &&
                    op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                    ggml_cuda8_soft_max_is_plain(op));

        // row reductions
        case GGML_OP_SUM_ROWS:
            return (op->type == GGML_TYPE_F32 &&
                    op->src[0] && op->src[0]->type == GGML_TYPE_F32);

        // quantized x F32 matrix multiply (Q8_0, Q4_K, Q6_K)
        case GGML_OP_MUL_MAT:
            return (op->type == GGML_TYPE_F32 &&
                    op->src[0] &&
                    (op->src[0]->type == GGML_TYPE_Q8_0 ||
                     op->src[0]->type == GGML_TYPE_Q4_K ||
                     op->src[0]->type == GGML_TYPE_Q6_K) &&
                    op->src[1] && op->src[1]->type == GGML_TYPE_F32);

        // RMS normalization
        case GGML_OP_RMS_NORM:
            return (op->type == GGML_TYPE_F32 &&
                    op->src[0] && op->src[0]->type == GGML_TYPE_F32);

        // rotary positional embeddings (mode=0 only)
        case GGML_OP_ROPE: {
            if (op->type != GGML_TYPE_F32) return false;
            if (!op->src[0] || op->src[0]->type != GGML_TYPE_F32) return false;
            if (!op->src[1] || op->src[1]->type != GGML_TYPE_I32) return false;
            // check mode=0 and ext_factor=0
            int32_t params[15];
            std::memcpy(params, op->op_params, sizeof(params));
            int mode = params[2];
            float ext_factor;
            std::memcpy(&ext_factor, &params[7], sizeof(float));
            return (mode == 0 && ext_factor == 0.0f);
        }

        // contiguous copy
        case GGML_OP_CONT:
            return (op->type == GGML_TYPE_F32 &&
                    op->src[0] && op->src[0]->type == GGML_TYPE_F32);

        // causal masking
        case GGML_OP_DIAG_MASK_INF:
            return (op->type == GGML_TYPE_F32 &&
                    op->src[0] && op->src[0]->type == GGML_TYPE_F32);

        // embedding lookup (F32, Q4_K, Q6_K)
        case GGML_OP_GET_ROWS:
            return (op->type == GGML_TYPE_F32 &&
                    op->src[0] &&
                    (op->src[0]->type == GGML_TYPE_F32 ||
                     op->src[0]->type == GGML_TYPE_Q4_K ||
                     op->src[0]->type == GGML_TYPE_Q6_K) &&
                    op->src[1] && op->src[1]->type == GGML_TYPE_I32);

        default:
            return false;
    }
}

static bool ggml_backend_cuda8_device_supports_buft(ggml_backend_dev_t dev,
                                                      ggml_backend_buffer_type_t buft) {
    (void) dev;
    // Accept our own buffer type
    return (buft == ggml_cuda8_ggml_buffer_type());
}

static bool ggml_backend_cuda8_device_offload_op(ggml_backend_dev_t dev,
                                                   const struct ggml_tensor * op) {
    (void) dev;
    // Only offload if the weight tensor (src[0]) is already on our GPU.
    // This prevents the scheduler from routing CPU-resident layers to CUDA8.
    switch (op->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_GET_ROWS:
            if (op->src[0] && op->src[0]->buffer &&
                op->src[0]->buffer->buft == ggml_cuda8_ggml_buffer_type()) {
                return true;
            }
            return false;
        default:
            return false;
    }
}

static const ggml_backend_device_i ggml_backend_cuda8_device_interface = {
    /* .get_name                = */ ggml_backend_cuda8_device_get_name,
    /* .get_description         = */ ggml_backend_cuda8_device_get_description,
    /* .get_memory              = */ ggml_backend_cuda8_device_get_memory,
    /* .get_type                = */ ggml_backend_cuda8_device_get_type,
    /* .get_props               = */ ggml_backend_cuda8_device_get_props,
    /* .init_backend            = */ ggml_backend_cuda8_device_init_backend,
    /* .get_buffer_type         = */ ggml_backend_cuda8_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_cuda8_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_backend_cuda8_device_supports_op,
    /* .supports_buft           = */ ggml_backend_cuda8_device_supports_buft,
    /* .offload_op              = */ ggml_backend_cuda8_device_offload_op,
    /* .event_new               = */ NULL,
    /* .event_free              = */ NULL,
    /* .event_synchronize       = */ NULL,
};

// -- registry -----------------------------------------------------------------

struct ggml_backend_cuda8_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_cuda8_reg_get_name(ggml_backend_reg_t reg) {
    (void) reg;
    return "CUDA8";
}

static size_t ggml_backend_cuda8_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_cuda8_reg_context * ctx =
        (ggml_backend_cuda8_reg_context *) reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_cuda8_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_cuda8_reg_context * ctx =
        (ggml_backend_cuda8_reg_context *) reg->context;
    if (index >= ctx->devices.size()) return NULL;
    return ctx->devices[index];
}

static void * ggml_backend_cuda8_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    (void) reg;
    (void) name;
    return NULL;
}

static const ggml_backend_reg_i ggml_backend_cuda8_reg_interface = {
    /* .get_name          = */ ggml_backend_cuda8_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cuda8_reg_get_device_count,
    /* .get_device        = */ ggml_backend_cuda8_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cuda8_reg_get_proc_address,
};

// -- public entry point -------------------------------------------------------

extern "C" void ggml_cuda8_ggml_buffer_type_set_device(ggml_backend_dev_t dev);
extern "C" ggml_backend_reg_t ggml_backend_cuda8_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    if (!initialized) {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            // No CUDA devices -- return empty registry
            static ggml_backend_cuda8_reg_context empty_ctx;
            reg.api_version = GGML_BACKEND_API_VERSION;
            reg.iface   = ggml_backend_cuda8_reg_interface;
            reg.context = &empty_ctx;
            initialized = true;
            return &reg;
        }

        // Only register Fermi-era devices (compute capability 2.x)
        // or if the user explicitly wants CUDA8 backend
        static ggml_backend_cuda8_reg_context * ctx = new ggml_backend_cuda8_reg_context;

        for (int i = 0; i < device_count; ++i) {
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) continue;

            // Accept compute capability 2.x (Fermi) through 3.x (Kepler)
            // These are the GPUs that can't run modern CUDA but work with CUDA 8
            if (prop.major > 3) continue;

            ggml_backend_cuda8_device_context * dev_ctx =
                new ggml_backend_cuda8_device_context;
            dev_ctx->device = i;
            dev_ctx->name = std::string("CUDA8_") + std::to_string(i);
            dev_ctx->description = prop.name;

            ggml_backend_dev_t dev = new ggml_backend_device {
                /* .iface   = */ ggml_backend_cuda8_device_interface,
                /* .reg     = */ &reg,
                /* .context = */ dev_ctx,
            };
            // Set device pointer on buffer type so ggml can find our device
            ggml_cuda8_ggml_buffer_type_set_device(dev);
            ctx->devices.push_back(dev);
        }

        reg.api_version = GGML_BACKEND_API_VERSION;
        reg.iface   = ggml_backend_cuda8_reg_interface;
        reg.context = ctx;
        initialized = true;
    }

    return &reg;
}
