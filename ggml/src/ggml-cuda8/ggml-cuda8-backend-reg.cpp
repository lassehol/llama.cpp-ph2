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

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <utility>
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

// -- G39: unsupported-op logging ---------------------------------------------
//
// supports_op returns a bare bool, so when a real model falls back to the CPU
// there is nothing recording which op was refused. GGML_SCHED_DEBUG=2 shows
// where each node ended up running, but not why the CUDA8 backend declined it.
//
// Enable with GGML_CUDA8_DEBUG_OPS=1. Each distinct op signature is printed once
// when first refused, and a summary ordered by frequency is printed at exit.
// That summary is the implementation queue: the ops at the top are the ones
// costing the most graph splits.
//
// C++11 only - this file is compiled both by the host toolchain and by GCC 5.4
// inside the CUDA 8 container.

static bool cuda8_debug_ops_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * s = std::getenv("GGML_CUDA8_DEBUG_OPS");
        enabled = (s != NULL && s[0] != '\0' && s[0] != '0') ? 1 : 0;
    }
    return enabled == 1;
}

static std::map<std::string, int> & cuda8_rejected_ops() {
    static std::map<std::string, int> m;
    return m;
}

static std::mutex & cuda8_rejected_mutex() {
    static std::mutex m;
    return m;
}

// e.g. "GLU/swiglu dst=f32 src0=f32" - enough to identify what to implement,
// without so much detail that every shape becomes its own entry.
static std::string cuda8_op_signature(const struct ggml_tensor * op) {
    std::string s = ggml_op_name(op->op);

    if (op->op == GGML_OP_UNARY) {
        s += "/";
        s += ggml_unary_op_name(ggml_get_unary_op(op));
    } else if (op->op == GGML_OP_GLU) {
        s += "/";
        s += ggml_glu_op_name(ggml_get_glu_op(op));
    }

    s += " dst=";
    s += ggml_type_name(op->type);

    for (int i = 0; i < 2; i++) {
        if (op->src[i] != NULL) {
            s += (i == 0) ? " src0=" : " src1=";
            s += ggml_type_name(op->src[i]->type);
        }
    }

    // SOFT_MAX is refused for reasons invisible in the types alone (G37).
    if (op->op == GGML_OP_SOFT_MAX && !ggml_cuda8_soft_max_is_plain(op)) {
        s += " [soft_max_ext: mask/sinks/scale/max_bias]";
    }

    return s;
}

static void cuda8_print_rejection_summary() {
    std::map<std::string, int> & m = cuda8_rejected_ops();
    if (m.empty()) {
        return;
    }

    std::vector<std::pair<int, std::string> > sorted;
    for (std::map<std::string, int>::const_iterator it = m.begin(); it != m.end(); ++it) {
        sorted.push_back(std::make_pair(it->second, it->first));
    }
    std::sort(sorted.begin(), sorted.end());
    std::reverse(sorted.begin(), sorted.end());

    std::fprintf(stderr, "\nggml-cuda8: ops refused by supports_op (ran on CPU), by frequency:\n");
    for (size_t i = 0; i < sorted.size(); i++) {
        std::fprintf(stderr, "  %8d  %s\n", sorted[i].first, sorted[i].second.c_str());
    }
    std::fprintf(stderr, "\n");
}

static void cuda8_note_rejected(const struct ggml_tensor * op) {
    if (!cuda8_debug_ops_enabled()) {
        return;
    }

    const std::string sig = cuda8_op_signature(op);

    std::lock_guard<std::mutex> lock(cuda8_rejected_mutex());
    std::map<std::string, int> & m = cuda8_rejected_ops();
    std::map<std::string, int>::iterator it = m.find(sig);

    if (it != m.end()) {
        it->second += 1;
        return;
    }

    std::fprintf(stderr, "ggml-cuda8: unsupported -> CPU: %s\n", sig.c_str());
    m.insert(std::make_pair(sig, 1));

    static bool summary_registered = false;
    if (!summary_registered) {
        std::atexit(cuda8_print_rejection_summary);
        summary_registered = true;
    }
}

static bool ggml_backend_cuda8_device_supports_op_impl(const struct ggml_tensor * op);

static bool ggml_backend_cuda8_device_supports_op(ggml_backend_dev_t dev,
                                                    const struct ggml_tensor * op) {
    (void) dev;

    const bool supported = ggml_backend_cuda8_device_supports_op_impl(op);
    if (!supported) {
        cuda8_note_rejected(op);
    }
    return supported;
}

static bool ggml_backend_cuda8_device_supports_op_impl(const struct ggml_tensor * op) {
    switch (op->op) {
        // no-ops (metadata only)
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
		case GGML_OP_MUL_MAT: {
			//if (op->src[0]->type == GGML_TYPE_F32) return false; // TEMP
            if (!op->src[0] || !op->src[1]) return false;
            if (op->type != GGML_TYPE_F32) return false;
            if (op->src[1]->type != GGML_TYPE_F32) return false;
            if (op->src[0]->type == GGML_TYPE_Q8_0 ||
                op->src[0]->type == GGML_TYPE_Q4_K ||
                op->src[0]->type == GGML_TYPE_Q6_K) {
                return true;
            }
            // G42: batched F32xF32 (attention matmuls). Re-validates shape
            // independently rather than trusting the dispatcher alone.
            if (op->src[0]->type != GGML_TYPE_F32) return false;
            if (op->src[0]->nb[0] != sizeof(float)) return false;
            if (op->src[1]->nb[0] != sizeof(float)) return false;
            if (op->nb[0] != sizeof(float)) return false;
            if (op->src[0]->ne[0] != op->src[1]->ne[0]) return false;
            if (op->src[0]->ne[2] <= 0 || op->src[0]->ne[3] <= 0) return false;
            if (op->src[1]->ne[2] % op->src[0]->ne[2] != 0) return false;
            if (op->src[1]->ne[3] % op->src[0]->ne[3] != 0) return false;
            if (op->ne[0] != op->src[0]->ne[1]) return false;
            if (op->ne[1] != op->src[1]->ne[1]) return false;
            if (op->ne[2] != op->src[1]->ne[2]) return false;
            if (op->ne[3] != op->src[1]->ne[3]) return false;
            return true;
        }

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
		case GGML_OP_SOFT_MAX: {
			//if (op->src[0]->type == GGML_TYPE_F32) return false; // TEMP
			if (op->type != GGML_TYPE_F32) return false;
			if (!op->src[0] || op->src[0]->type != GGML_TYPE_F32) return false;
			if (ggml_cuda8_soft_max_is_plain(op)) return true;
			// G41: "proper" masked/scaled/ALiBi-biased case.
			if (!ggml_cuda8_soft_max_is_supported_ext(op)) return false;  // sinks / F16 mask
			if (op->src[1] != NULL) {
				const struct ggml_tensor * mask = op->src[1];
				if (mask->ne[0] != op->src[0]->ne[0]) return false;
				if (mask->ne[1] != op->src[0]->ne[1]) return false;
				if (mask->ne[2] != 1 && mask->ne[2] != op->src[0]->ne[2]) return false;
				if (mask->ne[3] != 1 && mask->ne[3] != op->src[0]->ne[3]) return false;
			}
			return true;
		}

        // row reductions
        case GGML_OP_SUM_ROWS:
            return (op->type == GGML_TYPE_F32 &&
                    op->src[0] && op->src[0]->type == GGML_TYPE_F32);


        // RMS normalization
        case GGML_OP_RMS_NORM:
			return (op->type == GGML_TYPE_F32 &&
                    op->src[0] && op->src[0]->type == GGML_TYPE_F32);

        // rotary positional embeddings: NORMAL (0) and NEOX (2)
        case GGML_OP_ROPE: {
            if (op->type != GGML_TYPE_F32) return false;
            if (!op->src[0] || op->src[0]->type != GGML_TYPE_F32) return false;
            if (!op->src[1] || op->src[1]->type != GGML_TYPE_I32) return false;

            // G45: freq_factors. rope_yarn divides theta by freq_factors[pair]
            // (ggml-cpu/ops.cpp, ggml_rope_cache_init). The kernel does not read
            // src[2], so accepting a node carrying one would silently apply the
            // wrong rotation - the same failure mode as SOFT_MAX in G37.
            if (op->src[2] != NULL) return false;

            int32_t params[15];
            std::memcpy(params, op->op_params, sizeof(params));

            const int mode = params[2];

            float ext_factor, attn_factor;
            std::memcpy(&ext_factor,  &params[7], sizeof(float));
            std::memcpy(&attn_factor, &params[8], sizeof(float));

            // MROPE (8), VISION (24) and IMROPE (40) use different pair layouts
            // and section handling that this kernel does not implement.
            if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
                return false;
            }

            // YaRN: ext_factor blends interpolated and extrapolated theta.
            if (ext_factor != 0.0f) return false;

            // attn_factor scales the cos/sin magnitude (mscale in rope_yarn).
            // The kernel always uses magnitude 1.
            if (attn_factor != 1.0f) return false;

            return true;
        }

        // contiguous copy
        case GGML_OP_CONT:
			//if (op->src[0]->type == GGML_TYPE_F32) return false; // TEMP
            return (op->type == GGML_TYPE_F32 &&
                    op->src[0] && op->src[0]->type == GGML_TYPE_F32);

        // causal masking
        case GGML_OP_DIAG_MASK_INF:
            return (op->type == GGML_TYPE_F32 &&
                    op->src[0] && op->src[0]->type == GGML_TYPE_F32);

        // G43: SET_ROWS - the KV cache write. Refusing this is fatal rather
        // than merely slow: the destination is pre-allocated in our buffer, so
        // the scheduler has nowhere else to put it and aborts. Hence the F32
        // cache requirement (--cache-type-k/v f32) until G49 adds F16 stores.
        case GGML_OP_SET_ROWS: {
			//if (op->src[0]->type == GGML_TYPE_F32) return false; // TEMP
            if (op->type != GGML_TYPE_F32) return false;
            if (!op->src[0] || op->src[0]->type != GGML_TYPE_F32) return false;
            if (!op->src[1] || op->src[1]->type != GGML_TYPE_I64) return false;

            if (op->ne[0] != op->src[0]->ne[0]) return false;
            if (op->ne[2] != op->src[0]->ne[2]) return false;
            if (op->ne[3] != op->src[0]->ne[3]) return false;

            // Index broadcast over dims 2/3, per the CPU reference.
            if (op->src[1]->ne[1] == 0 || op->src[1]->ne[2] == 0) return false;
            if (op->src[0]->ne[2] % op->src[1]->ne[1] != 0) return false;
            if (op->src[0]->ne[3] % op->src[1]->ne[2] != 0) return false;
            if (op->src[1]->ne[0] < op->src[0]->ne[1]) return false;

            return true;
        }

        // G40: SwiGLU gated activation - the FFN.
        case GGML_OP_GLU: {
            if (op->type != GGML_TYPE_F32) return false;
            if (!op->src[0] || op->src[0]->type != GGML_TYPE_F32) return false;

            // Only plain SWIGLU. SWIGLU_OAI carries alpha/limit in
            // op_params[2..3] that the kernel does not apply; REGLU/GEGLU/
            // GEGLU_ERF/GEGLU_QUICK use different activations entirely.
            if (ggml_get_glu_op(op) != GGML_GLU_OP_SWIGLU) return false;

            // src[1] NULL is legitimate: the halves form, where gate and up
            // are the two halves of each src0 row.
            if (op->src[1] != NULL) {
                if (op->src[1]->type != GGML_TYPE_F32) return false;
                if (!ggml_are_same_shape(op->src[0], op->src[1])) return false;
                if (!ggml_is_contiguous_1(op->src[1])) return false;
            } else {
                if ((op->src[0]->ne[0] % 2) != 0) return false;
            }

            // The kernel indexes rows by nb[1], so rows must be contiguous.
            if (!ggml_is_contiguous_1(op->src[0])) return false;
            if (!ggml_is_contiguous_1(op))         return false;

            const int64_t nc = op->src[1] ? op->src[0]->ne[0] : op->src[0]->ne[0] / 2;
            if (op->ne[0] != nc) return false;

            return true;
        }

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
