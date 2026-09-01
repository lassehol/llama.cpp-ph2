// ggml-cuda8-ggml-backend-supports-op-smoke.cpp
// G36C: supports_op integration test for CUDA8 backend
// G37:  added SOFT_MAX soft_max_ext rejection cases
//
// Calls ggml_backend_cuda8_reg() directly (avoids C++17 ggml-backend-reg.cpp).
// Tests:
//   1. Registry returns valid backend with devices
//   2. Device interface works (name, description, memory, type, props)
//   3. supports_op returns true for all 15 dispatch ops + 5 no-ops
//   4. supports_op returns false for unsupported ops, including the
//      soft_max_ext forms (mask / sinks / scale / max_bias) that would
//      otherwise be silently miscomputed
//   5. init_backend from device works

#include "ggml-cuda8-ggml-backend.h"
#include "ggml-cuda8-ggml-buffer.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// Declared in ggml-cuda8-backend-reg.cpp
extern "C" ggml_backend_reg_t ggml_backend_cuda8_reg(void);

// G37: stamp SOFT_MAX op_params the way ggml_soft_max_impl() does.
// Note that a zeroed op_params block is NOT a plain softmax - it reads as
// scale=0.0f - so every SOFT_MAX fixture has to set this explicitly.
static void set_soft_max_params(ggml_tensor * t, float scale, float max_bias) {
    float p[2] = { scale, max_bias };
    std::memcpy(t->op_params, p, sizeof(p));
}

// G45: stamp ROPE op_params the way ggml_rope_ext() does.
// Same trap as SOFT_MAX: a memset fixture leaves attn_factor at 0.0f, but real
// ggml passes 1.0f, and supports_op refuses anything else because the kernel
// does not apply it.
static void set_rope_params(ggml_tensor * t, int n_dims, int mode,
                            float freq_base, float freq_scale,
                            float ext_factor, float attn_factor) {
    int32_t p[15];
    std::memset(p, 0, sizeof(p));
    p[1] = n_dims;
    p[2] = mode;
    std::memcpy(&p[5],  &freq_base,   sizeof(float));
    std::memcpy(&p[6],  &freq_scale,  sizeof(float));
    std::memcpy(&p[7],  &ext_factor,  sizeof(float));
    std::memcpy(&p[8],  &attn_factor, sizeof(float));
    std::memcpy(t->op_params, p, sizeof(p));
}

// Create a minimal fake tensor for supports_op testing
static ggml_tensor make_fake(ggml_type type, int64_t ne0, int64_t ne1) {
    ggml_tensor t;
    std::memset(&t, 0, sizeof(t));
    t.type  = type;
    t.ne[0] = ne0; t.ne[1] = ne1; t.ne[2] = 1; t.ne[3] = 1;
    t.nb[0] = (type == GGML_TYPE_F32) ? sizeof(float) :
              (type == GGML_TYPE_I32) ? sizeof(int32_t) :
              sizeof(float);
    t.nb[1] = t.nb[0] * ne0;
    t.nb[2] = t.nb[1] * ne1;
    t.nb[3] = t.nb[2];
    return t;
}

static bool test_op(ggml_backend_dev_t dev, const char * label,
                     ggml_tensor * op_t, bool expected) {
    bool result = dev->iface.supports_op(dev, op_t);
    bool pass = (result == expected);
    std::printf("  %-28s expected=%-5s got=%-5s %s\n",
                label,
                expected ? "true" : "false",
                result   ? "true" : "false",
                pass ? "PASS" : "FAIL");
    return pass;
}

int main() {
    std::printf("ggml-cuda8-supports-op-smoke: starting\n\n");

 // -- 1. Registry --
    std::printf("== Registry ==\n");
    ggml_backend_reg_t reg = ggml_backend_cuda8_reg();
    if (reg == NULL) {
        std::fprintf(stderr, "FAIL: ggml_backend_cuda8_reg() returned NULL\n");
        return 1;
    }
    std::printf("  api_version: %d\n", reg->api_version);
    std::printf("  name: %s\n", reg->iface.get_name(reg));

    size_t n_dev = reg->iface.get_device_count(reg);
    std::printf("  device_count: %zu\n", n_dev);

    if (n_dev == 0) {
        std::fprintf(stderr, "FAIL: no CUDA8 devices found (need cc 2.x-3.x)\n");
        return 1;
    }

 // -- 2. Device interface --
    ggml_backend_dev_t dev = reg->iface.get_device(reg, 0);
    if (dev == NULL) {
        std::fprintf(stderr, "FAIL: get_device(0) returned NULL\n");
        return 1;
    }

    std::printf("\n== Device 0 ==\n");
    std::printf("  name:        %s\n", dev->iface.get_name(dev));
    std::printf("  description: %s\n", dev->iface.get_description(dev));

    size_t mem_free = 0, mem_total = 0;
    dev->iface.get_memory(dev, &mem_free, &mem_total);
    std::printf("  memory:      %.1f MiB free / %.1f MiB total\n",
                (double)mem_free / (1024.0*1024.0),
                (double)mem_total / (1024.0*1024.0));

    enum ggml_backend_dev_type dtype = dev->iface.get_type(dev);
    std::printf("  type:        %d (GPU=%d)\n", (int)dtype, (int)GGML_BACKEND_DEVICE_TYPE_GPU);

    ggml_backend_buffer_type_t buft = dev->iface.get_buffer_type(dev);
    std::printf("  buffer_type: %s\n", buft ? buft->iface.get_name(buft) : "NULL");

 // -- 3. supports_op: TRUE cases --
    std::printf("\n== supports_op: expected TRUE ==\n");
    bool ok = true;

    ggml_tensor f32_128   = make_fake(GGML_TYPE_F32, 128, 1);
    ggml_tensor f32_64    = make_fake(GGML_TYPE_F32, 64, 1);
    ggml_tensor f32_1     = make_fake(GGML_TYPE_F32, 1, 1);
    ggml_tensor i32_1     = make_fake(GGML_TYPE_I32, 1, 1);
    ggml_tensor i32_4     = make_fake(GGML_TYPE_I32, 4, 1);
    ggml_tensor q8_128x64 = make_fake(GGML_TYPE_Q8_0, 128, 64);
    ggml_tensor f32_2d    = make_fake(GGML_TYPE_F32, 128, 64);
    ggml_tensor op_t;

    // NONE
    op_t = make_fake(GGML_TYPE_F32, 128, 1);
    op_t.op = GGML_OP_NONE;
    ok &= test_op(dev, "NONE", &op_t, true);

    // RESHAPE
    op_t = make_fake(GGML_TYPE_F32, 64, 2);
    op_t.op = GGML_OP_RESHAPE;
    op_t.src[0] = &f32_128;
    ok &= test_op(dev, "RESHAPE", &op_t, true);

    // VIEW
    op_t = make_fake(GGML_TYPE_F32, 64, 1);
    op_t.op = GGML_OP_VIEW;
    op_t.src[0] = &f32_128;
    ok &= test_op(dev, "VIEW", &op_t, true);

    // PERMUTE
    op_t = make_fake(GGML_TYPE_F32, 64, 2);
    op_t.op = GGML_OP_PERMUTE;
    op_t.src[0] = &f32_2d;
    ok &= test_op(dev, "PERMUTE", &op_t, true);

    // TRANSPOSE
    op_t = make_fake(GGML_TYPE_F32, 64, 128);
    op_t.op = GGML_OP_TRANSPOSE;
    op_t.src[0] = &f32_2d;
    ok &= test_op(dev, "TRANSPOSE", &op_t, true);

    // ADD
    op_t = make_fake(GGML_TYPE_F32, 128, 1);
    op_t.op = GGML_OP_ADD;
    op_t.src[0] = &f32_128;
    op_t.src[1] = &f32_128;
    ok &= test_op(dev, "ADD (F32)", &op_t, true);

    // MUL elem
    op_t = make_fake(GGML_TYPE_F32, 128, 1);
    op_t.op = GGML_OP_MUL;
    op_t.src[0] = &f32_128;
    op_t.src[1] = &f32_128;
    ok &= test_op(dev, "MUL (elem F32)", &op_t, true);

    // MUL scalar
    op_t = make_fake(GGML_TYPE_F32, 128, 1);
    op_t.op = GGML_OP_MUL;
    op_t.src[0] = &f32_128;
    op_t.src[1] = &f32_1;
    ok &= test_op(dev, "MUL (scalar F32)", &op_t, true);

    // SOFT_MAX (plain: no mask, no sinks, scale=1, max_bias=0)
    op_t = make_fake(GGML_TYPE_F32, 64, 1);
    op_t.op = GGML_OP_SOFT_MAX;
    op_t.src[0] = &f32_64;
    set_soft_max_params(&op_t, 1.0f, 0.0f);
    ok &= test_op(dev, "SOFT_MAX (plain)", &op_t, true);

    // SUM_ROWS
    op_t = make_fake(GGML_TYPE_F32, 1, 64);
    op_t.op = GGML_OP_SUM_ROWS;
    op_t.src[0] = &f32_2d;
    ok &= test_op(dev, "SUM_ROWS", &op_t, true);

    // MUL_MAT Q8_0xF32
    op_t = make_fake(GGML_TYPE_F32, 64, 1);
    op_t.op = GGML_OP_MUL_MAT;
    op_t.src[0] = &q8_128x64;
    op_t.src[1] = &f32_128;
    ok &= test_op(dev, "MUL_MAT (Q8_0xF32)", &op_t, true);

    // RMS_NORM
    op_t = make_fake(GGML_TYPE_F32, 128, 1);
    op_t.op = GGML_OP_RMS_NORM;
    op_t.src[0] = &f32_128;
    ok &= test_op(dev, "RMS_NORM", &op_t, true);

    // ROPE NORMAL (mode=0)
    op_t = make_fake(GGML_TYPE_F32, 64, 1);
    op_t.op = GGML_OP_ROPE;
    op_t.src[0] = &f32_64;
    op_t.src[1] = &i32_1;
    set_rope_params(&op_t, 64, 0, 10000.0f, 1.0f, 0.0f, 1.0f);
    ok &= test_op(dev, "ROPE (mode=0 NORMAL)", &op_t, true);

    // G45: ROPE NEOX (mode=2) - now supported. Qwen3 and friends use this.
    op_t = make_fake(GGML_TYPE_F32, 64, 1);
    op_t.op = GGML_OP_ROPE;
    op_t.src[0] = &f32_64;
    op_t.src[1] = &i32_1;
    set_rope_params(&op_t, 64, 2, 1000000.0f, 1.0f, 0.0f, 1.0f);
    ok &= test_op(dev, "ROPE (mode=2 NEOX)", &op_t, true);

    // CONT
    op_t = make_fake(GGML_TYPE_F32, 64, 1);
    op_t.op = GGML_OP_CONT;
    op_t.src[0] = &f32_64;
    ok &= test_op(dev, "CONT", &op_t, true);

    // DIAG_MASK_INF
    op_t = make_fake(GGML_TYPE_F32, 64, 8);
    op_t.op = GGML_OP_DIAG_MASK_INF;
    op_t.src[0] = &f32_2d;
    ok &= test_op(dev, "DIAG_MASK_INF", &op_t, true);

    // GET_ROWS
    op_t = make_fake(GGML_TYPE_F32, 128, 4);
    op_t.op = GGML_OP_GET_ROWS;
    op_t.src[0] = &f32_2d;
    op_t.src[1] = &i32_4;
    ok &= test_op(dev, "GET_ROWS", &op_t, true);

 // -- 4. supports_op: FALSE cases --
    std::printf("\n== supports_op: expected FALSE ==\n");

    // MUL_MAT F32xF32
    op_t = make_fake(GGML_TYPE_F32, 64, 1);
    op_t.op = GGML_OP_MUL_MAT;
    op_t.src[0] = &f32_2d;
    op_t.src[1] = &f32_128;
    ok &= test_op(dev, "MUL_MAT (F32xF32)", &op_t, false);

    // ROPE MROPE (mode=8) - different pair layout and section handling
    op_t = make_fake(GGML_TYPE_F32, 64, 1);
    op_t.op = GGML_OP_ROPE;
    op_t.src[0] = &f32_64;
    op_t.src[1] = &i32_1;
    set_rope_params(&op_t, 64, 8, 10000.0f, 1.0f, 0.0f, 1.0f);
    ok &= test_op(dev, "ROPE (mode=8 MROPE)", &op_t, false);

    // ROPE VISION (mode=24)
    op_t = make_fake(GGML_TYPE_F32, 64, 1);
    op_t.op = GGML_OP_ROPE;
    op_t.src[0] = &f32_64;
    op_t.src[1] = &i32_1;
    set_rope_params(&op_t, 64, 24, 10000.0f, 1.0f, 0.0f, 1.0f);
    ok &= test_op(dev, "ROPE (mode=24 VISION)", &op_t, false);

    // ROPE YaRN
    op_t = make_fake(GGML_TYPE_F32, 64, 1);
    op_t.op = GGML_OP_ROPE;
    op_t.src[0] = &f32_64;
    op_t.src[1] = &i32_1;
    set_rope_params(&op_t, 64, 0, 10000.0f, 1.0f, 1.0f /* ext */, 1.0f);
    ok &= test_op(dev, "ROPE (YaRN ext=1)", &op_t, false);

    // G45: attn_factor scales cos/sin magnitude; the kernel always uses 1.
    op_t = make_fake(GGML_TYPE_F32, 64, 1);
    op_t.op = GGML_OP_ROPE;
    op_t.src[0] = &f32_64;
    op_t.src[1] = &i32_1;
    set_rope_params(&op_t, 64, 0, 10000.0f, 1.0f, 0.0f, 0.8f /* attn */);
    ok &= test_op(dev, "ROPE (attn_factor!=1)", &op_t, false);

    // G45: freq_factors in src[2] rescale theta per pair; the kernel ignores it.
    op_t = make_fake(GGML_TYPE_F32, 64, 1);
    op_t.op = GGML_OP_ROPE;
    op_t.src[0] = &f32_64;
    op_t.src[1] = &i32_1;
    op_t.src[2] = &f32_64;   // freq_factors
    set_rope_params(&op_t, 64, 0, 10000.0f, 1.0f, 0.0f, 1.0f);
    ok &= test_op(dev, "ROPE (freq_factors)", &op_t, false);

    // ROPE with zeroed op_params - attn_factor reads as 0, not a valid node
    op_t = make_fake(GGML_TYPE_F32, 64, 1);
    op_t.op = GGML_OP_ROPE;
    op_t.src[0] = &f32_64;
    op_t.src[1] = &i32_1;
    ok &= test_op(dev, "ROPE (zero params)", &op_t, false);

    // G37: soft_max_ext forms. The SOFTMAX_ROWS_F32 kernel implements none of
    // these - it takes no mask, no sinks, and never reads op_params. If
    // supports_op claims them, the result is not a crash but silently wrong
    // attention weights, so each must be refused and left to the CPU backend.

    // SOFT_MAX with attention mask (src[1])
    op_t = make_fake(GGML_TYPE_F32, 64, 8);
    op_t.op = GGML_OP_SOFT_MAX;
    op_t.src[0] = &f32_2d;
    op_t.src[1] = &f32_2d;  // mask
    set_soft_max_params(&op_t, 1.0f, 0.0f);
    ok &= test_op(dev, "SOFT_MAX (mask)", &op_t, false);

    // SOFT_MAX with attention sinks (src[2])
    op_t = make_fake(GGML_TYPE_F32, 64, 8);
    op_t.op = GGML_OP_SOFT_MAX;
    op_t.src[0] = &f32_2d;
    op_t.src[2] = &f32_1;  // sinks
    set_soft_max_params(&op_t, 1.0f, 0.0f);
    ok &= test_op(dev, "SOFT_MAX (sinks)", &op_t, false);

    // SOFT_MAX with attention scale (1/sqrt(128) - the real-attention case)
    op_t = make_fake(GGML_TYPE_F32, 64, 1);
    op_t.op = GGML_OP_SOFT_MAX;
    op_t.src[0] = &f32_64;
    set_soft_max_params(&op_t, 0.08838835f, 0.0f);
    ok &= test_op(dev, "SOFT_MAX (scale!=1)", &op_t, false);

    // SOFT_MAX with ALiBi max_bias
    op_t = make_fake(GGML_TYPE_F32, 64, 8);
    op_t.op = GGML_OP_SOFT_MAX;
    op_t.src[0] = &f32_2d;
    op_t.src[1] = &f32_2d;  // max_bias > 0 implies a mask
    set_soft_max_params(&op_t, 1.0f, 8.0f);
    ok &= test_op(dev, "SOFT_MAX (max_bias)", &op_t, false);

    // SOFT_MAX with zeroed op_params - scale reads as 0.0f, not a plain softmax
    op_t = make_fake(GGML_TYPE_F32, 64, 1);
    op_t.op = GGML_OP_SOFT_MAX;
    op_t.src[0] = &f32_64;
    ok &= test_op(dev, "SOFT_MAX (zero params)", &op_t, false);

    // CPY
    op_t = make_fake(GGML_TYPE_F32, 128, 1);
    op_t.op = GGML_OP_CPY;
    op_t.src[0] = &f32_128;
    ok &= test_op(dev, "CPY", &op_t, false);

    // SCALE
    op_t = make_fake(GGML_TYPE_F32, 128, 1);
    op_t.op = GGML_OP_SCALE;
    op_t.src[0] = &f32_128;
    ok &= test_op(dev, "SCALE", &op_t, false);

 // -- 5. Backend init --
    std::printf("\n== Backend init from device ==\n");
    ggml_backend_t backend = dev->iface.init_backend(dev, NULL);
    if (backend) {
        std::printf("  init_backend: PASS (name=%s)\n",
                    backend->iface.get_name(backend));
        backend->iface.free(backend);
    } else {
        std::printf("  init_backend: FAIL\n");
        ok = false;
    }

 // -- Result --
    std::printf("\nggml-cuda8-supports-op-smoke: %s\n", ok ? "SUCCESS" : "FAIL");
    return ok ? 0 : 1;
}
