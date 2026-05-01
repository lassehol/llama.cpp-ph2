// ggml-cuda8-ggml-backend-graph-builder-transformer-block-smoke.cpp
// G32A: Full transformer block pipeline through CUDA8 graph_compute
//
// Pipeline:
//   x[128] -> rms_norm(eps) -> mul(w_norm) -> mul_mat(W_q8[64x128])
//          -> add(bias) -> mul(scale) -> soft_max -> y[64]
//
// This exercises every major op category:
//   - Normalization: RMS_NORM + element-wise MUL
//   - Projection:    Q8_0 x F32 MUL_MAT
//   - Bias:          F32 ADD
//   - Scaling:       scalar MUL
//   - Activation:    SOFTMAX
//
// All ops dispatched through ggml_backend_i.graph_compute on GTX 560 / CUDA 8.

#include "ggml-cuda8-ggml-backend.h"
#include "ggml-cuda8-ggml-buffer.h"
#include "q8_0-mmv.cuh"

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <vector>


static uint16_t fp32_to_fp16_bits(float value) {
    union { float f; uint32_t u; } in;
    in.f = value;
    const uint32_t f = in.u;
    const uint32_t sign = (f >> 16) & 0x8000u;
    int32_t exp = (int32_t)((f >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = f & 0x007FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x00800000u;
        const int shift = 14 - exp;
        uint32_t half_mant = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp == 0xFF - 127 + 15) {
        return (mant == 0) ? (uint16_t)(sign | 0x7C00u)
                           : (uint16_t)(sign | 0x7C00u | (mant >> 13));
    }
    if (exp > 30) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static bool expect_resident(const char * label, const ggml_tensor * t,
                            size_t bytes, ggml_backend_buffer_t expected,
                            size_t expected_offset) {
    ggml_backend_buffer_t owner = NULL;
    size_t offset = 0;
    if (!ggml_cuda8_ggml_tensor_is_device_resident(t, bytes, &owner, &offset)) {
        std::fprintf(stderr, "%s: residency not found\n", label);
        return false;
    }
    if (owner != expected || offset != expected_offset) {
        std::fprintf(stderr, "%s: residency mismatch offset=%zu expected=%zu\n",
                     label, offset, expected_offset);
        return false;
    }
    std::printf("%s residency PASS offset=%zu\n", label, offset);
    return true;
}

static void force_1d_f32_data_layout(ggml_tensor * t, int64_t n, void * data) {
    t->type  = GGML_TYPE_F32;
    t->ne[0] = n; t->ne[1] = 1; t->ne[2] = 1; t->ne[3] = 1;
    t->nb[0] = sizeof(float);
    t->nb[1] = n * sizeof(float);
    t->nb[2] = t->nb[1];
    t->nb[3] = t->nb[1];
    t->data  = data;
}

static void force_2d_f32_data_layout(ggml_tensor * t, int64_t cols, int64_t rows, void * data) {
    t->type  = GGML_TYPE_F32;
    t->ne[0] = cols; t->ne[1] = rows; t->ne[2] = 1; t->ne[3] = 1;
    t->nb[0] = sizeof(float);
    t->nb[1] = cols * sizeof(float);
    t->nb[2] = rows * t->nb[1];
    t->nb[3] = t->nb[2];
    t->data  = data;
}

static void force_2d_q8_0_data_layout(ggml_tensor * t, int64_t cols, int64_t rows, void * data) {
    const int64_t bpr = (cols + GGML_CUDA8_QK8_0 - 1) / GGML_CUDA8_QK8_0;
    t->type  = GGML_TYPE_Q8_0;
    t->ne[0] = cols; t->ne[1] = rows; t->ne[2] = 1; t->ne[3] = 1;
    t->nb[0] = sizeof(ggml_cuda8_q8_0_block);
    t->nb[1] = (size_t)bpr * sizeof(ggml_cuda8_q8_0_block);
    t->nb[2] = (size_t)rows * t->nb[1];
    t->nb[3] = t->nb[2];
    t->data  = data;
}

static void pack_q8_0(const std::vector<float> & A,
                       std::vector<ggml_cuda8_q8_0_block> & Aq,
                       int rows, int cols) {
    const int bpr = (cols + GGML_CUDA8_QK8_0 - 1) / GGML_CUDA8_QK8_0;
    Aq.resize((size_t)rows * bpr);
    for (int r = 0; r < rows; ++r) {
        for (int ib = 0; ib < bpr; ++ib) {
            ggml_cuda8_q8_0_block & block = Aq[(size_t)r * bpr + ib];
            float max_abs = 0.0f;
            for (int k = 0; k < GGML_CUDA8_QK8_0; ++k) {
                const int c = ib * GGML_CUDA8_QK8_0 + k;
                const float v = (c < cols) ? A[(size_t)r * cols + c] : 0.0f;
                if (std::fabs(v) > max_abs) max_abs = std::fabs(v);
            }
            const float d = (max_abs > 0.0f) ? max_abs / 127.0f : 1.0f;
            const float id = 1.0f / d;
            block.d = fp32_to_fp16_bits(d);
            for (int k = 0; k < GGML_CUDA8_QK8_0; ++k) {
                const int c = ib * GGML_CUDA8_QK8_0 + k;
                const float v = (c < cols) ? A[(size_t)r * cols + c] : 0.0f;
                int q = (int)std::round(v * id);
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                block.qs[k] = (int8_t)q;
            }
        }
    }
}

// CPU reference: dequantize Q8_0 row, dot with F32 vector
static void mul_mat_q8_0_ref(
    const std::vector<ggml_cuda8_q8_0_block> & Aq,
    const std::vector<float> & x,
    std::vector<float> & y,
    int rows, int cols)
{
    const int bpr = (cols + GGML_CUDA8_QK8_0 - 1) / GGML_CUDA8_QK8_0;
    y.resize(rows);
    for (int r = 0; r < rows; ++r) {
        float sum = 0.0f;
        const ggml_cuda8_q8_0_block * row_blocks = &Aq[(size_t)r * bpr];
        for (int ib = 0; ib < bpr; ++ib) {
            const ggml_cuda8_q8_0_block & block = row_blocks[ib];
            // decode fp16 scale
            const uint16_t dh = block.d;
            const uint32_t sign = ((uint32_t)dh & 0x8000u) << 16;
            const uint32_t exponent = (uint32_t)((dh >> 10) & 0x1F);
            const uint32_t mantissa = (uint32_t)(dh & 0x03FF);
            uint32_t fbits;
            if (exponent == 0) {
                if (mantissa == 0) { fbits = sign; }
                else {
                    uint32_t m = mantissa;
                    int e = -1;
                    while ((m & 0x0400) == 0) { m <<= 1; e--; }
                    m &= ~0x0400u;
                    fbits = sign | ((uint32_t)(127 - 15 + 1 + e) << 23) | (m << 13);
                }
            } else if (exponent == 31) {
                fbits = sign | 0x7F800000u | (mantissa << 13);
            } else {
                fbits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
            }
            union { uint32_t u; float f; } conv;
            conv.u = fbits;
            const float d = conv.f;
            for (int k = 0; k < GGML_CUDA8_QK8_0; ++k) {
                const int c = ib * GGML_CUDA8_QK8_0 + k;
                if (c < cols) {
                    sum += d * (float)block.qs[k] * x[c];
                }
            }
        }
        y[r] = sum;
    }
}

int main() {
    std::printf("ggml-cuda8-ggml-backend-graph-builder-transformer-block-smoke: starting\n");

    // Dimensions
    const int embd  = 128;   // embedding / hidden dim
    const int proj   = 64;    // projection dim (e.g., head_dim)
    const float eps  = 1e-5f;
    const float scale_val = 0.125f;  // 1/sqrt(head_dim)

    const size_t vec_embd_bytes = (size_t)embd * sizeof(float);
    const size_t vec_proj_bytes = (size_t)proj * sizeof(float);
    const size_t scalar_bytes   = sizeof(float);

    const int bpr = (embd + GGML_CUDA8_QK8_0 - 1) / GGML_CUDA8_QK8_0;
    const size_t q8_bytes = (size_t)proj * bpr * sizeof(ggml_cuda8_q8_0_block);

    // -- ggml context ---------------------------------------------------------
    std::vector<uint8_t> ggml_mem(4 * 1024 * 1024);
    ggml_init_params ip;
    std::memset(&ip, 0, sizeof(ip));
    ip.mem_size   = ggml_mem.size();
    ip.mem_buffer = ggml_mem.data();
    ip.no_alloc   = false;

    ggml_context * gctx = ggml_init(ip);
    if (!gctx) { std::fprintf(stderr, "ggml_init failed\n"); return 1; }

    // -- build graph ----------------------------------------------------------
    // Inputs
    ggml_tensor * tx     = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, embd);   // input x
    ggml_tensor * tw     = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, embd);   // norm weights
    ggml_tensor * tWq    = ggml_new_tensor_2d(gctx, GGML_TYPE_Q8_0, embd, proj);  // W_q
    ggml_tensor * tbias  = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, proj);   // bias
    ggml_tensor * tscale = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 1);      // scale

    // Pipeline
    ggml_tensor * tnorm   = ggml_rms_norm(gctx, tx, eps);           // 1. RMS_NORM
    ggml_tensor * tnormed = ggml_mul(gctx, tnorm, tw);              // 2. MUL (elem)
    ggml_tensor * th      = ggml_mul_mat(gctx, tWq, tnormed);      // 3. MUL_MAT Q8_0
    ggml_tensor * thb     = ggml_add(gctx, th, tbias);              // 4. ADD bias
    ggml_tensor * ths     = ggml_mul(gctx, thb, tscale);            // 5. MUL scalar
    ggml_tensor * ty      = ggml_soft_max(gctx, ths);               // 6. SOFTMAX

    ggml_cgraph * graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(graph, ty);

    std::printf("real ggml graph n_nodes=%d n_leafs=%d\n",
                graph->n_nodes, graph->n_leafs);
    for (int i = 0; i < graph->n_nodes; ++i)
        std::printf("  node %d op=%d (%s)\n", i,
                    (int)graph->nodes[i]->op, ggml_op_name(graph->nodes[i]->op));

    // -- backend init ---------------------------------------------------------
    ggml_backend_t backend = ggml_cuda8_ggml_backend_init(0);
    if (!backend) { std::fprintf(stderr, "backend init failed\n"); return 1; }
    std::printf("backend name: %s\n", backend->iface.get_name(backend));

    ggml_backend_buffer_type_t buft =
        ggml_cuda8_ggml_backend_get_default_buffer_type(backend);

    // -- buffer layout --------------------------------------------------------
    // Generous alignment: 4096 per slot
    const size_t A = 4096;
    size_t slot = 0;
    const size_t off_x      = slot; slot += ((vec_embd_bytes + A - 1) / A) * A;
    const size_t off_w      = slot; slot += ((vec_embd_bytes + A - 1) / A) * A;
    const size_t off_Wq     = slot; slot += ((q8_bytes + A - 1) / A) * A;
    const size_t off_bias   = slot; slot += ((vec_proj_bytes + A - 1) / A) * A;
    const size_t off_scale  = slot; slot += A;  // single float, aligned
    const size_t off_norm   = slot; slot += ((vec_embd_bytes + A - 1) / A) * A;
    const size_t off_normed = slot; slot += ((vec_embd_bytes + A - 1) / A) * A;
    const size_t off_h      = slot; slot += ((vec_proj_bytes + A - 1) / A) * A;
    const size_t off_hb     = slot; slot += ((vec_proj_bytes + A - 1) / A) * A;
    const size_t off_hs     = slot; slot += ((vec_proj_bytes + A - 1) / A) * A;
    const size_t off_y      = slot; slot += ((vec_proj_bytes + A - 1) / A) * A;
    const size_t total      = slot;

    std::printf("buffer layout: total=%zu bytes\n", total);

    ggml_backend_buffer_t buffer = buft->iface.alloc_buffer(buft, total);
    if (!buffer) { std::fprintf(stderr, "alloc failed\n"); return 1; }
    uint8_t * base = (uint8_t *) buffer->iface.get_base(buffer);
    std::printf("buffer base: %p  size: %zu\n", (void*)base, buffer->size);

    // -- wire tensors to device buffer ----------------------------------------
    force_1d_f32_data_layout(tx,      embd, base + off_x);
    force_1d_f32_data_layout(tw,      embd, base + off_w);
    force_2d_q8_0_data_layout(tWq,    embd, proj, base + off_Wq);
    force_1d_f32_data_layout(tbias,   proj, base + off_bias);
    force_1d_f32_data_layout(tscale,  1,    base + off_scale);
    force_1d_f32_data_layout(tnorm,   embd, base + off_norm);
    force_1d_f32_data_layout(tnormed, embd, base + off_normed);
    force_1d_f32_data_layout(th,      proj, base + off_h);
    force_1d_f32_data_layout(thb,     proj, base + off_hb);
    force_1d_f32_data_layout(ths,     proj, base + off_hs);
    force_2d_f32_data_layout(ty,      proj, 1, base + off_y);

    // Register residency
    ggml_tensor * all_tensors[] = {tx, tw, tWq, tbias, tscale, tnorm, tnormed, th, thb, ths, ty};
    for (int i = 0; i < 11; ++i) {
        if (buffer->iface.init_tensor(buffer, all_tensors[i]) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "init_tensor failed for tensor %d\n", i);
            buffer->iface.free_buffer(buffer);
            backend->iface.free(backend);
            ggml_free(gctx); return 1;
        }
    }

    // -- prepare host data ----------------------------------------------------
    std::vector<float> h_x(embd), h_w(embd), h_bias(proj);
    float h_scale = scale_val;

    srand(42);
    for (int i = 0; i < embd; ++i) {
        h_x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_w[i] = ((float)rand() / RAND_MAX) * 0.5f + 0.75f;  // norm weights ~1.0
    }
    for (int i = 0; i < proj; ++i)
        h_bias[i] = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;

    // Weight matrix (random, then quantize)
    std::vector<float> W_f32((size_t)proj * embd);
    for (int r = 0; r < proj; ++r)
        for (int c = 0; c < embd; ++c)
            W_f32[(size_t)r * embd + c] = ((float)((r*17 + c*31 + 7) % 251) - 125.0f) * 0.003f;

    std::vector<ggml_cuda8_q8_0_block> Wq_host;
    pack_q8_0(W_f32, Wq_host, proj, embd);

    // -- CPU reference --------------------------------------------------------
    // 1. RMS_NORM
    float ss = 0.0f;
    for (int i = 0; i < embd; ++i) ss += h_x[i] * h_x[i];
    float rms_scale = 1.0f / std::sqrt(ss / (float)embd + eps);
    std::vector<float> h_norm(embd);
    for (int i = 0; i < embd; ++i) h_norm[i] = h_x[i] * rms_scale;

    // 2. MUL (elem)
    std::vector<float> h_normed(embd);
    for (int i = 0; i < embd; ++i) h_normed[i] = h_norm[i] * h_w[i];

    // 3. MUL_MAT Q8_0
    std::vector<float> h_h;
    mul_mat_q8_0_ref(Wq_host, h_normed, h_h, proj, embd);

    // 4. ADD bias
    std::vector<float> h_hb(proj);
    for (int i = 0; i < proj; ++i) h_hb[i] = h_h[i] + h_bias[i];

    // 5. MUL scalar
    std::vector<float> h_hs(proj);
    for (int i = 0; i < proj; ++i) h_hs[i] = h_hb[i] * h_scale;

    // 6. SOFTMAX
    float max_val = h_hs[0];
    for (int i = 1; i < proj; ++i) if (h_hs[i] > max_val) max_val = h_hs[i];
    std::vector<float> h_ref(proj);
    float sum_exp = 0.0f;
    for (int i = 0; i < proj; ++i) {
        h_ref[i] = std::exp(h_hs[i] - max_val);
        sum_exp += h_ref[i];
    }
    for (int i = 0; i < proj; ++i) h_ref[i] /= sum_exp;

    // -- upload ---------------------------------------------------------------
    buffer->iface.clear(buffer, 0);
    buffer->iface.set_tensor(buffer, tx,     &h_x[0],     0, vec_embd_bytes);
    buffer->iface.set_tensor(buffer, tw,     &h_w[0],     0, vec_embd_bytes);
    buffer->iface.set_tensor(buffer, tWq,    &Wq_host[0], 0, q8_bytes);
    buffer->iface.set_tensor(buffer, tbias,  &h_bias[0],  0, vec_proj_bytes);
    buffer->iface.set_tensor(buffer, tscale, &h_scale,    0, scalar_bytes);

    // -- compute --------------------------------------------------------------
    std::printf("dispatching graph_compute...\n");
    enum ggml_status status = backend->iface.graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "graph_compute returned %d\n", (int)status);
        buffer->iface.free_buffer(buffer);
        backend->iface.free(backend);
        ggml_free(gctx); return 1;
    }

    // -- download + verify ----------------------------------------------------
    std::vector<float> h_y(proj, 0.0f);
    buffer->iface.get_tensor(buffer, ty, &h_y[0], 0, vec_proj_bytes);

    float max_err = 0.0f;
    for (int i = 0; i < proj; ++i) {
        float e = std::fabs(h_y[i] - h_ref[i]);
        if (e > max_err) max_err = e;
    }

    // Softmax sum should be ~1.0
    float gpu_sum = 0.0f;
    for (int i = 0; i < proj; ++i) gpu_sum += h_y[i];

    std::printf("transformer block pipeline:\n");
    std::printf("  embd=%d proj=%d eps=%.1e scale=%.3f\n", embd, proj, eps, scale_val);
    std::printf("  max_err=%.6e  softmax_sum=%.9f\n", max_err, gpu_sum);

    bool pass = (max_err < 1e-3f) && (std::fabs(gpu_sum - 1.0f) < 1e-4f);
    std::printf("transformer block pipeline: %s\n", pass ? "PASS" : "FAIL");

    // -- cleanup --------------------------------------------------------------
    buffer->iface.free_buffer(buffer);
    backend->iface.free(backend);
    ggml_free(gctx);

    std::printf("ggml-cuda8-ggml-backend-graph-builder-transformer-block-smoke: %s\n",
                pass ? "SUCCESS" : "FAIL");
    return pass ? 0 : 1;
}
