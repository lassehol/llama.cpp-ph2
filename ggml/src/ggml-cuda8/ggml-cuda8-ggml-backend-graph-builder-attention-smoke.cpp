// ggml-cuda8-ggml-backend-graph-builder-attention-smoke.cpp
// G34A: Full single-token attention pipeline through CUDA8 graph_compute
//
// 11-op pipeline (7 op types):
//   1. GET_ROWS(embed, token_id)       -> x [128]
//   2. RMS_NORM(x, eps)                -> norm [128]
//   3. MUL(norm, w_norm)               -> x_n [128]       (elem-wise)
//   4. MUL_MAT(W_q[64x128], x_n)      -> q [64]          (Q8_0)
//   5. MUL_MAT(W_k[64x128], x_n)      -> k [64]          (Q8_0)
//   6. MUL_MAT(W_v[64x128], x_n)      -> v [64]          (Q8_0)
//   7. MUL(q, scale)                   -> q_s [64]        (scalar)
//   8. ADD(q_s, k)                     -> scores [64]     (Q+K interaction)
//   9. SOFTMAX(scores)                 -> probs [64]
//  10. MUL(probs, v)                   -> attn [64]       (elem-wise)
//  11. ADD(attn, bias)                 -> out [64]
//
// Exercises: GET_ROWS, RMS_NORM, MUL_F32, MUL_MAT_Q8_0 (x3),
//            MUL_SCALAR, ADD_F32 (x2), SOFTMAX

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

// -- helpers ------------------------------------------------------------------

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
    if (exp == 0xFF - 127 + 15)
        return (mant == 0) ? (uint16_t)(sign | 0x7C00u)
                           : (uint16_t)(sign | 0x7C00u | (mant >> 13));
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
    return true;
}

static void force_1d_f32(ggml_tensor * t, int64_t n, void * data) {
    t->type = GGML_TYPE_F32;
    t->ne[0] = n; t->ne[1] = 1; t->ne[2] = 1; t->ne[3] = 1;
    t->nb[0] = sizeof(float);
    t->nb[1] = n * sizeof(float);
    t->nb[2] = t->nb[1]; t->nb[3] = t->nb[1];
    t->data = data;
}

static void force_2d_f32(ggml_tensor * t, int64_t cols, int64_t rows, void * data) {
    t->type = GGML_TYPE_F32;
    t->ne[0] = cols; t->ne[1] = rows; t->ne[2] = 1; t->ne[3] = 1;
    t->nb[0] = sizeof(float);
    t->nb[1] = cols * sizeof(float);
    t->nb[2] = rows * t->nb[1]; t->nb[3] = t->nb[2];
    t->data = data;
}

static void force_1d_i32(ggml_tensor * t, int64_t n, void * data) {
    t->type = GGML_TYPE_I32;
    t->ne[0] = n; t->ne[1] = 1; t->ne[2] = 1; t->ne[3] = 1;
    t->nb[0] = sizeof(int32_t);
    t->nb[1] = n * sizeof(int32_t);
    t->nb[2] = t->nb[1]; t->nb[3] = t->nb[1];
    t->data = data;
}

static void force_2d_q8_0(ggml_tensor * t, int64_t cols, int64_t rows, void * data) {
    const int64_t bpr = (cols + GGML_CUDA8_QK8_0 - 1) / GGML_CUDA8_QK8_0;
    t->type = GGML_TYPE_Q8_0;
    t->ne[0] = cols; t->ne[1] = rows; t->ne[2] = 1; t->ne[3] = 1;
    t->nb[0] = sizeof(ggml_cuda8_q8_0_block);
    t->nb[1] = (size_t)bpr * sizeof(ggml_cuda8_q8_0_block);
    t->nb[2] = (size_t)rows * t->nb[1]; t->nb[3] = t->nb[2];
    t->data = data;
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

static float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    const uint32_t exp = (uint32_t)((h >> 10) & 0x1F);
    const uint32_t mant = (uint32_t)(h & 0x03FF);
    uint32_t fbits;
    if (exp == 0) {
        if (mant == 0) { fbits = sign; }
        else {
            uint32_t m = mant; int e = -1;
            while ((m & 0x0400) == 0) { m <<= 1; e--; }
            m &= ~0x0400u;
            fbits = sign | ((uint32_t)(127 - 15 + 1 + e) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        fbits = sign | 0x7F800000u | (mant << 13);
    } else {
        fbits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    union { uint32_t u; float f; } conv;
    conv.u = fbits;
    return conv.f;
}

static void mul_mat_q8_0_ref(
    const std::vector<ggml_cuda8_q8_0_block> & Aq,
    const std::vector<float> & x,
    std::vector<float> & y, int rows, int cols) {
    const int bpr = (cols + GGML_CUDA8_QK8_0 - 1) / GGML_CUDA8_QK8_0;
    y.resize(rows);
    for (int r = 0; r < rows; ++r) {
        float sum = 0.0f;
        const ggml_cuda8_q8_0_block * rb = &Aq[(size_t)r * bpr];
        for (int ib = 0; ib < bpr; ++ib) {
            const float d = fp16_to_fp32(rb[ib].d);
            for (int k = 0; k < GGML_CUDA8_QK8_0; ++k) {
                const int c = ib * GGML_CUDA8_QK8_0 + k;
                if (c < cols) sum += d * (float)rb[ib].qs[k] * x[c];
            }
        }
        y[r] = sum;
    }
}

// -- main ---------------------------------------------------------------------

int main() {
    std::printf("ggml-cuda8-attention-smoke: starting\n");

    // Dimensions
    const int vocab = 32;
    const int embd  = 128;
    const int proj  = 64;   // head_dim
    const float eps = 1e-5f;
    const float scale_val = 1.0f / std::sqrt((float)proj);  // 0.125
    const int token_id = 5;

    const size_t embd_bytes  = (size_t)embd * sizeof(float);
    const size_t proj_bytes  = (size_t)proj * sizeof(float);
    const size_t embed_bytes = (size_t)vocab * embd * sizeof(float);
    const size_t tok_bytes   = sizeof(int32_t);
    const size_t scalar_bytes = sizeof(float);
    const int bpr = (embd + GGML_CUDA8_QK8_0 - 1) / GGML_CUDA8_QK8_0;
    const size_t q8_bytes = (size_t)proj * bpr * sizeof(ggml_cuda8_q8_0_block);

    // -- ggml context ---------------------------------------------------------
    std::vector<uint8_t> ggml_mem(8 * 1024 * 1024);
    ggml_init_params ip;
    std::memset(&ip, 0, sizeof(ip));
    ip.mem_size = ggml_mem.size();
    ip.mem_buffer = ggml_mem.data();
    ip.no_alloc = false;

    ggml_context * gctx = ggml_init(ip);
    if (!gctx) { std::fprintf(stderr, "ggml_init failed\n"); return 1; }

    // -- build graph ----------------------------------------------------------
    // Leaf tensors
    ggml_tensor * t_embed = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, embd, vocab);
    ggml_tensor * t_tok   = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    ggml_tensor * t_wnorm = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, embd);
    ggml_tensor * t_Wq    = ggml_new_tensor_2d(gctx, GGML_TYPE_Q8_0, embd, proj);
    ggml_tensor * t_Wk    = ggml_new_tensor_2d(gctx, GGML_TYPE_Q8_0, embd, proj);
    ggml_tensor * t_Wv    = ggml_new_tensor_2d(gctx, GGML_TYPE_Q8_0, embd, proj);
    ggml_tensor * t_scale = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 1);
    ggml_tensor * t_bias  = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, proj);

    // Pipeline: 11 ops
    ggml_tensor * t_x      = ggml_get_rows(gctx, t_embed, t_tok);    // 1. GET_ROWS
    ggml_tensor * t_norm   = ggml_rms_norm(gctx, t_x, eps);          // 2. RMS_NORM
    ggml_tensor * t_xn     = ggml_mul(gctx, t_norm, t_wnorm);        // 3. MUL elem
    ggml_tensor * t_q      = ggml_mul_mat(gctx, t_Wq, t_xn);        // 4. Q projection
    ggml_tensor * t_k      = ggml_mul_mat(gctx, t_Wk, t_xn);        // 5. K projection
    ggml_tensor * t_v      = ggml_mul_mat(gctx, t_Wv, t_xn);        // 6. V projection
    ggml_tensor * t_qs     = ggml_mul(gctx, t_q, t_scale);           // 7. scale Q
    ggml_tensor * t_scores = ggml_add(gctx, t_qs, t_k);              // 8. Q+K scores
    ggml_tensor * t_probs  = ggml_soft_max(gctx, t_scores);          // 9. SOFTMAX
    ggml_tensor * t_attn   = ggml_mul(gctx, t_probs, t_v);           // 10. probs*V
    ggml_tensor * t_out    = ggml_add(gctx, t_attn, t_bias);         // 11. +bias

    ggml_cgraph * graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(graph, t_out);

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
    const size_t A = 4096;  // alignment
    size_t slot = 0;
    const size_t off_embed  = slot; slot += ((embed_bytes + A-1)/A)*A;
    const size_t off_tok    = slot; slot += A;
    const size_t off_wnorm  = slot; slot += ((embd_bytes + A-1)/A)*A;
    const size_t off_Wq     = slot; slot += ((q8_bytes + A-1)/A)*A;
    const size_t off_Wk     = slot; slot += ((q8_bytes + A-1)/A)*A;
    const size_t off_Wv     = slot; slot += ((q8_bytes + A-1)/A)*A;
    const size_t off_scale  = slot; slot += A;
    const size_t off_bias   = slot; slot += ((proj_bytes + A-1)/A)*A;
    const size_t off_x      = slot; slot += ((embd_bytes + A-1)/A)*A;
    const size_t off_norm   = slot; slot += ((embd_bytes + A-1)/A)*A;
    const size_t off_xn     = slot; slot += ((embd_bytes + A-1)/A)*A;
    const size_t off_q      = slot; slot += ((proj_bytes + A-1)/A)*A;
    const size_t off_k      = slot; slot += ((proj_bytes + A-1)/A)*A;
    const size_t off_v      = slot; slot += ((proj_bytes + A-1)/A)*A;
    const size_t off_qs     = slot; slot += ((proj_bytes + A-1)/A)*A;
    const size_t off_scores = slot; slot += ((proj_bytes + A-1)/A)*A;
    const size_t off_probs  = slot; slot += ((proj_bytes + A-1)/A)*A;
    const size_t off_attn   = slot; slot += ((proj_bytes + A-1)/A)*A;
    const size_t off_out    = slot; slot += ((proj_bytes + A-1)/A)*A;
    const size_t total      = slot;

    std::printf("buffer layout: total=%zu bytes (%zu KB)\n", total, total/1024);

    ggml_backend_buffer_t buffer = buft->iface.alloc_buffer(buft, total);
    if (!buffer) { std::fprintf(stderr, "alloc failed\n"); return 1; }
    uint8_t * base = (uint8_t *) buffer->iface.get_base(buffer);

    // -- wire tensors to device buffer ----------------------------------------
    force_2d_f32(t_embed, embd, vocab, base + off_embed);
    force_1d_i32(t_tok, 1, base + off_tok);
    force_1d_f32(t_wnorm, embd, base + off_wnorm);
    force_2d_q8_0(t_Wq, embd, proj, base + off_Wq);
    force_2d_q8_0(t_Wk, embd, proj, base + off_Wk);
    force_2d_q8_0(t_Wv, embd, proj, base + off_Wv);
    force_1d_f32(t_scale, 1, base + off_scale);
    force_1d_f32(t_bias, proj, base + off_bias);
    force_1d_f32(t_x, embd, base + off_x);
    force_1d_f32(t_norm, embd, base + off_norm);
    force_1d_f32(t_xn, embd, base + off_xn);
    force_1d_f32(t_q, proj, base + off_q);
    force_1d_f32(t_k, proj, base + off_k);
    force_1d_f32(t_v, proj, base + off_v);
    force_1d_f32(t_qs, proj, base + off_qs);
    force_1d_f32(t_scores, proj, base + off_scores);
    force_2d_f32(t_probs, proj, 1, base + off_probs);
    force_1d_f32(t_attn, proj, base + off_attn);
    force_1d_f32(t_out, proj, base + off_out);

    // Register residency for all 19 tensors
    ggml_tensor * all[] = {t_embed, t_tok, t_wnorm, t_Wq, t_Wk, t_Wv, t_scale, t_bias,
                           t_x, t_norm, t_xn, t_q, t_k, t_v, t_qs, t_scores, t_probs, t_attn, t_out};
    for (int i = 0; i < 19; ++i) {
        if (buffer->iface.init_tensor(buffer, all[i]) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "init_tensor %d failed\n", i);
            buffer->iface.free_buffer(buffer);
            backend->iface.free(backend);
            ggml_free(gctx); return 1;
        }
    }

    // -- prepare host data ----------------------------------------------------
    // Embedding table
    std::vector<float> h_embed((size_t)vocab * embd);
    srand(42);
    for (int i = 0; i < vocab * embd; ++i)
        h_embed[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    int32_t h_tok = token_id;

    // Norm weights (~1.0)
    std::vector<float> h_wnorm(embd);
    for (int i = 0; i < embd; ++i)
        h_wnorm[i] = ((float)rand() / RAND_MAX) * 0.5f + 0.75f;

    // Q/K/V weight matrices (random F32 -> quantized Q8_0)
    std::vector<float> Wq_f32((size_t)proj * embd);
    std::vector<float> Wk_f32((size_t)proj * embd);
    std::vector<float> Wv_f32((size_t)proj * embd);
    for (int r = 0; r < proj; ++r)
        for (int c = 0; c < embd; ++c) {
            Wq_f32[(size_t)r*embd+c] = ((float)((r*17+c*31+7) % 251) - 125.0f) * 0.003f;
            Wk_f32[(size_t)r*embd+c] = ((float)((r*19+c*37+11) % 251) - 125.0f) * 0.003f;
            Wv_f32[(size_t)r*embd+c] = ((float)((r*23+c*41+13) % 251) - 125.0f) * 0.003f;
        }

    std::vector<ggml_cuda8_q8_0_block> Wq_q8, Wk_q8, Wv_q8;
    pack_q8_0(Wq_f32, Wq_q8, proj, embd);
    pack_q8_0(Wk_f32, Wk_q8, proj, embd);
    pack_q8_0(Wv_f32, Wv_q8, proj, embd);

    float h_scale = scale_val;

    std::vector<float> h_bias(proj);
    for (int i = 0; i < proj; ++i)
        h_bias[i] = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;

    // -- CPU reference --------------------------------------------------------
    // 1. GET_ROWS: copy row token_id from embed
    std::vector<float> ref_x(embd);
    for (int i = 0; i < embd; ++i)
        ref_x[i] = h_embed[(size_t)token_id * embd + i];

    // 2. RMS_NORM
    float ss = 0.0f;
    for (int i = 0; i < embd; ++i) ss += ref_x[i] * ref_x[i];
    float rms = 1.0f / std::sqrt(ss / (float)embd + eps);
    std::vector<float> ref_norm(embd);
    for (int i = 0; i < embd; ++i) ref_norm[i] = ref_x[i] * rms;

    // 3. MUL elem
    std::vector<float> ref_xn(embd);
    for (int i = 0; i < embd; ++i) ref_xn[i] = ref_norm[i] * h_wnorm[i];

    // 4-6. Q/K/V MUL_MAT
    std::vector<float> ref_q, ref_k, ref_v;
    mul_mat_q8_0_ref(Wq_q8, ref_xn, ref_q, proj, embd);
    mul_mat_q8_0_ref(Wk_q8, ref_xn, ref_k, proj, embd);
    mul_mat_q8_0_ref(Wv_q8, ref_xn, ref_v, proj, embd);

    // 7. scale Q
    std::vector<float> ref_qs(proj);
    for (int i = 0; i < proj; ++i) ref_qs[i] = ref_q[i] * h_scale;

    // 8. ADD (Q+K scores)
    std::vector<float> ref_scores(proj);
    for (int i = 0; i < proj; ++i) ref_scores[i] = ref_qs[i] + ref_k[i];

    // 9. SOFTMAX
    float max_s = ref_scores[0];
    for (int i = 1; i < proj; ++i) if (ref_scores[i] > max_s) max_s = ref_scores[i];
    std::vector<float> ref_probs(proj);
    float sum_exp = 0.0f;
    for (int i = 0; i < proj; ++i) {
        ref_probs[i] = std::exp(ref_scores[i] - max_s);
        sum_exp += ref_probs[i];
    }
    for (int i = 0; i < proj; ++i) ref_probs[i] /= sum_exp;

    // 10. MUL elem (probs * V)
    std::vector<float> ref_attn(proj);
    for (int i = 0; i < proj; ++i) ref_attn[i] = ref_probs[i] * ref_v[i];

    // 11. ADD bias
    std::vector<float> ref_out(proj);
    for (int i = 0; i < proj; ++i) ref_out[i] = ref_attn[i] + h_bias[i];

    // -- upload ---------------------------------------------------------------
    buffer->iface.clear(buffer, 0);
    buffer->iface.set_tensor(buffer, t_embed, &h_embed[0], 0, embed_bytes);
    buffer->iface.set_tensor(buffer, t_tok,   &h_tok,      0, tok_bytes);
    buffer->iface.set_tensor(buffer, t_wnorm, &h_wnorm[0], 0, embd_bytes);
    buffer->iface.set_tensor(buffer, t_Wq,    &Wq_q8[0],   0, q8_bytes);
    buffer->iface.set_tensor(buffer, t_Wk,    &Wk_q8[0],   0, q8_bytes);
    buffer->iface.set_tensor(buffer, t_Wv,    &Wv_q8[0],   0, q8_bytes);
    buffer->iface.set_tensor(buffer, t_scale, &h_scale,     0, scalar_bytes);
    buffer->iface.set_tensor(buffer, t_bias,  &h_bias[0],   0, proj_bytes);

    // -- compute --------------------------------------------------------------
    std::printf("dispatching graph_compute (%d nodes)...\n", graph->n_nodes);
    enum ggml_status status = backend->iface.graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "graph_compute returned %d\n", (int)status);
        buffer->iface.free_buffer(buffer);
        backend->iface.free(backend);
        ggml_free(gctx); return 1;
    }

    // -- verify ---------------------------------------------------------------
    std::vector<float> h_out(proj, 0.0f);
    buffer->iface.get_tensor(buffer, t_out, &h_out[0], 0, proj_bytes);

    float max_err = 0.0f;
    for (int i = 0; i < proj; ++i) {
        float e = std::fabs(h_out[i] - ref_out[i]);
        if (e > max_err) max_err = e;
    }

    // Also verify softmax sum
    std::vector<float> h_probs(proj, 0.0f);
    buffer->iface.get_tensor(buffer, t_probs, &h_probs[0], 0, proj_bytes);
    float gpu_sum = 0.0f;
    for (int i = 0; i < proj; ++i) gpu_sum += h_probs[i];

    std::printf("attention pipeline results:\n");
    std::printf("  vocab=%d embd=%d proj=%d token_id=%d\n", vocab, embd, proj, token_id);
    std::printf("  max_err=%.6e  softmax_sum=%.9f\n", max_err, gpu_sum);

    bool pass = (max_err < 1e-3f) && (std::fabs(gpu_sum - 1.0f) < 1e-4f);
    std::printf("attention pipeline: %s\n", pass ? "PASS" : "FAIL");

    buffer->iface.free_buffer(buffer);
    backend->iface.free(backend);
    ggml_free(gctx);

    std::printf("ggml-cuda8-attention-smoke: %s\n", pass ? "SUCCESS" : "FAIL");
    return pass ? 0 : 1;
}
