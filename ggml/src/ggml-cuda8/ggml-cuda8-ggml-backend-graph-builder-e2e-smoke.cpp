// ggml-cuda8-ggml-backend-graph-builder-e2e-smoke.cpp
// G35A: Full end-to-end pipeline -- ALL ops in one graph_compute
//
// 15-op pipeline (9 op types):
//   1.  GET_ROWS(embed, tok)           -> emb [128]
//   2.  RMS_NORM(emb, eps)             -> norm [128]
//   3.  MUL(norm, w_norm)              -> x_n [128]    (elem)
//   4.  MUL_MAT(Wq[64x128], x_n)      -> q [64]       (Q8_0)
//   5.  MUL_MAT(Wk[64x128], x_n)      -> k [64]       (Q8_0)
//   6.  MUL_MAT(Wv[64x128], x_n)      -> v [64]       (Q8_0)
//   7.  ROPE(q, pos=7, n_dims=64)      -> q_r [64]
//   8.  ROPE(k, pos=7, n_dims=64)      -> k_r [64]
//   9.  MUL(q_r, scale)               -> q_s [64]      (scalar)
//  10.  ADD(q_s, k_r)                  -> scores [64]
//  11.  DIAG_MASK_INF(scores, n_past=31) -> masked [64]
//  12.  SOFTMAX(masked)                -> probs [64]
//  13.  MUL(probs, v)                  -> attn [64]     (elem)
//  14.  CONT(attn)                     -> attn_c [64]
//  15.  ADD(attn_c, bias)              -> out [64]

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
#include <float.h>
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

static void rope_ref(const std::vector<float> & in, std::vector<float> & out,
                      int n_dims, int pos, float freq_base, float freq_scale) {
    out.resize(n_dims);
    float theta_scale = std::pow(freq_base, -2.0f / (float)n_dims);
    for (int pair = 0; pair < n_dims / 2; ++pair) {
        float theta = (float)pos * std::pow(theta_scale, (float)pair) * freq_scale;
        float cos_t = std::cos(theta);
        float sin_t = std::sin(theta);
        float x0 = in[pair * 2];
        float x1 = in[pair * 2 + 1];
        out[pair * 2]     = x0 * cos_t - x1 * sin_t;
        out[pair * 2 + 1] = x0 * sin_t + x1 * cos_t;
    }
}

// -- main ---------------------------------------------------------------------

int main() {
    std::printf("ggml-cuda8-e2e-smoke: starting\n");

    const int vocab = 32;
    const int embd  = 128;
    const int proj  = 64;
    const float eps = 1e-5f;
    const float scale_val = 1.0f / std::sqrt((float)proj);
    const int token_id = 5;
    const int rope_pos = 7;
    const int n_past = 31;  // causal mask: keep cols 0-31, mask 32-63
    const float freq_base  = 10000.0f;
    const float freq_scale = 1.0f;

    const size_t embd_bytes  = (size_t)embd * sizeof(float);
    const size_t proj_bytes  = (size_t)proj * sizeof(float);
    const size_t embed_bytes = (size_t)vocab * embd * sizeof(float);
    const size_t tok_bytes   = sizeof(int32_t);
    const size_t pos_bytes   = sizeof(int32_t);
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

    // -- leaf tensors ---------------------------------------------------------
    ggml_tensor * t_embed = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, embd, vocab);
    ggml_tensor * t_tok   = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    ggml_tensor * t_wnorm = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, embd);
    ggml_tensor * t_Wq    = ggml_new_tensor_2d(gctx, GGML_TYPE_Q8_0, embd, proj);
    ggml_tensor * t_Wk    = ggml_new_tensor_2d(gctx, GGML_TYPE_Q8_0, embd, proj);
    ggml_tensor * t_Wv    = ggml_new_tensor_2d(gctx, GGML_TYPE_Q8_0, embd, proj);
    ggml_tensor * t_scale = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 1);
    ggml_tensor * t_bias  = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, proj);
    ggml_tensor * t_pos   = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);

    // -- build graph (15 ops) -------------------------------------------------
    ggml_tensor * t_emb    = ggml_get_rows(gctx, t_embed, t_tok);     //  1
    ggml_tensor * t_norm   = ggml_rms_norm(gctx, t_emb, eps);         //  2
    ggml_tensor * t_xn     = ggml_mul(gctx, t_norm, t_wnorm);         //  3
    ggml_tensor * t_q      = ggml_mul_mat(gctx, t_Wq, t_xn);         //  4
    ggml_tensor * t_k      = ggml_mul_mat(gctx, t_Wk, t_xn);         //  5
    ggml_tensor * t_v      = ggml_mul_mat(gctx, t_Wv, t_xn);         //  6
    ggml_tensor * t_qr     = ggml_rope(gctx, t_q, t_pos, proj, 0);   //  7
    ggml_tensor * t_kr     = ggml_rope(gctx, t_k, t_pos, proj, 0);   //  8
    ggml_tensor * t_qs     = ggml_mul(gctx, t_qr, t_scale);           //  9
    ggml_tensor * t_scores = ggml_add(gctx, t_qs, t_kr);              // 10
    ggml_tensor * t_masked = ggml_diag_mask_inf(gctx, t_scores, n_past); // 11
    ggml_tensor * t_probs  = ggml_soft_max(gctx, t_masked);           // 12
    ggml_tensor * t_attn   = ggml_mul(gctx, t_probs, t_v);            // 13
    ggml_tensor * t_attnc  = ggml_cont(gctx, t_attn);                 // 14
    ggml_tensor * t_out    = ggml_add(gctx, t_attnc, t_bias);         // 15

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
    const size_t A = 4096;
    size_t slot = 0;
    // Leaves (9)
    const size_t off_embed = slot; slot += ((embed_bytes+A-1)/A)*A;
    const size_t off_tok   = slot; slot += A;
    const size_t off_wnorm = slot; slot += ((embd_bytes+A-1)/A)*A;
    const size_t off_Wq    = slot; slot += ((q8_bytes+A-1)/A)*A;
    const size_t off_Wk    = slot; slot += ((q8_bytes+A-1)/A)*A;
    const size_t off_Wv    = slot; slot += ((q8_bytes+A-1)/A)*A;
    const size_t off_scale = slot; slot += A;
    const size_t off_bias  = slot; slot += ((proj_bytes+A-1)/A)*A;
    const size_t off_pos   = slot; slot += A;
    // Intermediates (15)
    const size_t off_emb    = slot; slot += ((embd_bytes+A-1)/A)*A;
    const size_t off_norm   = slot; slot += ((embd_bytes+A-1)/A)*A;
    const size_t off_xn     = slot; slot += ((embd_bytes+A-1)/A)*A;
    const size_t off_q      = slot; slot += ((proj_bytes+A-1)/A)*A;
    const size_t off_k      = slot; slot += ((proj_bytes+A-1)/A)*A;
    const size_t off_v      = slot; slot += ((proj_bytes+A-1)/A)*A;
    const size_t off_qr     = slot; slot += ((proj_bytes+A-1)/A)*A;
    const size_t off_kr     = slot; slot += ((proj_bytes+A-1)/A)*A;
    const size_t off_qs     = slot; slot += ((proj_bytes+A-1)/A)*A;
    const size_t off_scores = slot; slot += ((proj_bytes+A-1)/A)*A;
    const size_t off_masked = slot; slot += ((proj_bytes+A-1)/A)*A;
    const size_t off_probs  = slot; slot += ((proj_bytes+A-1)/A)*A;
    const size_t off_attn   = slot; slot += ((proj_bytes+A-1)/A)*A;
    const size_t off_attnc  = slot; slot += ((proj_bytes+A-1)/A)*A;
    const size_t off_out    = slot; slot += ((proj_bytes+A-1)/A)*A;
    const size_t total      = slot;

    std::printf("buffer layout: total=%zu bytes (%zu KB)\n", total, total/1024);

    ggml_backend_buffer_t buffer = buft->iface.alloc_buffer(buft, total);
    if (!buffer) { std::fprintf(stderr, "alloc failed\n"); return 1; }
    uint8_t * base = (uint8_t *) buffer->iface.get_base(buffer);

    // -- wire tensors ---------------------------------------------------------
    force_2d_f32(t_embed, embd, vocab, base + off_embed);
    force_1d_i32(t_tok, 1, base + off_tok);
    force_1d_f32(t_wnorm, embd, base + off_wnorm);
    force_2d_q8_0(t_Wq, embd, proj, base + off_Wq);
    force_2d_q8_0(t_Wk, embd, proj, base + off_Wk);
    force_2d_q8_0(t_Wv, embd, proj, base + off_Wv);
    force_1d_f32(t_scale, 1, base + off_scale);
    force_1d_f32(t_bias, proj, base + off_bias);
    force_1d_i32(t_pos, 1, base + off_pos);
    force_1d_f32(t_emb, embd, base + off_emb);
    force_1d_f32(t_norm, embd, base + off_norm);
    force_1d_f32(t_xn, embd, base + off_xn);
    force_1d_f32(t_q, proj, base + off_q);
    force_1d_f32(t_k, proj, base + off_k);
    force_1d_f32(t_v, proj, base + off_v);
    force_1d_f32(t_qr, proj, base + off_qr);
    force_1d_f32(t_kr, proj, base + off_kr);
    force_1d_f32(t_qs, proj, base + off_qs);
    force_1d_f32(t_scores, proj, base + off_scores);
    force_1d_f32(t_masked, proj, base + off_masked);
    force_2d_f32(t_probs, proj, 1, base + off_probs);
    force_1d_f32(t_attn, proj, base + off_attn);
    force_1d_f32(t_attnc, proj, base + off_attnc);
    force_1d_f32(t_out, proj, base + off_out);

    // Register residency for all 24 tensors
    ggml_tensor * all[] = {
        t_embed, t_tok, t_wnorm, t_Wq, t_Wk, t_Wv, t_scale, t_bias, t_pos,
        t_emb, t_norm, t_xn, t_q, t_k, t_v, t_qr, t_kr, t_qs,
        t_scores, t_masked, t_probs, t_attn, t_attnc, t_out
    };
    for (int i = 0; i < 24; ++i) {
        if (buffer->iface.init_tensor(buffer, all[i]) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "init_tensor %d failed\n", i);
            buffer->iface.free_buffer(buffer);
            backend->iface.free(backend);
            ggml_free(gctx); return 1;
        }
    }

    // -- host data ------------------------------------------------------------
    std::vector<float> h_embed((size_t)vocab * embd);
    srand(42);
    for (int i = 0; i < vocab * embd; ++i)
        h_embed[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    int32_t h_tok = token_id;
    int32_t h_pos = rope_pos;
    float h_scale = scale_val;

    std::vector<float> h_wnorm(embd);
    for (int i = 0; i < embd; ++i)
        h_wnorm[i] = ((float)rand() / RAND_MAX) * 0.5f + 0.75f;

    std::vector<float> Wq_f32((size_t)proj * embd);
    std::vector<float> Wk_f32((size_t)proj * embd);
    std::vector<float> Wv_f32((size_t)proj * embd);
    for (int r = 0; r < proj; ++r)
        for (int c = 0; c < embd; ++c) {
            Wq_f32[(size_t)r*embd+c] = ((float)((r*17+c*31+7)  % 251) - 125.0f) * 0.003f;
            Wk_f32[(size_t)r*embd+c] = ((float)((r*19+c*37+11) % 251) - 125.0f) * 0.003f;
            Wv_f32[(size_t)r*embd+c] = ((float)((r*23+c*41+13) % 251) - 125.0f) * 0.003f;
        }

    std::vector<ggml_cuda8_q8_0_block> Wq_q8, Wk_q8, Wv_q8;
    pack_q8_0(Wq_f32, Wq_q8, proj, embd);
    pack_q8_0(Wk_f32, Wk_q8, proj, embd);
    pack_q8_0(Wv_f32, Wv_q8, proj, embd);

    std::vector<float> h_bias(proj);
    for (int i = 0; i < proj; ++i)
        h_bias[i] = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;

    // -- CPU reference (15 steps) ---------------------------------------------
    // 1. GET_ROWS
    std::vector<float> ref_emb(embd);
    for (int i = 0; i < embd; ++i)
        ref_emb[i] = h_embed[(size_t)token_id * embd + i];

    // 2. RMS_NORM
    float ss = 0.0f;
    for (int i = 0; i < embd; ++i) ss += ref_emb[i] * ref_emb[i];
    float rms = 1.0f / std::sqrt(ss / (float)embd + eps);
    std::vector<float> ref_norm(embd);
    for (int i = 0; i < embd; ++i) ref_norm[i] = ref_emb[i] * rms;

    // 3. MUL (elem)
    std::vector<float> ref_xn(embd);
    for (int i = 0; i < embd; ++i) ref_xn[i] = ref_norm[i] * h_wnorm[i];

    // 4-6. Q/K/V MUL_MAT
    std::vector<float> ref_q, ref_k, ref_v;
    mul_mat_q8_0_ref(Wq_q8, ref_xn, ref_q, proj, embd);
    mul_mat_q8_0_ref(Wk_q8, ref_xn, ref_k, proj, embd);
    mul_mat_q8_0_ref(Wv_q8, ref_xn, ref_v, proj, embd);

    // 7-8. ROPE
    std::vector<float> ref_qr, ref_kr;
    rope_ref(ref_q, ref_qr, proj, rope_pos, freq_base, freq_scale);
    rope_ref(ref_k, ref_kr, proj, rope_pos, freq_base, freq_scale);

    // 9. MUL scalar
    std::vector<float> ref_qs(proj);
    for (int i = 0; i < proj; ++i) ref_qs[i] = ref_qr[i] * h_scale;

    // 10. ADD
    std::vector<float> ref_scores(proj);
    for (int i = 0; i < proj; ++i) ref_scores[i] = ref_qs[i] + ref_kr[i];

    // 11. DIAG_MASK_INF (n_past=31: keep cols 0-31, mask 32-63)
    std::vector<float> ref_masked(proj);
    for (int i = 0; i < proj; ++i) {
        if (i > n_past)
            ref_masked[i] = ref_scores[i] - FLT_MAX;
        else
            ref_masked[i] = ref_scores[i];
    }

    // 12. SOFTMAX
    float max_s = ref_masked[0];
    for (int i = 1; i < proj; ++i) if (ref_masked[i] > max_s) max_s = ref_masked[i];
    std::vector<float> ref_probs(proj);
    float sum_exp = 0.0f;
    for (int i = 0; i < proj; ++i) {
        ref_probs[i] = std::exp(ref_masked[i] - max_s);
        sum_exp += ref_probs[i];
    }
    for (int i = 0; i < proj; ++i) ref_probs[i] /= sum_exp;

    // 13. MUL (elem: probs * v)
    std::vector<float> ref_attn(proj);
    for (int i = 0; i < proj; ++i) ref_attn[i] = ref_probs[i] * ref_v[i];

    // 14. CONT (identity)
    std::vector<float> ref_attnc(ref_attn);

    // 15. ADD bias
    std::vector<float> ref_out(proj);
    for (int i = 0; i < proj; ++i) ref_out[i] = ref_attnc[i] + h_bias[i];

    // -- upload ---------------------------------------------------------------
    buffer->iface.clear(buffer, 0);
    buffer->iface.set_tensor(buffer, t_embed, &h_embed[0], 0, embed_bytes);
    buffer->iface.set_tensor(buffer, t_tok,   &h_tok,       0, tok_bytes);
    buffer->iface.set_tensor(buffer, t_wnorm, &h_wnorm[0],  0, embd_bytes);
    buffer->iface.set_tensor(buffer, t_Wq,    &Wq_q8[0],    0, q8_bytes);
    buffer->iface.set_tensor(buffer, t_Wk,    &Wk_q8[0],    0, q8_bytes);
    buffer->iface.set_tensor(buffer, t_Wv,    &Wv_q8[0],    0, q8_bytes);
    buffer->iface.set_tensor(buffer, t_scale, &h_scale,      0, scalar_bytes);
    buffer->iface.set_tensor(buffer, t_bias,  &h_bias[0],    0, proj_bytes);
    buffer->iface.set_tensor(buffer, t_pos,   &h_pos,        0, pos_bytes);

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

    // Softmax sum (should be ~1.0 over all 64, but only 32 are non-zero)
    std::vector<float> h_probs(proj, 0.0f);
    buffer->iface.get_tensor(buffer, t_probs, &h_probs[0], 0, proj_bytes);
    float gpu_sum = 0.0f;
    int masked_count = 0;
    for (int i = 0; i < proj; ++i) {
        gpu_sum += h_probs[i];
        if (h_probs[i] < 1e-30f) masked_count++;
    }

    std::printf("end-to-end pipeline results:\n");
    std::printf("  vocab=%d embd=%d proj=%d token=%d rope_pos=%d n_past=%d\n",
                vocab, embd, proj, token_id, rope_pos, n_past);
    std::printf("  max_err=%.6e  softmax_sum=%.9f  masked=%d/%d\n",
                max_err, gpu_sum, masked_count, proj);

    bool pass = (max_err < 1e-3f) &&
                (std::fabs(gpu_sum - 1.0f) < 1e-4f) &&
                (masked_count == proj - n_past - 1);

    std::printf("end-to-end pipeline: %s\n", pass ? "PASS" : "FAIL");

    buffer->iface.free_buffer(buffer);
    backend->iface.free(backend);
    ggml_free(gctx);

    std::printf("ggml-cuda8-e2e-smoke: %s\n", pass ? "SUCCESS" : "FAIL");
    return pass ? 0 : 1;
}
