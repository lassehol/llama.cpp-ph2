// ggml/src/ggml-cuda8/ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke.cpp
// G17A: real GGML graph-builder Q8_0 x F32 vector smoke through CUDA8 graph_compute.
//
// This intentionally uses a simple synthetic Q8_0 matrix:
//   - d = 1.0 encoded as fp16 bits 0x3c00
//   - qs[] contains small integer values
//
// That avoids needing a full host-side quantizer while still exercising:
//   real ggml_mul_mat(Q8_0, F32)
//     -> ggml_build_forward_expand
//     -> backend-owned CUDA8 buffer residency
//     -> ggml_backend_i.graph_compute
//     -> CUDA8 Q8_0 x F32 vector dispatcher

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
    int32_t exp = (int32_t) ((f >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = f & 0x007FFFFFu;

    if (exp <= 0) {
        if (exp < -10) {
            return (uint16_t) sign;
        }

        mant |= 0x00800000u;
        const int shift = 14 - exp;
        uint32_t half_mant = mant >> shift;

        if ((mant >> (shift - 1)) & 1u) {
            half_mant += 1u;
        }

        return (uint16_t) (sign | half_mant);
    }

    if (exp >= 31) {
        return (uint16_t) (sign | 0x7C00u);
    }

    uint32_t half = sign | ((uint32_t) exp << 10) | (mant >> 13);

    if (mant & 0x00001000u) {
        half += 1u;
    }

    return (uint16_t) half;
}

static float fp16_bits_to_fp32(uint16_t h) {
    const uint32_t h_exp  = h & 0x7C00u;
    const uint32_t h_sig  = h & 0x03FFu;
    const uint32_t h_sign = h & 0x8000u;

    uint32_t f_sign = h_sign << 16;
    uint32_t f_exp;
    uint32_t f_sig;

    if (h_exp == 0) {
        if (h_sig == 0) {
            union { uint32_t u; float f; } out;
            out.u = f_sign;
            return out.f;
        }

        uint32_t sig = h_sig;
        int exp = -14;

        while ((sig & 0x0400u) == 0) {
            sig <<= 1;
            --exp;
        }

        sig &= 0x03FFu;

        f_exp = (uint32_t) (exp + 127) << 23;
        f_sig = sig << 13;

        union { uint32_t u; float f; } out;
        out.u = f_sign | f_exp | f_sig;
        return out.f;
    }

    if (h_exp == 0x7C00u) {
        f_exp = 0xFFu << 23;
        f_sig = h_sig << 13;

        union { uint32_t u; float f; } out;
        out.u = f_sign | f_exp | f_sig;
        return out.f;
    }

    const int exp = (int) (h_exp >> 10) - 15;

    f_exp = (uint32_t) (exp + 127) << 23;
    f_sig = h_sig << 13;

    union { uint32_t u; float f; } out;
    out.u = f_sign | f_exp | f_sig;
    return out.f;
}


static bool expect_resident(
    const char * label,
    const ggml_tensor * t,
    size_t bytes,
    ggml_backend_buffer_t expected,
    size_t expected_offset
) {
    ggml_backend_buffer_t owner = NULL;
    size_t offset = 0;

    if (!ggml_cuda8_ggml_tensor_is_device_resident(t, bytes, &owner, &offset)) {
        std::fprintf(stderr, "%s: residency not found\n", label);
        return false;
    }

    if (owner != expected || offset != expected_offset) {
        std::fprintf(stderr,
            "%s: residency mismatch offset=%zu expected=%zu\n",
            label, offset, expected_offset);
        return false;
    }

    std::printf("%s residency PASS offset=%zu\n", label, offset);
    return true;
}

static void force_2d_q8_0_data_layout(
    ggml_tensor * t,
    int64_t cols,
    int64_t rows,
    void * data
) {
    const int64_t blocks_per_row =
        (cols + GGML_CUDA8_QK8_0 - 1) / GGML_CUDA8_QK8_0;

    t->type = GGML_TYPE_Q8_0;

    t->ne[0] = cols;
    t->ne[1] = rows;
    t->ne[2] = 1;
    t->ne[3] = 1;

    // For quantized tensors nb[0] is one quant block, not one scalar.
    t->nb[0] = sizeof(ggml_cuda8_q8_0_block);
    t->nb[1] = (size_t) blocks_per_row * sizeof(ggml_cuda8_q8_0_block);
    t->nb[2] = (size_t) rows * t->nb[1];
    t->nb[3] = t->nb[2];

    t->data = data;
}

static void force_1d_f32_data_layout(
    ggml_tensor * t,
    int64_t n,
    void * data
) {
    t->type = GGML_TYPE_F32;

    t->ne[0] = n;
    t->ne[1] = 1;
    t->ne[2] = 1;
    t->ne[3] = 1;

    t->nb[0] = sizeof(float);
    t->nb[1] = (size_t) n * sizeof(float);
    t->nb[2] = t->nb[1];
    t->nb[3] = t->nb[1];

    t->data = data;
}

static void fill_f32_matrix(
    std::vector<float> & A,
    int rows,
    int cols
) {
    A.resize((size_t) rows * cols);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const int v = (r * 17 + c * 31 + 7) % 251;
            A[(size_t) r * cols + c] = ((float) v - 125.0f) * 0.003f;
        }
    }
}

static void pack_q8_0(
    const std::vector<float> & A,
    std::vector<ggml_cuda8_q8_0_block> & Aq,
    int rows,
    int cols
) {
    const int blocks_per_row =
        (cols + GGML_CUDA8_QK8_0 - 1) / GGML_CUDA8_QK8_0;

    Aq.resize((size_t) rows * blocks_per_row);

    for (int r = 0; r < rows; ++r) {
        for (int ib = 0; ib < blocks_per_row; ++ib) {
            ggml_cuda8_q8_0_block & block =
                Aq[(size_t) r * blocks_per_row + ib];

            float max_abs = 0.0f;

            for (int k = 0; k < GGML_CUDA8_QK8_0; ++k) {
                const int c = ib * GGML_CUDA8_QK8_0 + k;

                float v = 0.0f;
                if (c < cols) {
                    v = A[(size_t) r * cols + c];
                }

                max_abs = std::max(max_abs, std::fabs(v));
            }

            float d = 0.0f;
            float id = 0.0f;

            if (max_abs > 0.0f) {
                d = max_abs / 127.0f;
                id = 127.0f / max_abs;
            }

            block.d = fp32_to_fp16_bits(d);

            for (int k = 0; k < GGML_CUDA8_QK8_0; ++k) {
                const int c = ib * GGML_CUDA8_QK8_0 + k;

                float v = 0.0f;
                if (c < cols) {
                    v = A[(size_t) r * cols + c];
                }

                int q = 0;

                if (max_abs > 0.0f) {
                    q = (int) std::floor(v * id + (v >= 0.0f ? 0.5f : -0.5f));
                }

                if (q > 127) {
                    q = 127;
                }

                if (q < -127) {
                    q = -127;
                }

                block.qs[k] = (int8_t) q;
            }
        }
    }
}

static void fill_x(
    std::vector<float> & x
) {
    for (size_t i = 0; i < x.size(); ++i) {
        const int v = ((int) i * 13 + 5) % 19;
        x[i] = ((float) v - 9.0f) * 0.03125f;
    }
}

static void cpu_ref_q8_mmv(
    const std::vector<ggml_cuda8_q8_0_block> & Aq,
    const std::vector<float> & x,
    std::vector<float> & y,
    int rows,
    int cols
) {
    const int blocks_per_row =
        (cols + GGML_CUDA8_QK8_0 - 1) / GGML_CUDA8_QK8_0;

    for (int r = 0; r < rows; ++r) {
        float sum = 0.0f;

        const ggml_cuda8_q8_0_block * row_blocks =
            &Aq[(size_t) r * blocks_per_row];

        for (int ib = 0; ib < blocks_per_row; ++ib) {
            const ggml_cuda8_q8_0_block & block = row_blocks[ib];

            const float d = fp16_bits_to_fp32(block.d);

            float block_sum = 0.0f;
            const int base_col = ib * GGML_CUDA8_QK8_0;

            for (int k = 0; k < GGML_CUDA8_QK8_0; ++k) {
                const int c = base_col + k;
                if (c < cols) {
                    block_sum += ((float) block.qs[k]) * x[c];
                }
            }

            sum += d * block_sum;
        }

        y[r] = sum;
    }
}

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    std::printf("ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke: starting\n");

    const int rows = 64;
    const int cols = 64;

    const int blocks_per_row =
        (cols + GGML_CUDA8_QK8_0 - 1) / GGML_CUDA8_QK8_0;

    const size_t bytes_Aq =
        (size_t) rows * blocks_per_row * sizeof(ggml_cuda8_q8_0_block);
    const size_t bytes_x =
        (size_t) cols * sizeof(float);
    const size_t bytes_y =
        (size_t) rows * sizeof(float);

    std::vector<uint8_t> ggml_mem(4 * 1024 * 1024);

    struct ggml_init_params params;
    std::memset(&params, 0, sizeof(params));
    params.mem_size = ggml_mem.size();
    params.mem_buffer = ggml_mem.data();
    params.no_alloc = false;

    ggml_context * gctx = ggml_init(params);
    if (gctx == NULL) {
        std::fprintf(stderr, "ggml_init failed\n");
        return 1;
    }

    ggml_tensor * Aq = ggml_new_tensor_2d(gctx, GGML_TYPE_Q8_0, cols, rows);
    ggml_tensor * x  = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, cols);
    ggml_tensor * y  = ggml_mul_mat(gctx, Aq, x);

    if (Aq == NULL || x == NULL || y == NULL) {
        std::fprintf(stderr, "ggml graph tensor creation failed\n");
        ggml_free(gctx);
        return 1;
    }

    ggml_cgraph * graph = ggml_new_graph(gctx);
    if (graph == NULL) {
        std::fprintf(stderr, "ggml_new_graph failed\n");
        ggml_free(gctx);
        return 1;
    }

    ggml_build_forward_expand(graph, y);

    std::printf("real ggml graph n_nodes=%d n_leafs=%d\n",
        graph->n_nodes, graph->n_leafs);

    for (int i = 0; i < graph->n_nodes; ++i) {
        std::printf("real graph node %d op=%d\n",
            i,
            graph->nodes[i] ? (int) graph->nodes[i]->op : -1);
    }

    if (graph->n_nodes != 1) {
        std::fprintf(stderr, "expected 1 graph node, got %d\n", graph->n_nodes);
        ggml_free(gctx);
        return 1;
    }

    ggml_backend_t backend = ggml_cuda8_ggml_backend_init(0);
    if (backend == NULL) {
        std::fprintf(stderr, "backend init failed\n");
        ggml_free(gctx);
        return 1;
    }

    if (!ggml_cuda8_ggml_backend_is_cuda8(backend) ||
        backend->iface.graph_compute == NULL) {
        std::fprintf(stderr, "backend identity/graph_compute check failed\n");
        backend->iface.free(backend);
        ggml_free(gctx);
        return 1;
    }

    std::printf("backend name: %s\n", backend->iface.get_name(backend));

    ggml_backend_buffer_type_t buft =
        ggml_cuda8_ggml_backend_get_default_buffer_type(backend);

    if (buft == NULL) {
        std::fprintf(stderr, "default buffer type NULL\n");
        backend->iface.free(backend);
        ggml_free(gctx);
        return 1;
    }

    std::printf("backend default buffer type: %s\n", buft->iface.get_name(buft));

    const size_t off_Aq = 0;
    const size_t off_x  = 8192;
    const size_t off_y  = 12288;
    const size_t total_size = 16384;

    ggml_backend_buffer_t buffer = buft->iface.alloc_buffer(buft, total_size);
    if (buffer == NULL) {
        std::fprintf(stderr, "buffer alloc failed\n");
        backend->iface.free(backend);
        ggml_free(gctx);
        return 1;
    }

    void * base = buffer->iface.get_base(buffer);
    if (base == NULL) {
        std::fprintf(stderr, "buffer base NULL\n");
        buffer->iface.free_buffer(buffer);
        backend->iface.free(backend);
        ggml_free(gctx);
        return 1;
    }

    std::printf("backend-owned q8_0 mmv buffer size: %zu\n", buffer->size);
    std::printf("buffer base:                         %p\n", base);

    uint8_t * base_u8 = (uint8_t *) base;

    force_2d_q8_0_data_layout(Aq, cols, rows, base_u8 + off_Aq);
    force_1d_f32_data_layout(x,  cols,       base_u8 + off_x);
    force_1d_f32_data_layout(y,  rows,       base_u8 + off_y);

    if (buffer->iface.init_tensor(buffer, Aq) != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, x)  != GGML_STATUS_SUCCESS ||
        buffer->iface.init_tensor(buffer, y)  != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "init_tensor failed\n");
        buffer->iface.free_buffer(buffer);
        backend->iface.free(backend);
        ggml_free(gctx);
        return 1;
    }

    if (!expect_resident("Aq", Aq, bytes_Aq, buffer, off_Aq) ||
        !expect_resident("x",  x,  bytes_x,  buffer, off_x)  ||
        !expect_resident("y",  y,  bytes_y,  buffer, off_y)) {
        buffer->iface.free_buffer(buffer);
        backend->iface.free(backend);
        ggml_free(gctx);
        return 1;
    }

    std::vector<float> A_f32;
    std::vector<ggml_cuda8_q8_0_block> Aq_host;
    std::vector<float> x_host(cols);
    std::vector<float> y_ref(rows, 0.0f);
    std::vector<float> y_out(rows, 0.0f);

    fill_f32_matrix(A_f32, rows, cols);
    pack_q8_0(A_f32, Aq_host, rows, cols);
    fill_x(x_host);
    cpu_ref_q8_mmv(Aq_host, x_host, y_ref, rows, cols);

    buffer->iface.clear(buffer, 0);
    buffer->iface.set_tensor(buffer, Aq, &Aq_host[0], 0, bytes_Aq);
    buffer->iface.set_tensor(buffer, x,  &x_host[0],  0, bytes_x);

    enum ggml_status status = backend->iface.graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "backend graph_compute returned %d\n", (int) status);
        buffer->iface.free_buffer(buffer);
        backend->iface.free(backend);
        ggml_free(gctx);
        return 1;
    }

    buffer->iface.get_tensor(buffer, y, &y_out[0], 0, bytes_y);

    double max_abs_err = 0.0;
    double max_rel_err = 0.0;

    for (int i = 0; i < rows; ++i) {
        const double r = (double) y_ref[i];
        const double g = (double) y_out[i];

        const double abs_err = std::fabs(g - r);
        const double rel_err = abs_err / std::max(1e-9, std::fabs(r));

        if (abs_err > max_abs_err) max_abs_err = abs_err;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }

    std::printf("real graph-builder packed Q8_0xF32_VEC max_abs_err=%.9g max_rel_err=%.9g\n",
        max_abs_err,
        max_rel_err);

    if (max_abs_err > 2e-4 && max_rel_err > 2e-4) {
        std::fprintf(stderr, "real graph-builder packed Q8_0xF32_VEC verification FAIL\n");
        buffer->iface.free_buffer(buffer);
        backend->iface.free(backend);
        ggml_free(gctx);
        return 1;
    }

    std::printf("real graph-builder packed Q8_0xF32_VEC verification PASS\n");

    buffer->iface.free_buffer(buffer);
    backend->iface.free(backend);
    ggml_free(gctx);

    std::printf("ggml-cuda8-ggml-backend-graph-builder-q8_0-mmv-smoke: SUCCESS\n");
    return 0;
}
